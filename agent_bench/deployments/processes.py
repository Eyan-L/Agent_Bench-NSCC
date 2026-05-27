from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping
from urllib import error, request


class ProcessError(RuntimeError):
    """Raised when a managed process fails during launch or readiness."""


def wait_for_http_ready(
    *,
    url: str,
    headers: Mapping[str, str] | None = None,
    proc: subprocess.Popen[bytes] | None = None,
    timeout_seconds: int = 600,
    poll_seconds: float = 2.0,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            details = f"; see {log_path}" if log_path else ""
            raise ProcessError(f"process exited before readiness{details}")
        if http_ready(url, headers=headers):
            return
        time.sleep(poll_seconds)
    details = f"; see {log_path}" if log_path else ""
    raise ProcessError(f"endpoint was not ready after {timeout_seconds}s{details}")


def http_ready(url: str, *, headers: Mapping[str, str] | None = None, timeout: int = 2) -> bool:
    req = request.Request(url)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (OSError, error.URLError, error.HTTPError):
        return False


def terminate_process_group(proc: subprocess.Popen[bytes], *, graceful_timeout: int = 30) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=graceful_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=10)


def tail_text(path: Path, *, lines: int = 80) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])
