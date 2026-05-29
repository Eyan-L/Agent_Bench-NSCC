#!/usr/bin/env python3
"""OpenClaw exec server — runs on the HOST.

Receives exec requests from bridge.py (running inside the OC container) over
a unix-domain socket, dispatches them to whatever backend the harness picked
(docker exec, singularity exec, plain local shell, ...), and returns the
result. The wire format mirrors openclaw/infra/exec-host.ts so this is a
straight port of OC's own protocol — just running as a free-standing server
instead of being routed through OC's gateway+node-companion path.

The interception happens at OC's `buildExecSpec` extension point (patched
into the exec-runtime bundle): OC's agent loop generates a tool call, OC's
runtime asks for the argv, the patched code returns a path to bridge.py
inside the container, the bridge talks to this server. OC never knows what
backend actually ran the command — only sees stdout/stderr/exitcode.

To OC, every exec "happens locally"; to us, every exec is a structured
request we route. Adding a new backend == writing a new dispatcher case.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("oc-exec-server")


# ---------------------------------------------------------------------------
# Backends. Each backend turns an ExecHostRequest into a ExecHostRunResult.
# Add new ones here; the harness picks which by name.
# ---------------------------------------------------------------------------


@dataclass
class ExecRequest:
    # Raw argv to execute. The client decides framing (e.g. exec tool sends
    # ["/bin/sh","-lc",cmd]; fs-bridge sends ["sh","-c",script,"openclaw-sandbox-fs",...]).
    command: List[str]
    cwd: Optional[str]
    env: Optional[Dict[str, str]]
    timeout_ms: Optional[int]
    agent_id: Optional[str]
    session_key: Optional[str]
    stdin: Optional[bytes] = None


@dataclass
class ExecResult:
    # Binary-safe — base64 wrap happens at the wire layer.
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool
    success: bool


Backend = Callable[[ExecRequest], ExecResult]


def _run_subprocess(argv: List[str], request: ExecRequest) -> ExecResult:
    """Common subprocess wrapper used by all backends."""
    timeout_s = (request.timeout_ms / 1000.0) if request.timeout_ms else None
    try:
        proc = subprocess.run(
            argv,
            input=request.stdin,
            capture_output=True,
            timeout=timeout_s,
        )
        return ExecResult(
            stdout=proc.stdout or b"",
            stderr=proc.stderr or b"",
            exit_code=proc.returncode,
            timed_out=False,
            success=proc.returncode == 0,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            stdout=(e.stdout if isinstance(e.stdout, bytes) else b""),
            stderr=((e.stderr if isinstance(e.stderr, bytes) else b"")
                    + f"\n[oc-exec-server] timeout after {timeout_s}s".encode("utf-8")),
            exit_code=124,
            timed_out=True,
            success=False,
        )


def docker_backend(container: str) -> Backend:
    """Send each ExecRequest to a pre-started docker container via `docker exec`.

    The request's `command` is forwarded verbatim — clients send either
    `["/bin/sh","-lc",cmd]` (exec tool) or `["sh","-c",script,"openclaw-sandbox-fs",
    ...args]` (fs-bridge plan). We hand it to docker exec with -w/-e flags.

    PATH is deliberately filtered out: the OC runtime passes its own PATH
    (the OC container's, e.g. /usr/local/sbin:...), which would override
    the task container's PATH if forwarded via `docker exec -e PATH=...`.
    The harness sets PATH on the task container at docker-run time (with
    the conda env's bin directory prepended for SWE-bench), so dropping
    PATH from the per-exec env preserves it.
    """
    def run(req: ExecRequest) -> ExecResult:
        argv = ["docker", "exec", "-i"]
        if req.cwd:
            argv += ["-w", req.cwd]
        for key, value in (req.env or {}).items():
            if key == "PATH":
                continue
            argv += ["-e", f"{key}={value}"]
        argv += [container]
        argv += list(req.command)
        return _run_subprocess(argv, req)
    return run


def singularity_backend(instance: str) -> Backend:
    """Send each ExecRequest to a pre-started singularity instance via `singularity exec`.

    PATH is filtered for the same reason as docker_backend: the agent runtime
    may forward its own PATH, but the task instance was started with the
    benchmark image's PATH (including the SWE-bench testbed env).
    """
    def run(req: ExecRequest) -> ExecResult:
        argv = ["singularity", "exec"]
        if req.cwd:
            argv += ["--pwd", req.cwd]
        for key, value in (req.env or {}).items():
            if key == "PATH":
                continue
            argv += ["--env", f"{key}={value}"]
        argv += [f"instance://{instance}"]
        argv += list(req.command)
        return _run_subprocess(argv, req)
    return run


# ---------------------------------------------------------------------------
# Protocol. Mirrors openclaw/infra/exec-host.ts:
#   client → JSONL: {type:"exec", id, nonce, ts, hmac, requestJson}
#   server → JSONL: {type:"exec-res", ok:true, payload:{...}}
#                  | {type:"exec-res", ok:false, error:{...}}
#   HMAC-SHA256(token, "<nonce>:<ts>:<requestJson>")
# ---------------------------------------------------------------------------

_RECENT_NONCES: Dict[str, float] = {}
_NONCE_LOCK = threading.Lock()
_NONCE_TTL_S = 600.0  # reject replays older than 10 min


def _verify_and_record(nonce: str, ts: int, request_json: str,
                       token: str, hmac_hex: str) -> Optional[str]:
    now_ms = time.time() * 1000
    if abs(now_ms - ts) > 5 * 60 * 1000:  # 5 min skew
        return "ts-skew"
    expected = hmac.new(
        token.encode("utf-8"),
        f"{nonce}:{ts}:{request_json}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, hmac_hex):
        return "bad-hmac"
    cutoff = time.time() - _NONCE_TTL_S
    with _NONCE_LOCK:
        for k, v in list(_RECENT_NONCES.items()):
            if v < cutoff:
                del _RECENT_NONCES[k]
        if nonce in _RECENT_NONCES:
            return "replay"
        _RECENT_NONCES[nonce] = time.time()
    return None


def _make_handler(token: str, backend: Backend):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            # Read one JSONL line.
            buf = bytearray()
            while True:
                chunk = self.request.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in chunk:
                    break
            line, _, _ = bytes(buf).partition(b"\n")
            try:
                frame = json.loads(line.decode("utf-8"))
            except Exception as e:
                self._respond_error("bad-json", f"invalid JSON: {e}")
                return
            if frame.get("type") != "exec":
                self._respond_error("bad-type", f"unknown type {frame.get('type')!r}")
                return
            request_json = frame.get("requestJson", "")
            verify_err = _verify_and_record(
                nonce=frame.get("nonce", ""),
                ts=int(frame.get("ts", 0)),
                request_json=request_json,
                token=token,
                hmac_hex=frame.get("hmac", ""),
            )
            if verify_err:
                self._respond_error("auth-failed", f"verification: {verify_err}")
                return
            try:
                raw_req = json.loads(request_json)
            except Exception as e:
                self._respond_error("bad-request", f"invalid requestJson: {e}")
                return
            stdin_b64 = raw_req.get("stdinB64")
            try:
                stdin_bytes = base64.b64decode(stdin_b64) if stdin_b64 else None
            except Exception as e:
                self._respond_error("bad-stdin", f"stdinB64 decode failed: {e}")
                return
            req = ExecRequest(
                command=raw_req.get("command") or [],
                cwd=raw_req.get("cwd"),
                env=raw_req.get("env") or {},
                timeout_ms=raw_req.get("timeoutMs"),
                agent_id=raw_req.get("agentId"),
                session_key=raw_req.get("sessionKey"),
                stdin=stdin_bytes,
            )
            logger.info("exec id=%s cwd=%s cmd=%s",
                        frame.get("id"), req.cwd, req.command[-1][:120] if req.command else "")
            try:
                result = backend(req)
            except Exception as e:
                logger.exception("backend failed")
                self._respond_error("backend-failed", str(e))
                return
            self._respond_ok(result)

        def _respond_ok(self, result: ExecResult) -> None:
            # Send both base64 (binary-safe, preferred by bridge) AND text
            # (best-effort UTF-8 decode for any legacy client). Bridge uses
            # base64 fields when present.
            self._send({
                "type": "exec-res",
                "ok": True,
                "payload": {
                    "stdoutB64": base64.b64encode(result.stdout).decode("ascii"),
                    "stderrB64": base64.b64encode(result.stderr).decode("ascii"),
                    "stdout": result.stdout.decode("utf-8", errors="replace"),
                    "stderr": result.stderr.decode("utf-8", errors="replace"),
                    "exitCode": result.exit_code,
                    "timedOut": result.timed_out,
                    "success": result.success,
                },
            })

        def _respond_error(self, code: str, message: str) -> None:
            self._send({
                "type": "exec-res",
                "ok": False,
                "error": {"code": code, "message": message},
            })

        def _send(self, obj: dict) -> None:
            try:
                self.request.sendall(
                    (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
                )
            except Exception:
                logger.exception("failed to send response")

    return Handler


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(socket_path: str, token: str, backend: Backend) -> None:
    sock_path = Path(socket_path)
    if sock_path.exists():
        sock_path.unlink()
    server = _ThreadingUnixServer(str(sock_path), _make_handler(token, backend))
    # Bridge runs as root inside OC container; socket must be readable from there.
    os.chmod(sock_path, 0o666)
    logger.info(
        "listening on %s (backend=%s)",
        sock_path,
        getattr(backend, "__name__", type(backend).__name__),
    )

    def _sigterm(signum, _frame):
        # `server.shutdown()` from a same-thread signal handler deadlocks
        # (shutdown waits for serve_forever to return, but serve_forever
        # can't progress while the handler is running). Just exit hard —
        # the harness re-creates socket path + token per run, no state to flush.
        logger.info("signal %s received, exiting", signum)
        try:
            sock_path.unlink(missing_ok=True)
        except Exception:
            pass
        os._exit(0)
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        if sock_path.exists():
            sock_path.unlink()


def _build_backend(args) -> Backend:
    if args.backend == "docker":
        if not args.target:
            sys.exit("--target (container name) required for backend=docker")
        b = docker_backend(args.target)
        b.__name__ = f"docker[{args.target}]"
        return b
    if args.backend == "singularity":
        if not args.target:
            sys.exit("--target (instance name) required for backend=singularity")
        b = singularity_backend(args.target)
        b.__name__ = f"singularity[{args.target}]"
        return b
    sys.exit(f"unknown backend: {args.backend}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True, help="Unix socket path to bind")
    ap.add_argument("--token-file", required=True, help="File containing HMAC token")
    ap.add_argument("--backend", required=True, choices=["docker", "singularity"])
    ap.add_argument("--target", required=True,
                    help="docker container name OR singularity instance name")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    token = Path(args.token_file).read_text().strip()
    if not token:
        sys.exit("token file is empty")
    backend = _build_backend(args)
    serve(args.socket, token, backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
