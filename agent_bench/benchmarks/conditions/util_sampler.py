"""Resource-utilization sampler - sidecar around an in-flight condition run.

Spawns ``agent_bench/telemetry/util_sampler.py`` as a child process. Each sample lands as
one JSON line in ``<run_dir>/util.jsonl`` (CPU/mem/loadavg/GPU, plus per-PID-
group CPU split when llm_pid/client_pid/gateway_pid are provided).

Benchmark-agnostic — used by both ``OpenClawAgentCondition`` (SWE-bench, TB)
and the ``TerminalBenchReactCondition``. The condition-side context manager
lives with benchmark utilities; the sampler process itself lives in telemetry.

Toggle off with ``OC_UTIL_SAMPLE=0``; tune cadence with
``OC_UTIL_SAMPLE_INTERVAL`` (seconds, default 1.0).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_UTIL_SAMPLER_SCRIPT = Path(__file__).resolve().parents[2] / "telemetry" / "util_sampler.py"


class UtilSampler:
    """Context manager that runs the sampler script while the agent runs."""

    def __init__(
        self,
        run_dir: Path,
        *,
        llm_pid: Optional[int] = None,
        client_pid: Optional[int] = None,
        gateway_pid: Optional[int] = None,
    ):
        self.run_dir = run_dir
        self.proc: Optional[subprocess.Popen] = None
        self.enabled = (
            os.environ.get("OC_UTIL_SAMPLE", "1") != "0"
            and _UTIL_SAMPLER_SCRIPT.exists()
        )
        try:
            self.interval = float(os.environ.get("OC_UTIL_SAMPLE_INTERVAL", "1.0"))
        except ValueError:
            self.interval = 1.0

        if llm_pid is None:
            try:
                llm_pid = int(os.environ.get("OC_LLM_PID") or 0) or None
            except ValueError:
                llm_pid = None
        if client_pid is None:
            client_pid = os.getpid()
        self.llm_pid = llm_pid
        self.client_pid = client_pid
        self.gateway_pid = gateway_pid

    def __enter__(self) -> "UtilSampler":
        if not self.enabled:
            return self
        out_path = self.run_dir / "util.jsonl"
        argv = [
            sys.executable, str(_UTIL_SAMPLER_SCRIPT),
            "--output", str(out_path),
            "--interval", str(self.interval),
        ]
        if self.llm_pid:
            argv += ["--llm-pid", str(self.llm_pid)]
        if self.client_pid:
            argv += ["--client-pid", str(self.client_pid)]
        if self.gateway_pid:
            argv += ["--gateway-pid", str(self.gateway_pid)]
        try:
            self.proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            logger.info(
                "  util sampler pid=%s -> %s (interval=%.2fs, "
                "llm=%s client=%s gateway=%s)",
                self.proc.pid, out_path.name, self.interval,
                self.llm_pid, self.client_pid, self.gateway_pid,
            )
        except Exception as e:
            logger.warning("util sampler launch failed: %s", e)
            self.proc = None
        return self

    def __exit__(self, *exc) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except Exception as e:
            logger.warning("util sampler stop failed: %s", e)
