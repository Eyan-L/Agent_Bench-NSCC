"""Resource-utilization sampler - sidecar around an in-flight condition run.

Spawns ``agent_bench/telemetry/util_sampler.py`` as a child process. Each sample lands as
one JSON line in ``<run_dir>/util.jsonl`` (CPU/mem/loadavg/GPU, plus per-container
split when llm/client/tools container names are provided).

Benchmark-agnostic — used by both ``OpenClawAgentCondition`` (SWE-bench, TB)
and the ``TerminalBenchReactCondition``. The condition-side context manager
lives with benchmark utilities; the sampler process itself lives in telemetry.

Toggle off with ``AB_TELEMETRY_UTIL_SAMPLE=0``; tune cadence with
``AB_TELEMETRY_UTIL_SAMPLE_INTERVAL`` (seconds, default 1.0).
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
        llm_container: Optional[str] = None,
        client_container: Optional[str] = None,
        tools_container: Optional[str] = None,
    ):
        self.run_dir = run_dir
        self.proc: Optional[subprocess.Popen] = None
        self.enabled = (
            os.environ.get("AB_TELEMETRY_UTIL_SAMPLE", "1") != "0"
            and _UTIL_SAMPLER_SCRIPT.exists()
        )
        try:
            self.interval = float(os.environ.get("AB_TELEMETRY_UTIL_SAMPLE_INTERVAL", "1.0"))
        except ValueError:
            self.interval = 1.0

        if llm_container is None:
            llm_container = (
                os.environ.get("OC_LLM_CONTAINER")
                or os.environ.get("VLLM_CONTAINER_NAME")
                or os.environ.get("LITELLM_CONTAINER_NAME")
            )
        if client_container is None:
            client_container = (
                os.environ.get("OC_CLIENT_CONTAINER")
                or os.environ.get("OC_GATEWAY_CONTAINER")
                or os.environ.get("OC_GATEWAY_NAME")
            )
        if tools_container is None:
            tools_container = (
                os.environ.get("OC_TOOLS_CONTAINER")
                or os.environ.get("OC_TASK_CONTAINER")
                or os.environ.get("TASK_CONTAINER_NAME")
            )
        self.llm_container = llm_container
        self.client_container = client_container
        self.tools_container = tools_container

    def __enter__(self) -> "UtilSampler":
        if not self.enabled:
            return self
        out_path = self.run_dir / "util.jsonl"
        argv = [
            sys.executable, str(_UTIL_SAMPLER_SCRIPT),
            "--output", str(out_path),
            "--interval", str(self.interval),
        ]
        if self.llm_container:
            argv += ["--llm-container", self.llm_container]
        if self.client_container:
            argv += ["--client-container", self.client_container]
        if self.tools_container and self.tools_container != self.client_container:
            argv += ["--tools-container", self.tools_container]
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
                "llm=%s client=%s tools=%s)",
                self.proc.pid, out_path.name, self.interval,
                self.llm_container, self.client_container, self.tools_container,
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
