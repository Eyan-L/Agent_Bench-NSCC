"""RoadmapBench harness specialization over Terminal-Bench containers."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

from agent_bench.benchmarks.terminal_bench.app import (
    TerminalBenchOpenClawHarness,
)
from agent_bench.deployments.containers.base import ContainerHandle

logger = logging.getLogger(__name__)


class RoadmapBenchOpenClawHarness(TerminalBenchOpenClawHarness):
    """Add Roadmap lifecycle policy while reusing the task runtime adapter."""

    def _runtime_policy_env(self, agent_id: str) -> Dict[str, str]:
        if agent_id != "roadmap_bench":
            return {}

        version = str(os.environ.get("AB_AGENT_RUNTIME_VERSION") or "").lstrip("v")
        if version != "2026.9.2":
            return {}
        return {
            "OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY": "continue",
            "OPENCLAW_AGENT_BENCH_EMPTY_STOP_RETRIES": "4",
            "OPENCLAW_AGENT_BENCH_LENGTH_NO_PROGRESS_RETRIES": "4",
            "OPENCLAW_AGENT_BENCH_LOOP_VETO_RECOVERY_PROBE": "1",
        }

    def _start_session_exporter(
        self,
        *,
        oc_handle: ContainerHandle,
        state_dir: Path,
        agent_id: str,
    ) -> Optional[int]:
        version = str(os.environ.get("AB_AGENT_RUNTIME_VERSION") or "").lstrip("v")
        if agent_id != "roadmap_bench" or version != "2026.9.2":
            return None

        source = Path(__file__).resolve().parent / "openclaw" / "sqlite_session_exporter.mjs"
        target = state_dir / "agent-bench" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        ready_path = target.parent / "sqlite-session-exporter.ready.json"
        ready_path.unlink(missing_ok=True)
        pid = self.container.exec_background(
            oc_handle,
            [
                "bash",
                "-c",
                "mkdir -p /root/.openclaw/logs; "
                "exec node /root/.openclaw/agent-bench/sqlite_session_exporter.mjs "
                f"/root/.openclaw/agents/{agent_id}/agent/openclaw-agent.sqlite "
                f"/root/.openclaw/agents/{agent_id}/sessions "
                "/root/.openclaw/agent-bench/sqlite-session-exporter.ready.json "
                ">/root/.openclaw/logs/sqlite-session-exporter.log 2>&1 "
                "</dev/null",
            ],
        )
        deadline = time.monotonic() + 30
        while not ready_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.25)
        if not ready_path.is_file():
            raise RuntimeError(
                "OpenClaw SQLite session exporter did not become ready within 30s"
            )
        logger.info("  started Roadmap SQLite session exporter pid=%s", pid)
        return pid
