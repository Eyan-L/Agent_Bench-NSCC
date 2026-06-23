from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent_bench.benchmarks.swebench_pro.common import AGENT_ID, TASK_WORKSPACE
from agent_bench.benchmarks.swebench_pro.scoring import image_for_instance, score_patch


class SWEBenchProHarnessMixin:
    """Shared SWE-bench Pro behavior layered onto existing agent harnesses."""

    agent_id = AGENT_ID
    task_workspace = TASK_WORKSPACE
    requires_swebench_package = False
    task_path = ""
    app_agents_rel = "agent_bench/benchmarks/swebench_pro/openclaw/agents"
    app_skills_rel = "agent_bench/benchmarks/swebench_pro/openclaw/skills"

    def _resolve_container_image_for_instance(
        self,
        *,
        instance_id: str,
        container_image: str,
        instance_data: Optional[Dict] = None,
    ) -> str:
        if instance_data:
            return image_for_instance(instance_data)
        if "UNUSED" in container_image:
            raise RuntimeError(
                f"SWE-bench Pro instance {instance_id} needs instance_data to resolve its Docker image"
            )
        return container_image

    def _resolve_task_image(self, container_image: str) -> str:
        if getattr(self, "container").backend_id == "docker":
            return container_image
        raise RuntimeError(
            "SWE-bench Pro currently supports Docker task containers only; "
            "PBS/Singularity support is intentionally deferred."
        )

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Mapping[str, Any]] = None,
        run_dir: Optional[Path] = None,
        **_kw,
    ) -> Dict:
        if patch is None:
            if not container_id:
                return {
                    "status": "ERROR",
                    "test_pass_rate": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "error_message": "run_tests called with patch=None and no container_id",
                    "evaluation_dir": None,
                }
            patch = self.collect_patch(container_id, workspace_root)

        if not patch.strip():
            return {
                "status": "FAIL",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 1,
                "error_message": "Empty patch (agent made no changes)",
                "evaluation_dir": None,
            }
        if instance_data is None:
            return {
                "status": "ERROR",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 0,
                "error_message": "SWE-bench Pro scoring requires instance_data",
                "evaluation_dir": None,
            }
        run_id = run_id or f"swebench-pro-{instance_id}-{os.urandom(3).hex()}"
        return score_patch(
            instance_id=instance_id,
            patch=patch,
            instance_data=instance_data,
            timeout_seconds=timeout_seconds,
            run_id=run_id,
            benchmark_data_root=Path(getattr(self, "benchmark_data_root")),
            run_dir=run_dir,
        )
