"""Roadmap prompt, Git baseline and frozen candidate for Codex CLI."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import RoadmapBenchOpenClawCondition
from agent_bench.harnesses.codex.condition import CodexAgentCondition
from agent_bench.deployments.containers import ContainerHandle


class RoadmapBenchCodexCondition(CodexAgentCondition):
    _capture_task_workspace_baseline = RoadmapBenchOpenClawCondition._capture_task_workspace_baseline
    _capture_host_workspace_baseline = staticmethod(RoadmapBenchOpenClawCondition._capture_host_workspace_baseline)
    _collect_container_workspace_patch = RoadmapBenchOpenClawCondition._collect_container_workspace_patch

    def _build_message(self, instance_data):
        return (
            "Lead the implementation of this multi-target software roadmap in "
            "the current repository. Inspect the codebase, identify target "
            "workstreams and dependencies, coordinate parallel work where "
            "appropriate, integrate the complete implementation, and validate "
            "all roadmap requirements. Declare DONE only when the integrated "
            "roadmap is complete.\n\n"
            f"Roadmap:\n{instance_data['instruction']}"
        )

    def run(self, instance_data, run_dir, seed, workspace_root, container_id, env_info=None):
        env_info = env_info if env_info is not None else {}
        baseline = self._capture_task_workspace_baseline(env_info, workspace_root)
        if not baseline:
            baseline = self._capture_host_workspace_baseline(Path(env_info["workspace_host_path"]))
        if not baseline or not baseline.get("git_head"):
            raise RuntimeError("Roadmap Codex task Git baseline was not captured")
        env_info["roadmap_workspace_baseline"] = baseline
        env_info["roadmap_workspace_quiesced"] = False
        return super().run(instance_data, run_dir, seed, workspace_root, container_id, env_info)

    def _finalize_runtime(self, env_info, run_dir, workspace_root):
        lifecycle, _ = super()._finalize_runtime(env_info, run_dir, workspace_root)
        baseline = env_info["roadmap_workspace_baseline"]
        patch = self._collect_container_workspace_patch(
            ContainerHandle(env_info["task_container_id"], env_info["task_container_name"], self.container.backend_id),
            workspace_root, baseline_head=baseline["git_head"],
        )
        candidate = run_dir / "candidate_workspace"
        shutil.copytree(env_info["workspace_host_path"], candidate,
                        ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
        (run_dir / "candidate_metadata.json").write_text(json.dumps({
            "baseline": baseline, "workspace_quiesced": True,
            "quiescence": lifecycle, "candidate_workspace": str(candidate),
            "patch_bytes": len(patch.encode()),
        }, indent=2))
        return lifecycle, patch
