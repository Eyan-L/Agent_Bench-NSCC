"""Roadmap-specific Hermes task prompt and frozen Git candidate extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition
from agent_bench.deployments.containers import ContainerHandle

_EXCLUDED_FILES = frozenset({
    "AGENTS.md", "BOOTSTRAP.md", "HEARTBEAT.md", "IDENTITY.md",
    "MEMORY.md", "SOUL.md", "TOOLS.md", "USER.md", "sessions.jsonl",
})
_EXCLUDED_DIRS = frozenset({".openclaw", ".hermes"})


class RoadmapBenchHermesCondition(HermesAgentCondition):
    """Hermes lead engineer with native synchronous delegation."""

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "You are the lead engineer responsible for implementing a multi-target "
            "software roadmap in the current repository. Read the roadmap and "
            "inspect the codebase. Identify meaningful, independent workstreams "
            "and their dependencies. When several independent workstreams exist, "
            "prefer delegating bounded tasks with delegate_task so they can run "
            "in parallel; do not delegate trivial or tightly coupled work. "
            "You own the overall plan, coordination, integration, cross-target "
            "constraints, and final validation. Review child results and validate "
            "the complete implementation. Reply DONE only when the work is complete. "
            "Do not read or modify /tests; it belongs to the verifier.\n\n"
            f"Roadmap:\n{instance_data['instruction']}"
        )

    def run(self, instance_data, run_dir, seed, workspace_root, container_id, env_info=None):
        env_info = env_info or {}
        harness = env_info.get("_harness")
        if harness is None or not env_info.get("task_container_id"):
            raise RuntimeError("Roadmap Hermes requires a live task runtime for Git baseline")
        self._run_dir = Path(run_dir)
        self._baseline = self._capture_baseline(env_info)
        (self._run_dir / "workspace_baseline.json").write_text(
            json.dumps(self._baseline, indent=2) + "\n"
        )
        result = super().run(
            instance_data, run_dir, seed, workspace_root, container_id, env_info,
        )
        result["agent_lifecycle"] = {
            "workspaceQuiesced": bool(env_info.get("workspace_quiesced")),
            "terminalEvidence": "hermes_oneshot_exit",
        }
        return result

    def _task_exec(self, env_info: Dict, argv: list[str], *, workdir: Optional[str] = None):
        harness = env_info["_harness"]
        name = env_info["task_container_name"]
        handle = ContainerHandle(
            id=env_info["task_container_id"], name=name,
            backend=env_info["backend"],
        )
        return harness.container.exec(
            handle, argv, check=False, timeout=120, workdir=workdir,
        )

    def _capture_baseline(self, env_info: Dict) -> Dict:
        workspace = env_info["task_workspace_root"]
        git = ["git", "-c", f"safe.directory={workspace}", "-C", workspace]
        head = self._task_exec(env_info, [*git, "rev-parse", "--verify", "HEAD"])
        tree = self._task_exec(env_info, [*git, "rev-parse", "HEAD^{tree}"])
        status = self._task_exec(env_info, [*git, "status", "--porcelain=v1", "--untracked-files=all"])
        if any(result.returncode != 0 for result in (head, tree, status)):
            raise RuntimeError("Roadmap task Git baseline is unavailable in the task runtime")
        return {
            "git_head": head.stdout.strip(),
            "git_tree": tree.stdout.strip(),
            "clean": not bool(status.stdout.strip()),
            "status": status.stdout.splitlines(),
        }

    def _collect_artifacts(self, env_info: Dict, container_id: str, workspace_root: str) -> str:
        harness = env_info["_harness"]
        # Export the Hermes transcript into its state bind before stopping the
        # agent. The inherited condition later calls _collect_sessions again;
        # the override below keeps that second call a no-op.
        HermesAgentCondition._collect_sessions(
            self, env_info, container_id, self._run_dir, harness.state_dir_in_agent,
        )
        harness.quiesce_for_scoring(env_info)
        workspace = env_info["task_workspace_root"]
        baseline = self._baseline["git_head"]
        git = ["git", "-c", f"safe.directory={workspace}", "-C", workspace]
        excludes = [f":(exclude){name}" for name in sorted(_EXCLUDED_FILES)]
        excludes += [f":(exclude){name}/**" for name in sorted(_EXCLUDED_DIRS)]
        tracked = self._task_exec(
            env_info, [*git, "diff", "--binary", baseline, "--", ".", *excludes],
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"Roadmap candidate diff failed: {tracked.stderr}")
        untracked = self._task_exec(
            env_info, [*git, "ls-files", "--others", "--exclude-standard", "-z"],
        )
        if untracked.returncode != 0:
            raise RuntimeError(f"Roadmap untracked enumeration failed: {untracked.stderr}")
        parts = [tracked.stdout]
        for relative in untracked.stdout.split("\0"):
            if not relative or Path(relative).parts[0] in (_EXCLUDED_FILES | _EXCLUDED_DIRS):
                continue
            diff = self._task_exec(
                env_info,
                ["git", "-c", f"safe.directory={workspace}",
                 "diff", "--no-index", "--binary", "--", "/dev/null", relative],
                workdir=workspace,
            )
            if diff.returncode not in (0, 1):
                raise RuntimeError(f"Roadmap untracked diff failed for {relative}: {diff.stderr}")
            parts.append(diff.stdout)
        return "".join(parts)

    def _collect_sessions(self, env_info, container_id, run_dir, workspace_root):
        # Already exported before the writer fence in _collect_artifacts.
        return None

    def _collect_bridge_data(self, env_info, container_id, run_dir, workspace_root):
        # The shared implementation points at an OpenClaw-only path.
        return None
