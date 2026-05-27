"""SWE-bench specialization of the OpenClaw agent condition.

Adds:
  - SWE-bench "fixing a bug" framing via ``_build_message``.
  - Patch collection via ``_collect_artifacts`` — runs ``git diff --cached``
    in the workspace after the agent declares DONE, strips OC's scaffold
    files so they don't bleed into the resulting patch.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition

logger = logging.getLogger(__name__)


# OpenClaw drops these files into the workspace at agent startup. They are not
# part of the SWE-bench repo and must be stripped before diffing or the
# resulting patch will be rejected by run_evaluation.
_OC_SCAFFOLD_FILES = (
    "AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md "
    "BOOT.md MEMORY.md fix.patch"
)
_OC_SCAFFOLD_DIRS = ".openclaw memory sessions"


class SWEBenchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw agent driving a SWE-bench instance container."""

    agent_id = "swebench"

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "You are fixing a bug. The repository is in the current working "
            "directory. Read the issue, modify the source files to fix it, then "
            "reply with the single word DONE.\n\n"
            f"Issue:\n{instance_data['problem_statement']}"
        )

    def _collect_artifacts(
        self,
        container_id: Optional[str],
        env_info: Dict,
        run_dir: Path,
        workspace_root: str,
    ) -> Optional[str]:
        """Collect the agent's edits as a unified diff (via git).

        With the bridge-routed sandbox the agent's edits land in
        ``env_info["workspace_host_path"]`` on the host (bind-mounted at
        /testbed in the task container). Running git on the host is much
        simpler than inside OC, where bind-mount ownership trips git's
        safe.directory check AND root can't write into the host user's
        .git/objects. Fall back to docker-exec inside the container only
        when the host path is missing.
        """
        ws_host_path = env_info.get("workspace_host_path") if env_info else None
        if ws_host_path and os.path.isdir(ws_host_path):
            return self._collect_patch_host_mode(ws_host_path)
        return self._collect_patch_docker_mode(container_id, workspace_root)

    @staticmethod
    def _collect_patch_host_mode(ws: str) -> str:
        # Scaffold cleanup is best-effort because some scaffolds are written
        # by OC running as root in the sandbox container (we're the host user
        # here) and rmtree can't remove root-owned files.
        for f in _OC_SCAFFOLD_FILES.split():
            try:
                os.remove(os.path.join(ws, f))
            except (FileNotFoundError, PermissionError):
                pass
        for d in _OC_SCAFFOLD_DIRS.split():
            shutil.rmtree(os.path.join(ws, d), ignore_errors=True)
        # SWE-bench's install step does `git commit -am SWE-bench` after
        # install, so HEAD is the post-setup baseline. `git diff HEAD`
        # captures exactly the agent's edits to tracked files and skips
        # untracked artifacts (build/, *.egg-info, __pycache__, .pytest_cache)
        # — these would otherwise become huge "new file" hunks and trip
        # the eval container's `patch -R` reverse-detection.
        r = subprocess.run(
            ["git", "-C", ws, "diff", "HEAD"],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else ""

    def _collect_patch_docker_mode(
        self,
        container_id: Optional[str],
        ws: str,
    ) -> str:
        # Docker mode. OC runs as root inside the container, but /testbed is a
        # bind-mount of a host directory whose files are owned by the host
        # user — git treats this as "dubious ownership" and silently refuses
        # to operate, returning empty diff. `-c safe.directory=*` whitelists
        # all paths so diff just works regardless of fs ownership.
        if container_id is None:
            return ""
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {_OC_SCAFFOLD_FILES} && "
             f"rm -rf {_OC_SCAFFOLD_DIRS}"],
            check=False,
        )
        r = self._exec(container_id,
                       ["git", "-c", "safe.directory=*", "-C", ws, "diff", "HEAD"],
                       check=False)
        return r.stdout if r.returncode == 0 else ""
