from __future__ import annotations

import logging
from typing import Optional, Dict
from pathlib import Path
from agent_bench.deployments.containers import Container

logger = logging.getLogger(__name__)

from agent_bench.benchmarks.swebench.hermes.app import SWEBenchHermesHarness
from agent_bench.benchmarks.swe_atlas_qa.app import SWEAtlasQAHarnessMixin


class SWEAtlasQAHermesHarness(SWEAtlasQAHarnessMixin, SWEBenchHermesHarness):
    """Hermes harness for SWE-Atlas QA benchmark."""

    backend_id = "swe_atlas_qa"
    default_harness_config = "profiles/harnesses/hermes_swe_atlas_qa.yaml"

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        *,
        container: Container,
        hm_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ):
        from pathlib import Path
        dummy_path = Path(".")
        super().__init__(
            benchmark_data_root=benchmark_data_root or dummy_path,
            hm_image=hm_image,
            workspace_base=workspace_base,
            container=container,
        )
        if benchmark_data_root is None:
            self.benchmark_data_root = None

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,
        base_commit: str,
        container_image: str,
        workspace_root: str = "/testbed",
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        self._current_instance_data = instance_data
        self._current_repo_id = repo_id
        self._current_base_commit = base_commit
        return super().prepare_environment(
            instance_id=instance_id,
            repo_id=repo_id,
            base_commit=base_commit,
            container_image=container_image,
            workspace_root=workspace_root,
            run_id=run_id,
            instance_data=instance_data,
        )

    def _prime_workspace(self, task_image: str, ws_host: Path) -> None:
        import subprocess
        import os
        inst = getattr(self, "_current_instance_data", None) or {}
        repo_url = inst.get("repo") or inst.get("repository_url")
        base_commit = inst.get("repository_base_commit") or getattr(self, "_current_base_commit", "main")

        if not repo_url:
            logger.warning("No repository_url found in instance_data; skipping clone.")
            return

        local_repo_path = os.environ.get("AB_REPO_ROOT") or os.getcwd()
        if "hyscale-lab/Agent_Bench" in repo_url or "Agent_Bench" in repo_url:
            logger.info(f"Detected local repository. Cloning locally from {local_repo_path} to avoid auth prompts...")
            repo_source = local_repo_path
        else:
            repo_source = repo_url

        logger.info(f"Cloning {repo_source} into host workspace {ws_host}...")
        try:
            # Run git clone
            subprocess.run(
                ["git", "clone", repo_source, "."],
                cwd=ws_host,
                check=True,
                capture_output=True,
                text=True,
            )
            # Run git checkout
            logger.info(f"Checking out base commit {base_commit}...")
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=ws_host,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            logger.exception("Failed to clone/checkout codebase repo on host")
            raise RuntimeError(f"Failed to clone/checkout codebase repo: {e}")
