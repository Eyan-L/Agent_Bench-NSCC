from __future__ import annotations

import logging
import os
from typing import Optional, Dict, List
from pathlib import Path
from agent_bench.deployments.containers import Container, Mount

logger = logging.getLogger(__name__)

from agent_bench.benchmarks.swebench.openclaw.app import SWEBenchHarness
from agent_bench.benchmarks.swe_atlas_qa.app import SWEAtlasQAHarnessMixin


class SWEAtlasQAHarness(SWEAtlasQAHarnessMixin, SWEBenchHarness):
    """OpenClaw harness for SWE-Atlas QA benchmark."""

    backend_id = "swe_atlas_qa"
    default_harness_config = "profiles/harnesses/openclaw_swe_atlas_qa.yaml"
    app_agents_rel = "agent_bench/benchmarks/swe_atlas_qa/openclaw/agents"
    app_skills_rel = "agent_bench/benchmarks/swe_atlas_qa/openclaw/skills"

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        *,
        container: Container,
        oc_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ):
        from pathlib import Path
        dummy_path = Path(".")
        super().__init__(
            benchmark_data_root=benchmark_data_root or dummy_path,
            container=container,
            oc_image=oc_image,
            workspace_base=workspace_base,
        )
        if benchmark_data_root is None:
            self.benchmark_data_root = None

    def _resolve_task_image(self, container_image: str) -> str:
        """Resolve an Atlas task image for Docker or Singularity."""
        if self.container.backend_id == "docker":
            return container_image

        sif_dir = os.environ.get("AB_DEPLOYMENT_TASK_IMAGE_DIR")
        if not sif_dir:
            raise RuntimeError(
                "AB_DEPLOYMENT_TASK_IMAGE_DIR is required for "
                "SWE-Atlas-QA with Singularity."
            )

        # Example:
        # ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_kovidgoyal_kitty_1.0
        # ->
        # $AB_DEPLOYMENT_TASK_IMAGE_DIR/swe_atlas_QnA_kovidgoyal_kitty_1.0.sif
        tag = container_image.rsplit(":", 1)[-1]
        return str(Path(sif_dir) / f"{tag}.sif")

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

    def _prime_workspace_from_image(self, task_image: str, ws_host: Path) -> None:
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

        try:
            if self.container.backend_id == "singularity":
                # NSCC compute nodes do not require host git/network access.
                # SWE-Atlas task SIFs already contain the prepared repository
                # under /app, including .git metadata at the benchmark commit.
                logger.info(
                    "Priming host workspace from Atlas task SIF %s...",
                    task_image,
                )
                subprocess.run(
                    [
                        "singularity",
                        "exec",
                        "--bind",
                        f"{ws_host}:/workspace-seed",
                        task_image,
                        "sh",
                        "-lc",
                        "cp -a /app/. /workspace-seed/",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                logger.info(
                    f"Cloning {repo_source} into host workspace {ws_host}..."
                )
                subprocess.run(
                    ["git", "clone", repo_source, "."],
                    cwd=ws_host,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Checking out base commit {base_commit}...")
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=ws_host,
                    check=True,
                    capture_output=True,
                    text=True,
                )
        except Exception as e:
            logger.exception("Failed to prime Atlas workspace")
            raise RuntimeError(f"Failed to prime Atlas workspace: {e}")

        # Seed custom agent files into the workspace directory
        import shutil
        app_agents = os.path.join(local_repo_path, self.app_agents_rel)
        if os.path.isdir(app_agents):
            logger.info(f"Seeding custom agent files from {app_agents} to {ws_host}...")
            for f in os.listdir(app_agents):
                src = os.path.join(app_agents, f)
                if os.path.isfile(src):
                    shutil.copyfile(src, ws_host / f)

    def _openclaw_runtime_workspace_mount(self, ws_host: Path) -> Optional[Mount]:
        if self.task_workspace == "/app":
            return Mount(host=ws_host, container="/workspace", mode="rw")
        return super()._openclaw_runtime_workspace_mount(ws_host)

    def _build_state_dir(
        self,
        *,
        run_id: str,
        workspace_root: str,
        sandbox_image: str,
        sandbox_binds: List[str],
        container_prefix: str,
        scope: str = "agent",
    ) -> Path:
        state_dir = super()._build_state_dir(
            run_id=run_id,
            workspace_root=workspace_root,
            sandbox_image=sandbox_image,
            sandbox_binds=sandbox_binds,
            container_prefix=container_prefix,
            scope=scope,
        )
        if workspace_root == "/app":
            import json
            cfg_path = state_dir / "openclaw.json"
            if cfg_path.exists():
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
                agents_defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
                agents_defaults["workspace"] = "/workspace"
                with open(cfg_path, "w") as fh:
                    json.dump(cfg, fh, indent=2)
        return state_dir

