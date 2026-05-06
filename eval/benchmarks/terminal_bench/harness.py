"""Singularity-backed harness for Terminal-Bench 2 tasks.

Same surface as eval/runner/harness.py:HostHarness — `prepare_environment`,
`run_tests`, `cleanup` — so the existing driver loop in
eval/runner/driver.py:execute_run drives it without changes.

Per-run flow:
    1. Resolve the per-task .sif under $OC_TB_SIF_DIR/<image-tag>.sif.
       If missing, `singularity pull` from `docker://<docker_image>` (one-time
       per docker tag, cached across runs).
    2. `singularity instance start` the task sif with binds:
         - <run_dir>/app  → /app  (rw, agent's workspace)
         - <task>/tests   → /tests (ro, vendored tests)
         - <run_dir>/logs → /logs/verifier (rw, reward.txt lands here)
    3. Hand the instance name back to the condition via env_info so the
       condition can `singularity exec instance://<name> ...` to drive work
       inside the container.
    4. After the condition finishes, run_tests `singularity exec` runs
       /tests/test.sh inside the instance and reads /logs/verifier/reward.txt.
    5. cleanup() stops the instance.

Why singularity, not docker:
    HPC compute nodes (the target deployment) have singularity but no docker
    daemon. The pull-once-on-login-node pattern is already used for the
    SWE-bench scoring sifs and the OC runtime sif.

Anti-cheat note:
    The verifier mounts /tests:ro to keep the agent from rewriting the
    test files. The reward file lands under /logs/verifier which is also
    bound rw — agents *could* write to it, but the verifier overwrites
    /logs/verifier/reward.txt as its last step (see test.sh).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TerminalBenchHarness:
    """Stand up a TB task container, score it, tear it down."""

    backend_id = "terminal_bench"

    # Reasonable time budgets — task.toml has its own per-task overrides
    # which the loader copies into instance_data.
    _PULL_TIMEOUT_SECONDS = 1800     # singularity pull from docker hub
    _INSTANCE_START_TIMEOUT_SECONDS = 60
    _DEFAULT_VERIFIER_TIMEOUT_SECONDS = 900

    def __init__(self, swebench_root: Optional[Path] = None,
                 workspace_base: Optional[Path] = None):
        # swebench_root accepted for API parity with the driver — unused.
        self.swebench_root = Path(swebench_root) if swebench_root else None
        self.workspace_base = Path(
            workspace_base or os.environ.get(
                "OC_HOST_WORKSPACE_BASE",
                os.path.expanduser("~/.cache/oc-eval/workspaces"),
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)

        self.sif_dir = Path(os.environ.get(
            "OC_TB_SIF_DIR",
            str(self.workspace_base.parent / "tb-sif"),
        ))
        self.sif_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which("singularity"):
            raise RuntimeError("singularity not on PATH")

    # ------------------------------------------------------------------
    # SIF acquisition
    # ------------------------------------------------------------------

    @staticmethod
    def _sif_name_for(docker_image: str) -> str:
        """Sanitize 'org/name:tag' -> 'org__name__tag.sif'."""
        return docker_image.replace("/", "__").replace(":", "__") + ".sif"

    def _ensure_sif(self, docker_image: str) -> Path:
        """Return path to a cached .sif for `docker_image`, pulling if absent."""
        sif_path = self.sif_dir / self._sif_name_for(docker_image)
        if sif_path.exists() and sif_path.stat().st_size > 0:
            return sif_path
        logger.info("pulling %s -> %s", docker_image, sif_path)
        # `singularity pull` writes atomically only on success; clean up any
        # partial file from a prior crash.
        tmp = sif_path.with_suffix(".sif.partial")
        if tmp.exists():
            tmp.unlink()
        r = subprocess.run(
            ["singularity", "pull", "--name", tmp.name,
             f"docker://{docker_image}"],
            cwd=str(sif_path.parent),
            capture_output=True, text=True,
            timeout=self._PULL_TIMEOUT_SECONDS,
        )
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"singularity pull docker://{docker_image} failed "
                f"(rc={r.returncode}): {r.stderr[-600:]}"
            )
        tmp.rename(sif_path)
        return sif_path

    # ------------------------------------------------------------------
    # Driver-facing API
    # ------------------------------------------------------------------

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,                           # ignored
        base_commit: str,                       # ignored
        container_image: Optional[str] = None,  # ignored — we use task.toml
        workspace_root: Optional[str] = None,   # ignored — we always use /app
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        if instance_data is None:
            raise RuntimeError(
                "TerminalBenchHarness.prepare_environment requires instance_data; "
                "the driver should pass it through."
            )
        if instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError(
                "TerminalBenchHarness only handles benchmark=terminal_bench instances "
                f"(got benchmark={instance_data.get('benchmark')!r})"
            )

        docker_image = instance_data["docker_image"]
        tests_host = Path(instance_data["tests_host_path"])
        if not tests_host.is_dir():
            raise FileNotFoundError(f"task tests dir missing: {tests_host}")

        sif_path = self._ensure_sif(docker_image)

        run_id = run_id or instance_id
        run_root = self.workspace_base / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        app_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        instance_name = self._instance_name_for(run_id)
        # If a stale instance with this name lingers (prior crash), tear it
        # down first; `singularity instance start` errors out otherwise.
        self._try_stop_instance(instance_name)

        argv = [
            "singularity", "instance", "start",
            "--writable-tmpfs",
            "--bind", f"{app_dir}:/app:rw",
            "--bind", f"{tests_host}:/tests:ro",
            "--bind", f"{logs_dir}:/logs/verifier:rw",
            str(sif_path),
            instance_name,
        ]
        logger.info("starting TB instance %s from %s", instance_name, sif_path.name)
        r = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=self._INSTANCE_START_TIMEOUT_SECONDS,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"singularity instance start failed (rc={r.returncode}): "
                f"{r.stderr[-600:]}"
            )

        return {
            "container_id": None,             # signals non-docker to driver
            "container_name": instance_name,  # used by condition to exec
            "backend": self.backend_id,
            "workspace_root": "/app",         # path inside the instance
            "host_workspace_root": str(app_dir),
            "host_logs_dir": str(logs_dir),
            "task_sif": str(sif_path),
            "instance_name": instance_name,
            # Echo TB metadata so conditions don't have to re-load task.toml.
            "tb_instruction": instance_data["instruction"],
            "tb_verifier_entrypoint": instance_data.get(
                "verifier_entrypoint", "/tests/test.sh"
            ),
            "tb_reward_file": instance_data.get(
                "reward_file", "/logs/verifier/reward.txt"
            ),
        }

    # ------------------------------------------------------------------

    def exec_in_container(
        self,
        instance_name: str,
        argv: List[str],
        timeout_seconds: int = 300,
        stdin_text: Optional[str] = None,
    ) -> Dict:
        """Run `argv` inside a running TB instance. Returns rc/stdout/stderr.

        Conditions use this to drive the agent's work — write files, run pip,
        spawn the server, etc. — without owning the singularity command line.
        """
        full = ["singularity", "exec", f"instance://{instance_name}", *argv]
        try:
            r = subprocess.run(
                full,
                input=stdin_text,
                capture_output=True, text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "rc": -1,
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + f"\n[harness] timeout after {timeout_seconds}s",
                "timed_out": True,
            }
        return {
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "timed_out": False,
        }

    # ------------------------------------------------------------------

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,            # ignored — TB has no patch
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        """Run /tests/test.sh inside the TB instance. Verdict from reward.txt."""
        instance_name = self._instance_name_for(run_id or instance_id)
        verifier = "/tests/test.sh"
        reward_path_in_container = "/logs/verifier/reward.txt"
        host_logs_dir = (
            self.workspace_base / (run_id or instance_id) / "logs" / "verifier"
        )

        timeout = timeout_seconds or self._DEFAULT_VERIFIER_TIMEOUT_SECONDS
        if instance_data:
            timeout = int(
                instance_data.get("verifier_timeout_sec", timeout)
            )

        logger.info("scoring TB instance %s (verifier=%s, timeout=%ds)",
                    instance_name, verifier, timeout)
        result = self.exec_in_container(
            instance_name,
            ["bash", verifier],
            timeout_seconds=timeout,
        )

        # Even on timeout, the verifier may have written reward.txt for a
        # partial run. Check the host-bound copy.
        host_reward = host_logs_dir / "reward.txt"
        reward_text = ""
        try:
            reward_text = host_reward.read_text().strip()
        except FileNotFoundError:
            pass

        if reward_text == "1":
            status = "PASS"
        elif reward_text == "0":
            status = "FAIL"
        elif result["timed_out"]:
            status = "TIMEOUT"
        else:
            # Verifier ran but didn't write a reward file we can parse —
            # treat as ERROR so the driver retries on next sweep.
            status = "ERROR"

        # Persist verifier output for post-hoc debugging. The driver writes
        # verdict.json with what we return, so callers don't have to.
        return {
            "status": status,
            "test_pass_rate": 1.0 if status == "PASS" else 0.0,
            "passed": 1 if status == "PASS" else 0,
            "failed": 0 if status == "PASS" else 1,
            "verifier_rc": result["rc"],
            "verifier_stdout_tail": result["stdout"][-2000:],
            "verifier_stderr_tail": result["stderr"][-2000:],
            "reward_raw": reward_text or None,
            "reward_path": str(host_reward),
            "evaluation_dir": str(host_logs_dir),
            "error_message": (
                None if status == "PASS"
                else f"verifier exit {result['rc']}, reward={reward_text!r}"
            ),
        }

    # ------------------------------------------------------------------

    def cleanup(self, container_id: Optional[str] = None, **kwargs) -> None:
        """Stop the singularity instance. Tolerant of missing instances —
        cleanup runs in a finally block so a torn-down instance during
        run_tests must not raise here."""
        # The driver passes a bag of kwargs from env_info. We need
        # container_name (or workspace_root parent) to know which instance
        # to stop. The driver currently passes container_id (None for us)
        # and workspace_root — fall back to scanning if neither suffices.
        instance_name = kwargs.get("container_name")
        if not instance_name:
            ws = kwargs.get("workspace_root")
            if ws:
                # workspace_root inside the container is /app — useless here.
                # Reconstruct from the host path if the caller supplied it.
                pass
        if instance_name:
            self._try_stop_instance(instance_name)
        # else: leave a stale instance — `singularity instance list` will
        # show it; the next run with the same run_id will tear it down in
        # _try_stop_instance.

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _instance_name_for(run_id: str) -> str:
        # Singularity instance names: alnum + dashes only, max 32 chars
        # (loose limit varies by version; keep short to be safe).
        safe = "".join(c if c.isalnum() or c == "-" else "-" for c in run_id)
        return f"tb-{safe}"[:32]

    @staticmethod
    def _try_stop_instance(name: str) -> None:
        try:
            subprocess.run(
                ["singularity", "instance", "stop", name],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:  # noqa: BLE001 — best-effort cleanup
            logger.debug("instance stop %s: ignored exception", name, exc_info=True)
