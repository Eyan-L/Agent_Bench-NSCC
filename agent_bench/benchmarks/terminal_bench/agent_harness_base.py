"""Shared Terminal-Bench harness scaffolding for non-OpenClaw agents."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Type

from agent_bench.benchmarks.conditions.agent_otel import otel_endpoint_for_container
from agent_bench.benchmarks.swebench.app import _force_rmtree
from agent_bench.benchmarks.terminal_bench.app import _seed_workspace_helpers
from agent_bench.deployments.containers import (
    Container,
    ContainerHandle,
    ContainerSpec,
    DockerContainer,
    Mount,
)

logger = logging.getLogger(__name__)


class TerminalBenchAgentHarnessBase:
    """Two-container Terminal-Bench harness for bridge-based agent runtimes.

    Subclasses provide the runtime-specific state rendering and environment.
    The task side is always Terminal-Bench's /app + /tests + /logs/verifier
    layout; agent tools are routed into the task container via the runtime's
    dispatcher.
    """

    harness_name = "agent"
    short_name = "agent"
    default_runtime_image = ""
    dispatcher_cls: Type
    state_dir_in_agent = "/opt/agent-bench/state"
    workspace_in_agent = "/app"
    task_workspace = "/app"
    default_verifier_timeout_seconds = 900

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        runtime_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
        container: Optional[Container] = None,
    ):
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.container = container or DockerContainer()
        if self.container.backend_id != "docker":
            raise RuntimeError(
                f"Terminal-Bench {self.harness_name} harness currently supports docker only "
                f"(got {self.container.backend_id!r})"
            )
        self.runtime_image = runtime_image or os.environ.get(
            "AB_AGENT_RUNTIME_IMAGE", self.default_runtime_image,
        )
        self.container.check_available()
        self.container.check_image(self.runtime_image)
        self.workspace_base = Path(
            workspace_base
            or os.environ.get(
                "AB_HARNESS_WORKSPACE_BASE",
                os.path.expanduser("~/.cache/oc-eval/workspaces"),
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base = self.workspace_base.parent / "states"
        self.state_base.mkdir(parents=True, exist_ok=True)
        self._workspace_by_container: Dict[str, Path] = {}

    # ------------------------------------------------------------------

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,
        base_commit: str,
        container_image: Optional[str] = None,
        workspace_root: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        if instance_data is None:
            raise RuntimeError(
                f"TerminalBench{self.harness_name.title()}Harness.prepare_environment "
                "requires instance_data; the driver should pass it through."
            )
        if instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError(
                f"TerminalBench{self.harness_name.title()}Harness only handles "
                f"benchmark=terminal_bench (got {instance_data.get('benchmark')!r})"
            )

        docker_image = instance_data["docker_image"]
        tests_host = Path(instance_data["tests_host_path"])
        if not tests_host.is_dir():
            raise FileNotFoundError(f"task tests dir missing: {tests_host}")

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"{self.short_name}-tb-net-{rid_short}"
        agent_name = f"{self.short_name}-tb-{rid_short}"
        task_name = self._task_container_name(run_id)

        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        app_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.container.copy_from_image(docker_image, self.task_workspace, app_dir)
        except Exception as exc:  # noqa: BLE001 - empty /app is common
            logger.debug("TB %s workspace prime skipped: %s", self.harness_name, exc)
        _seed_workspace_helpers(app_dir, instance_data)

        state_dir = self._build_state_dir(
            run_id=run_id,
            docker_image=docker_image,
            app_dir=app_dir,
            logs_dir=logs_dir,
            instance_data=instance_data,
        )

        self.container.create_network(net_name)
        task_spec = ContainerSpec(
            name=task_name,
            image=docker_image,
            network=net_name,
            user="0",
            labels={
                f"{self.harness_name}.sandbox": "1",
                "oc-eval.run_id": run_id,
            },
            mounts=(
                Mount(host=app_dir, container=self.task_workspace, mode="rw"),
                Mount(host=tests_host, container="/tests", mode="ro"),
                Mount(host=logs_dir, container="/logs/verifier", mode="rw"),
            ),
            workdir=self.task_workspace,
            env={"HOME": "/root"},
            entrypoint="sleep",
            command=("infinity",),
        )
        try:
            task_handle = self.container.start(task_spec)
        except Exception:
            self._best_effort_cleanup(net_name, agent_name, task_name)
            raise

        dispatcher = self.dispatcher_cls()
        try:
            dispatcher.start(state_dir=state_dir, backend="docker", target=task_name)
        except Exception:
            self._best_effort_cleanup(net_name, agent_name, task_name)
            raise
        bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()

        agent_env = self._agent_env(
            run_id=run_id,
            docker_image=docker_image,
            instance_data=instance_data,
        )
        agent_env.update(bridge_env)
        agent_mounts = [
            Mount(host=state_dir, container=self.state_dir_in_agent, mode="rw"),
            Mount(host=app_dir, container=self.workspace_in_agent, mode="rw"),
        ]
        agent_mounts.extend(bridge_mounts)
        agent_spec = ContainerSpec(
            name=agent_name,
            image=self.runtime_image,
            network=net_name,
            extra_hosts={"vllm-host": "host-gateway"},
            user="0",
            mounts=tuple(agent_mounts),
            workdir=self.workspace_in_agent,
            env=agent_env,
        )
        try:
            agent_handle = self.container.start(agent_spec)
        except Exception:
            dispatcher.stop()
            self._best_effort_cleanup(net_name, agent_name, task_name)
            raise

        self._workspace_by_container[agent_handle.id] = app_dir
        logger.info(
            "  TB %s ready: agent=%s task=%s bridge_pid=%s net=%s",
            self.harness_name,
            agent_handle.id[:12], task_handle.id[:12], dispatcher.pid, net_name,
        )
        return {
            "container_id": agent_handle.id,
            "container_name": agent_name,
            "backend": self.container.backend_id,
            "harness": self.harness_name,
            f"{self.short_name}_container_id": agent_handle.id,
            "task_container_id": task_handle.id,
            "task_container_name": task_name,
            "network_name": net_name,
            "workspace_root": self.workspace_in_agent,
            "task_workspace_root": self.task_workspace,
            "workspace_host_path": str(app_dir),
            "host_logs_dir": str(logs_dir),
            "state_dir": str(state_dir),
            "sandbox_image": docker_image,
            "container_engine": "docker",
            "instance_name": task_name,
            "tb_instruction": instance_data["instruction"],
            "tb_verifier_entrypoint": instance_data.get("verifier_entrypoint", "/tests/test.sh"),
            "tb_reward_file": instance_data.get("reward_file", "/logs/verifier/reward.txt"),
            "agent_env": agent_env,
            **self._condition_env_info(state_dir),
            **dispatcher.env_info(),
        }

    # ------------------------------------------------------------------

    def _build_state_dir(self, **kwargs) -> Path:
        raise NotImplementedError

    def _agent_env(self, **kwargs) -> dict[str, str]:
        raise NotImplementedError

    def _condition_env_info(self, state_dir: Path) -> dict[str, str]:
        return {}

    def _task_container_name(self, run_id: str) -> str:
        return f"{self.short_name}-tb-task-{run_id[:12]}"

    def _runtime_base_url(self, configured: Optional[str]) -> str:
        from agent_bench.core.config import endpoint_for_container
        raw = os.environ.get("OPENAI_BASE_URL") or configured or "http://localhost:8000/v1"
        return endpoint_for_container(raw, self.container.host_address)

    def _otel_endpoint(self) -> str:
        return otel_endpoint_for_container(self.container.host_address)

    def _template_path(self, harness_name: str, filename: str) -> Path:
        repo_root = Path(os.environ.get("AB_REPO_ROOT", os.getcwd()))
        candidates = [
            repo_root / "agent_bench" / "harnesses" / harness_name / "state" / filename,
            repo_root / "harnesses" / harness_name / "state" / filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"{harness_name} {filename} template missing; checked "
            + ", ".join(str(c) for c in candidates)
        )

    # ------------------------------------------------------------------

    def _handle(self, container_id: str) -> ContainerHandle:
        return ContainerHandle(id=container_id, name=container_id, backend=self.container.backend_id)

    def exec_in_container(
        self,
        container_id: str,
        argv: list[str],
        *,
        timeout: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        check: bool = False,
        stdin_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        return self.container.exec(
            self._handle(container_id),
            argv,
            check=check,
            timeout=timeout if timeout is not None else timeout_seconds,
            env=env,
            workdir=workdir,
            input_text=stdin_text,
        )

    def _exec_result_dict(
        self,
        target: str,
        argv: list[str],
        *,
        timeout_seconds: int,
        stdin_text: Optional[str] = None,
    ) -> Dict:
        try:
            proc = self.exec_in_container(
                target,
                argv,
                timeout_seconds=timeout_seconds,
                stdin_text=stdin_text,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "rc": -1,
                "stdout": _as_text(exc.stdout),
                "stderr": _as_text(exc.stderr) + f"\n[harness] timeout after {timeout_seconds}s",
                "timed_out": True,
            }
        return {
            "rc": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "timed_out": False,
        }

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
        run_dir: Optional[Path] = None,
        **_kw,
    ) -> Dict:
        run_id = run_id or instance_id
        task_name = self._task_container_name(run_id)
        host_logs_dir = self.workspace_base / run_id / "logs" / "verifier"
        timeout = timeout_seconds or self.default_verifier_timeout_seconds
        if instance_data:
            timeout = int(instance_data.get("verifier_timeout_sec", timeout))
        verifier = (instance_data or {}).get("verifier_entrypoint", "/tests/test.sh")
        logger.info(
            "scoring TB %s task container %s (verifier=%s, timeout=%ds)",
            self.harness_name, task_name, verifier, timeout,
        )
        result = self._exec_result_dict(task_name, ["bash", verifier], timeout_seconds=timeout)
        host_reward = host_logs_dir / "reward.txt"
        try:
            reward_text = host_reward.read_text().strip()
        except FileNotFoundError:
            reward_text = ""
        if reward_text == "1":
            status = "PASS"
        elif reward_text == "0":
            status = "FAIL"
        elif result["timed_out"]:
            status = "TIMEOUT"
        else:
            status = "ERROR"
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

    def collect_patch(self, container_id: str, workspace_root: str) -> str:
        return "# terminal_bench: no patch - agent state in container\n"

    def cleanup(
        self,
        container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
        container_name: Optional[str] = None,
        network_name: Optional[str] = None,
        workspace_host_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        bridge_pid: Optional[int] = None,
        bridge_socket: Optional[str] = None,
        **extra,
    ) -> None:
        self.dispatcher_cls.stop_external(pid=bridge_pid, socket_path=bridge_socket)
        uid, gid = os.getuid(), os.getgid()
        for cid in {container_id, extra.get(f"{self.short_name}_container_id")}:
            if not cid:
                continue
            self._workspace_by_container.pop(cid, None)
            try:
                self.exec_in_container(
                    cid,
                    ["chown", "-R", f"{uid}:{gid}", self.state_dir_in_agent],
                    timeout=30,
                    check=False,
                )
            except Exception:
                pass
        targets = [t for t in (container_id, extra.get(f"{self.short_name}_container_id"), task_container_id, task_container_name, container_name) if t]
        for cid in dict.fromkeys(targets):
            try:
                self.container.stop(self._handle(cid))
            except Exception:
                pass
        if network_name:
            try:
                self.container.remove_network(network_name)
            except Exception:
                pass
        keep = os.environ.get("AB_HARNESS_KEEP_WORKSPACE", "0") == "1"
        if not keep and workspace_host_path and os.path.isdir(workspace_host_path):
            _force_rmtree(Path(workspace_host_path).parent, ignore_errors=True)
        if not keep and state_dir and os.path.isdir(state_dir):
            _force_rmtree(Path(state_dir), ignore_errors=True)

    def _best_effort_cleanup(self, net: str, agent: str, task: Optional[str] = None) -> None:
        for name in (agent, task):
            if not name:
                continue
            try:
                self.container.stop(self._handle(name))
            except Exception:
                pass
        try:
            self.container.remove_network(net)
        except Exception:
            pass


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)
