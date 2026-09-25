"""Independent Hermes lifecycle for RoadmapBench's Terminal-Bench task images."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from agent_bench.benchmarks.swebench.app import _force_rmtree
from agent_bench.benchmarks.terminal_bench.app import (
    _as_text,
    _clear_reward_file,
    _docker_image_workdir,
    _require_tests_host_path,
    _resolve_task_workspace,
    _terminal_bench_reward,
)
from agent_bench.benchmarks.terminal_bench.hermes.app import TerminalBenchHermesHarness
from agent_bench.benchmarks.terminal_bench.agent_harness_base import _cell_scope
from agent_bench.deployments.containers import Container, ContainerHandle, ContainerSpec, Mount
from agent_bench.deployments.containers.limits import configured_task_container_cpus


class RoadmapBenchHermesHarness(TerminalBenchHermesHarness):
    """Run Hermes, an exec dispatcher, and the roadmap task as separate runtimes.

    This deliberately does not change the Docker-only Terminal-Bench Hermes
    harness. A Singularity task SIF must already be prepared before this run.
    """

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        *,
        container: Container,
        runtime_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ) -> None:
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.container = container
        if container.backend_id == "singularity":
            self.runtime_image = runtime_image or os.environ.get("AB_AGENT_RUNTIME_SIF", "")
            if not self.runtime_image:
                raise RuntimeError("AB_AGENT_RUNTIME_SIF is required for Roadmap Hermes on Singularity")
        elif container.backend_id == "docker":
            self.runtime_image = runtime_image or os.environ.get(
                "AB_AGENT_RUNTIME_IMAGE", self.default_runtime_image,
            )
        else:
            raise RuntimeError(f"Unsupported Roadmap Hermes backend: {container.backend_id}")
        container.check_available()
        container.check_image(self.runtime_image)
        self.task_container_cpus = configured_task_container_cpus()
        self.workspace_base = Path(
            workspace_base
            or os.environ.get("AB_HARNESS_WORKSPACE_BASE", "~/.cache/agent-bench/workspaces")
        ).expanduser()
        self.state_base = self.workspace_base.parent / "states"
        scope = _cell_scope()
        if scope:
            self.workspace_base = self.workspace_base / scope
            self.state_base = self.state_base / scope
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base.mkdir(parents=True, exist_ok=True)
        self._workspace_by_container: Dict[str, Path] = {}
        self._task_specs: Dict[str, ContainerSpec] = {}

    def _build_state_dir(self, **kwargs) -> Path:
        state_dir = super()._build_state_dir(**kwargs)
        delegation = (self._load_harness_config().get("hermes") or {}).get("delegation") or {}
        values = {
            "max_concurrent_children": int(delegation.get("max_concurrent_children", 16)),
            "max_iterations": int(delegation.get("max_iterations", 600)),
            "child_timeout_seconds": int(delegation.get("child_timeout_seconds", 7200)),
        }
        if any(value < 1 for value in values.values()):
            raise ValueError("Roadmap Hermes delegation settings must be positive")
        with (state_dir / "config.yaml").open("a") as config_file:
            config_file.write("\ndelegation:\n")
            for key, value in values.items():
                config_file.write(f"  {key}: {value}\n")
        return state_dir

    def _task_image(self, docker_image: str) -> str:
        if self.container.backend_id == "docker":
            return docker_image
        image_dir = os.environ.get("AB_DEPLOYMENT_TASK_IMAGE_DIR")
        if not image_dir:
            raise RuntimeError("AB_DEPLOYMENT_TASK_IMAGE_DIR is required for Roadmap Hermes")
        path = Path(image_dir) / (docker_image.replace("/", "__").replace(":", "__") + ".sif")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Prepared Roadmap task SIF missing: {path}")
        return str(path)

    def prepare_environment(
        self, instance_id: str, repo_id: str, base_commit: str,
        container_image: Optional[str] = None, workspace_root: Optional[str] = None,
        run_id: Optional[str] = None, instance_data: Optional[Dict] = None,
    ) -> Dict:
        if not instance_data or instance_data.get("benchmark_variant") != "roadmapbench":
            raise RuntimeError("RoadmapBenchHermesHarness requires a RoadmapBench task")
        docker_image = instance_data["docker_image"]
        image = self._task_image(docker_image)
        image_workdir = _docker_image_workdir(docker_image) if self.container.backend_id == "docker" else None
        task_workspace = _resolve_task_workspace(instance_data, image_workdir)
        run_id = run_id or instance_id
        slug = self._resource_slug(run_id)
        net_name = f"hm-rm-net-{slug}"
        agent_name = f"hm-rm-{slug}"
        task_name = self._task_container_name(run_id)
        self._preflight_cleanup(net_name, agent_name, task_name)

        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        tests_dir = run_root / "tests"
        app_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        tests_dir.mkdir(parents=True)
        try:
            self.container.copy_from_image(image, task_workspace, app_dir)
        except Exception as exc:
            _force_rmtree(run_root, ignore_errors=True)
            raise RuntimeError(f"Roadmap workspace prime failed: {image}:{task_workspace}") from exc

        state_dir = self._build_state_dir(
            run_id=run_id, docker_image=docker_image, app_dir=app_dir,
            logs_dir=logs_dir, instance_data=instance_data, task_workspace=task_workspace,
        )
        self.container.create_network(net_name)
        task_mounts = [
            Mount(app_dir, task_workspace),
            Mount(logs_dir, "/logs/verifier"),
        ]
        if self.container.backend_id == "singularity":
            task_mounts.append(Mount(tests_dir, "/tests"))
        task_spec = ContainerSpec(
            name=task_name, image=image, network=net_name, user="0",
            mounts=tuple(task_mounts), workdir=task_workspace,
            env={"HOME": "/root"}, entrypoint="sleep", command=("infinity",),
            cpus=self.task_container_cpus,
        )
        dispatcher = self.dispatcher_cls()
        try:
            task_handle = self.container.start(task_spec)
            self._task_specs[task_name] = task_spec
            dispatcher.start(
                state_dir=state_dir, backend=self.container.backend_id, target=task_name,
            )
            bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
            agent_env = self._agent_env(
                run_id=run_id, docker_image=docker_image,
                instance_data=instance_data, task_workspace=task_workspace,
            )
            agent_env.update(bridge_env)
            agent_env["HOME"] = "/root"
            agent_spec = ContainerSpec(
                name=agent_name, image=self.runtime_image, network=net_name,
                extra_hosts={"vllm-host": "host-gateway"}, user="0",
                mounts=(
                    Mount(state_dir, self.state_dir_in_agent),
                    Mount(app_dir, self.workspace_in_agent),
                    *bridge_mounts,
                ),
                workdir=self.workspace_in_agent, env=agent_env,
            )
            agent_handle = self.container.start(agent_spec)
        except BaseException:
            dispatcher.stop()
            self._best_effort_cleanup(net_name, agent_name, task_name)
            self._task_specs.pop(task_name, None)
            raise

        self._workspace_by_container[agent_handle.id] = app_dir
        return {
            "container_id": agent_handle.id,
            "container_name": agent_name,
            "hm_container_id": agent_handle.id,
            "task_container_id": task_handle.id,
            "task_container_name": task_name,
            "network_name": net_name,
            "backend": self.container.backend_id,
            "container_engine": self.container.backend_id,
            "harness": "hermes",
            "workspace_root": self.workspace_in_agent,
            "task_workspace_root": task_workspace,
            "workspace_host_path": str(app_dir),
            "host_logs_dir": str(logs_dir),
            "state_dir": str(state_dir),
            "tb_instruction": instance_data["instruction"],
            "tb_verifier_entrypoint": instance_data.get("verifier_entrypoint", "/tests/test.sh"),
            "tb_reward_file": instance_data.get("reward_file", "/logs/verifier/reward.txt"),
            "agent_env": agent_env,
            **self._condition_env_info(state_dir),
            **dispatcher.env_info(),
        }

    def quiesce_for_scoring(self, env_info: Dict) -> None:
        """Stop all agent and task writers before freezing the candidate."""
        task_name = env_info["task_container_name"]
        task_spec = self._task_specs[task_name]
        self.dispatcher_cls.stop_external(
            pid=env_info.get("bridge_pid"), socket_path=env_info.get("bridge_socket"),
        )
        self.container.stop(self._handle(env_info["container_id"]))
        self.container.stop(self._handle(task_name))
        task_handle = self.container.start(task_spec)
        env_info["task_container_id"] = task_handle.id
        env_info["workspace_quiesced"] = True

    def exec_in_container(self, container_id: str, argv: list[str], **kwargs):
        # The shared Hermes condition pins the legacy Terminal-Bench toolset.
        # Add delegation only for this dedicated Roadmap harness, without
        # altering the Terminal-Bench Hermes condition or its CLI behavior.
        if argv and argv[0] == "hermes" and "-z" in argv and "--toolsets" in argv:
            argv = list(argv)
            idx = argv.index("--toolsets") + 1
            argv[idx] = "terminal,file,code_execution,delegation"
        return super().exec_in_container(container_id, argv, **kwargs)

    def run_tests(
        self, container_id: Optional[str], instance_id: str, workspace_root: str,
        timeout_seconds: int, patch: Optional[str] = None,
        run_id: Optional[str] = None, instance_data: Optional[Dict] = None,
        run_dir: Optional[Path] = None, **_kw,
    ) -> Dict:
        if not instance_data:
            raise RuntimeError("Roadmap verifier requires instance_data")
        run_id = run_id or instance_id
        task_name = self._task_container_name(run_id)
        task_handle = self._handle(task_name)
        task_workspace = _resolve_task_workspace(instance_data)
        tests_host = _require_tests_host_path(instance_data)
        if self.container.backend_id == "singularity":
            tests_dir = self.workspace_base / run_id / "tests"
            if any(tests_dir.iterdir()):
                raise RuntimeError("Private /tests bind was populated before scoring")
        else:
            self.container.exec(task_handle, ["rm", "-rf", "/tests"], check=True, timeout=60)
        self.container.copy_into(task_handle, tests_host, "/tests")
        host_logs = self.workspace_base / run_id / "logs" / "verifier"
        reward_path = _clear_reward_file(host_logs)
        timeout = int(instance_data.get("verifier_timeout_sec", timeout_seconds or 900))
        verifier_env = (
            {"TAR_OPTIONS": "--no-same-owner", "PYTHONPATH": ""}
            if self.container.backend_id == "singularity" else None
        )
        try:
            result = self.container.exec(
                task_handle, ["bash", instance_data.get("verifier_entrypoint", "/tests/test.sh")],
                check=False, timeout=timeout, workdir=task_workspace, env=verifier_env,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess([], -1, _as_text(exc.stdout), _as_text(exc.stderr))
            timed_out = True
        reward_text = reward_path.read_text().strip() if reward_path.exists() else ""
        status, reward = _terminal_bench_reward(reward_text, timed_out=timed_out)
        return {
            "status": status, "verifier_status": status, "reward": reward,
            "test_pass_rate": reward, "passed": int(status == "PASS"),
            "failed": int(status != "PASS"), "verifier_rc": result.returncode,
            "verifier_timed_out": timed_out,
            "verifier_stdout_tail": _as_text(result.stdout)[-2000:],
            "verifier_stderr_tail": _as_text(result.stderr)[-2000:],
            "reward_raw": reward_text or None, "reward_path": str(reward_path),
            "evaluation_dir": str(host_logs),
            "error_message": None if status == "PASS" else f"verifier exit {result.returncode}, reward={reward_text!r}",
        }

    def cleanup(
        self, container_id: Optional[str] = None,
        task_container_name: Optional[str] = None, **kwargs,
    ) -> None:
        if task_container_name:
            self._task_specs.pop(task_container_name, None)
        super().cleanup(
            container_id=container_id, task_container_name=task_container_name,
            **kwargs,
        )
