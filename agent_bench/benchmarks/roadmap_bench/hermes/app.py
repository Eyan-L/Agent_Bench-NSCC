"""Hermes on the existing RoadmapBench task and verifier lifecycle."""
from __future__ import annotations

import json
import os
import shutil
import string
import time
from pathlib import Path
from typing import Dict, Optional

from agent_bench.benchmarks.conditions.agent_otel import otel_config_values, otel_env
from agent_bench.benchmarks.swebench.app import _force_rmtree
from agent_bench.benchmarks.terminal_bench.app import (
    TerminalBenchOpenClawHarness,
    _docker_image_workdir,
    _resolve_task_workspace,
    _seed_workspace_helpers,
)
from agent_bench.core.config import load_active_model_and_serving, runtime_endpoint_for_container
from agent_bench.core.profile_io import load_profile
from agent_bench.deployments.containers.base import ContainerHandle, ContainerSpec, Mount
from agent_bench.harnesses.hermes.dispatcher import HermesDispatcher
from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher


class RoadmapBenchHermesHarness(TerminalBenchOpenClawHarness):
    """Run patched Hermes beside a Roadmap task SIF, then score the frozen task."""

    def __init__(self, benchmark_data_root: Optional[Path] = None, *, container, **kwargs):
        if container.backend_id == "docker" and "oc_image" not in kwargs:
            kwargs["oc_image"] = (
                os.environ.get("AB_AGENT_RUNTIME_DOCKER")
                or "agent-bench/hermes-roadmap:v2026.5.16"
            )
        super().__init__(benchmark_data_root, container=container, **kwargs)
        # Content-derived run IDs may be identical across harness arms.
        self.workspace_base = self.workspace_base / "roadmap-hermes"
        self.state_base = self.state_base / "roadmap-hermes"
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _task_container_name(run_id: str) -> str:
        return f"hm-rm-task-{run_id[:12]}"

    def _build_hermes_state(self, run_id: str, task_workspace: str, docker_image: str) -> Path:
        repo_root = Path(os.environ.get("AB_REPO_ROOT", Path.cwd()))
        profile = load_profile(os.environ.get(
            "AB_HARNESS_CONFIG", str(repo_root / "profiles/harnesses/hermes_roadmapbench.yaml")
        ))
        if profile.get("harness") != "hermes":
            raise ValueError("Roadmap Hermes requires a Hermes harness profile")
        model, _serving = load_active_model_and_serving(repo_root=str(repo_root))
        hm = profile.get("hermes") or {}
        state = self.state_base / run_id
        if state.exists():
            _force_rmtree(state)
        (state / "logs").mkdir(parents=True)
        endpoint = os.environ.get("AB_SERVING_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL"
        ) or model.endpoint
        base_url = runtime_endpoint_for_container(endpoint, self.container.host_address)
        provider = os.environ.get("HERMES_INFERENCE_PROVIDER") or model.provider
        api_key = os.environ.get("OPENAI_API_KEY") or model.api_key or "sk-dummy"
        values = {
            "AB_MODEL_NAME": os.environ.get("AB_MODEL_NAME") or model.name,
            "PROVIDER": provider,
            "BASE_URL": base_url,
            "API_KEY": api_key,
            "TEMPERATURE": str(model.temperature),
            "AB_MODEL_MAX_OUTPUT_TOKENS": str(model.max_tokens),
            "CONTEXT_LENGTH": str(model.context_window),
            "MAX_TURNS": str((hm.get("run") or {}).get("max_turns", 600)),
            "TERMINAL_TIMEOUT": str((hm.get("run") or {}).get("terminal_timeout", 180)),
            "TERMINAL_CWD": task_workspace,
            "SANDBOX_IMAGE": docker_image,
            **otel_config_values(
                service_name="agent-bench-hermes-roadmap",
                run_id=run_id,
                endpoint=os.environ.get("AB_TELEMETRY_ENDPOINT", f"http://{self.container.host_address}:4318"),
                extra_attrs={"hm.run_id": run_id, "roadmap.instance_id": run_id},
                enabled=((profile.get("telemetry") or {}).get("otel") or {}).get("enabled", True) is not False,
            ),
        }
        template = repo_root / "agent_bench/benchmarks/roadmap_bench/hermes/config.yaml.template"
        (state / "config.yaml").write_text(string.Template(template.read_text()).substitute(values))
        return state

    def prepare_environment(
        self, instance_id: str, repo_id: str, base_commit: str,
        container_image: Optional[str] = None, workspace_root: Optional[str] = None,
        run_id: Optional[str] = None, instance_data: Optional[Dict] = None,
    ) -> Dict:
        if not instance_data or instance_data.get("benchmark_variant") != "roadmapbench":
            raise ValueError("RoadmapBenchHermesHarness requires RoadmapBench instance_data")
        docker_image = instance_data["docker_image"]
        task_workspace = _resolve_task_workspace(
            instance_data,
            _docker_image_workdir(docker_image) if self.container.backend_id == "docker" else None,
        )
        task_image = self._resolve_task_image(docker_image)
        run_id = run_id or instance_id
        net_name = f"hm-rm-net-{run_id[:12]}"
        agent_name = f"hm-rm-{run_id[:12]}"
        task_name = self._task_container_name(run_id)
        self._best_effort_stop(
            ContainerHandle(agent_name, agent_name, self.container.backend_id),
            ContainerHandle(task_name, task_name, self.container.backend_id),
        )
        self.container.remove_network(net_name)
        self.container.create_network(net_name)

        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        tests_dir = run_root / "tests"
        for directory in (app_dir, logs_dir, tests_dir):
            directory.mkdir(parents=True)
        prepare_start_ns = time.time_ns()
        try:
            self.container.copy_from_image(task_image, task_workspace, app_dir)
        except Exception as exc:
            _force_rmtree(run_root, ignore_errors=True)
            self.container.remove_network(net_name)
            raise RuntimeError("Roadmap workspace prime failed; refusing partial task workspace") from exc
        _seed_workspace_helpers(app_dir, instance_data, task_workspace)
        prepare_end_ns = time.time_ns()
        state = self._build_hermes_state(run_id, task_workspace, docker_image)

        task_mounts = [Mount(app_dir, task_workspace), Mount(logs_dir, "/logs/verifier")]
        if self.container.backend_id == "singularity":
            task_mounts.append(Mount(tests_dir, "/tests"))
        task_spec = ContainerSpec(
            name=task_name, image=task_image, network=net_name, user="0",
            mounts=tuple(task_mounts), env={"HOME": "/root"},
            workdir=task_workspace, entrypoint="sleep", command=("infinity",),
            cpus=self.task_container_cpus,
        )
        task_handle = None
        agent_handle = None
        dispatcher = None
        try:
            with self._task_specs_lock:
                self._task_specs[task_name] = task_spec
            task_handle = self.container.start(task_spec)
            dispatcher = HermesDispatcher()
            dispatcher.start(state_dir=state, backend=self.container.backend_id, target=task_name)
            bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
            model, _serving = load_active_model_and_serving(
                repo_root=os.environ.get("AB_REPO_ROOT", os.getcwd())
            )
            provider = os.environ.get("HERMES_INFERENCE_PROVIDER") or model.provider
            endpoint = os.environ.get("AB_SERVING_BASE_URL") or os.environ.get(
                "OPENAI_BASE_URL"
            ) or model.endpoint
            base_url = runtime_endpoint_for_container(endpoint, self.container.host_address)
            api_key = os.environ.get("OPENAI_API_KEY") or model.api_key or "sk-dummy"
            agent_env = {
                "HOME": "/root", "HERMES_HOME": "/opt/data",
                "AB_RUN_ID": run_id, "HERMES_RUN_ID": run_id,
                "HERMES_INFERENCE_MODEL": os.environ.get("AB_MODEL_NAME") or model.name,
                "HERMES_INFERENCE_PROVIDER": provider,
                "AB_AGENT_RUNTIME_EVENTS_FILE": "/opt/data/events.jsonl",
                "OPENAI_API_KEY": api_key, "OPENAI_BASE_URL": base_url,
                "MINIMAX_API_KEY": api_key, "MINIMAX_BASE_URL": base_url,
                "TERMINAL_ENV": "docker", "TERMINAL_CWD": task_workspace,
                "TERMINAL_DOCKER_IMAGE": docker_image,
                **bridge_env,
            }
            agent_env.update(otel_env(
                service_name="agent-bench-hermes-roadmap", run_id=run_id,
                endpoint=os.environ.get("AB_TELEMETRY_ENDPOINT", f"http://{self.container.host_address}:4318"),
                extra_attrs={"hm.run_id": run_id, "roadmap.instance_id": instance_id},
            ))
            agent_spec = ContainerSpec(
                name=agent_name, image=self.oc_image, network=net_name, user="0",
                mounts=(Mount(state, "/opt/data"), Mount(app_dir, task_workspace), *bridge_mounts),
                env=agent_env, workdir=task_workspace,
                extra_hosts={"vllm-host": "host-gateway"},
            )
            agent_handle = self.container.start(agent_spec)
            return {
                "container_id": agent_handle.id, "container_name": agent_handle.name,
                "hm_container_id": agent_handle.id,
                "task_container_id": task_handle.id, "task_container_name": task_handle.name,
                "backend": self.container.backend_id, "harness": "hermes",
                "network_name": net_name, "workspace_root": task_workspace,
                "task_workspace_root": task_workspace,
                "workspace_host_path": str(app_dir), "host_logs_dir": str(logs_dir),
                "state_dir": str(state), "sandbox_image": docker_image,
                "agent_env": agent_env,
                "tb_instruction": instance_data["instruction"],
                "tb_verifier_entrypoint": instance_data.get("verifier_entrypoint", "/tests/test.sh"),
                "tb_reward_file": instance_data.get("reward_file", "/logs/verifier/reward.txt"),
                "prepare_phase_timing": {
                    "workspace_prepare_start_ns": prepare_start_ns,
                    "workspace_prepare_end_ns": prepare_end_ns,
                    "workspace_prepare_seconds": (prepare_end_ns - prepare_start_ns) / 1e9,
                },
                **dispatcher.env_info(),
            }
        except BaseException:
            with self._task_specs_lock:
                self._task_specs.pop(task_name, None)
            if dispatcher:
                dispatcher.stop()
            self._best_effort_stop(agent_handle, task_handle)
            self.container.remove_network(net_name)
            raise

    def exec_in_container(self, container_id: str, argv: list[str], **kwargs):
        timeout = kwargs.pop("timeout", None) or kwargs.pop("timeout_seconds", None)
        return self.container.exec(
            ContainerHandle(container_id, container_id, self.container.backend_id),
            argv, timeout=timeout, **kwargs,
        )

    def quiesce_orphan_children(self, env_info: Dict) -> Dict:
        task_name = str(env_info.get("task_container_name") or "")
        agent_id = str(env_info.get("hm_container_id") or env_info.get("container_id") or "")
        if not task_name or not agent_id:
            raise RuntimeError("missing Hermes or task handle for candidate freeze")
        with self._task_specs_lock:
            task_spec = self._task_specs.get(task_name)
        if task_spec is None:
            raise RuntimeError(f"missing saved task spec for {task_name}")
        OpenClawRemoteDispatcher.stop_external(
            pid=env_info.get("bridge_pid"), socket_path=env_info.get("bridge_socket")
        )
        self._wait_for_process_exit(env_info.get("bridge_pid"))
        env_info["bridge_pid"] = None
        env_info["bridge_socket"] = None
        agent = ContainerHandle(agent_id, agent_id, self.container.backend_id)
        self.container.stop(agent)
        if self.container.is_running(agent):
            raise RuntimeError("Hermes runtime remained active after stop")
        task = ContainerHandle(str(env_info.get("task_container_id") or task_name), task_name, self.container.backend_id)
        self.container.stop(task)
        if self.container.is_running(task):
            raise RuntimeError("task runtime remained active after stop")
        restarted = self.container.start(task_spec)
        if not self.container.is_running(restarted):
            raise RuntimeError("restarted task runtime is not active")
        env_info["task_container_id"] = restarted.id
        env_info["task_container_name"] = restarted.name
        env_info["roadmap_workspace_quiesced"] = True
        return {
            "workspaceQuiesced": True,
            "quiescenceMethod": "stop-hermes-and-restart-task-runtime",
            "hermesRuntimeStopped": True,
            "execBridgeStopped": True,
            "taskRuntimeRestarted": True,
        }
