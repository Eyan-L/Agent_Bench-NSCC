"""Unmodified portable Codex CLI runs inside each task's own runtime."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path

from agent_bench.benchmarks.conditions.gateway_helpers import pick_free_gateway_port, release_gateway_port
from agent_bench.benchmarks.swebench.app import _force_rmtree
from agent_bench.benchmarks.terminal_bench.app import TerminalBenchOpenClawHarness, _docker_image_workdir, _resolve_task_workspace, _seed_workspace_helpers
from agent_bench.core.config import load_active_model_and_serving, runtime_endpoint_for_container
from agent_bench.core.profile_io import load_profile
from agent_bench.deployments.containers.limits import configured_task_container_cpus
from agent_bench.deployments.containers import ContainerHandle, ContainerSpec, Mount
from agent_bench.harnesses.codex.runtime import PAYLOAD_PATH, STATE_PATH, write_config


class TerminalBenchCodexHarness(TerminalBenchOpenClawHarness):
    harness_name = "codex"

    def __init__(self, benchmark_data_root=None, *, container, runtime_image=None, workspace_base=None):
        image = runtime_image or os.environ.get("AB_AGENT_RUNTIME_SIF" if container.backend_id == "singularity" else "AB_AGENT_RUNTIME_IMAGE") or "agent-bench/codex:0.157.1"
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.container = container
        self.runtime_image = self.oc_image = str(image)
        self.container.check_available()
        self.container.check_image(self.runtime_image)
        self.task_container_cpus = configured_task_container_cpus()
        base = Path(workspace_base or os.environ.get("AB_HARNESS_WORKSPACE_BASE", "~/.cache/agent-bench/workspaces")).expanduser()
        self.workspace_base = base / "codex"
        self.state_base = base.parent / "states/codex"
        self.payload_base = base / "codex-payloads"
        self._task_specs = {}
        self._task_specs_lock = threading.Lock()
        for directory in (self.workspace_base, self.state_base, self.payload_base):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _task_container_name(run_id):
        return f"cx-task-{run_id[:12]}"

    def _payload(self):
        image = Path(self.runtime_image)
        identity = self.runtime_image
        if image.is_file():
            info = image.stat()
            identity += f":{info.st_size}:{info.st_mtime_ns}"
        destination = self.payload_base / hashlib.sha256(identity.encode()).hexdigest()[:16]
        with (self.payload_base / f".{destination.name}.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if (destination / "bin/codex").is_file():
                return destination
            temporary = destination.with_name(destination.name + f".{os.getpid()}.new")
            self.container.copy_from_image(self.runtime_image, "/opt/codex", temporary)
            if not (temporary / "bin/codex").is_file():
                raise RuntimeError("Codex SIF missing /opt/codex/bin/codex")
            os.replace(temporary, destination)
        return destination

    def prepare_environment(self, instance_id, repo_id, base_commit, container_image=None,
                            workspace_root=None, run_id=None, instance_data=None):
        if not instance_data or instance_data.get("benchmark") not in {"terminal_bench", "roadmap_bench"}:
            raise ValueError("Codex requires Terminal-Bench or RoadmapBench instance_data")
        run_id = run_id or instance_id
        docker_image = instance_data["docker_image"]
        workspace = _resolve_task_workspace(instance_data, _docker_image_workdir(docker_image) if self.container.backend_id == "docker" else None)
        task_image = self._resolve_task_image(docker_image)
        task_name, adapter_name, network = self._task_container_name(run_id), f"cx-api-{run_id[:12]}", f"cx-net-{run_id[:12]}"
        task_handle = adapter_handle = None
        adapter_port = None
        root, state = self.workspace_base / run_id, self.state_base / run_id
        self._best_effort_stop(ContainerHandle(task_name, task_name, self.container.backend_id), ContainerHandle(adapter_name, adapter_name, self.container.backend_id))
        _force_rmtree(root, ignore_errors=True)
        _force_rmtree(state, ignore_errors=True)
        app, logs, tests, codex_state = root / "app", root / "logs/verifier", root / "tests", state / "codex"
        for path in (app, logs, tests, codex_state):
            path.mkdir(parents=True, exist_ok=True)
        self.container.remove_network(network)
        self.container.create_network(network)
        try:
            prepare_start = time.time_ns()
            self.container.copy_from_image(task_image, workspace, app)
            _seed_workspace_helpers(app, instance_data, workspace)
            payload = self._payload()
            repo = Path(os.environ.get("AB_REPO_ROOT", Path.cwd()))
            profile = load_profile(os.environ.get("AB_HARNESS_CONFIG", str(repo / "profiles/harnesses/codex_terminal_bench.yaml")))
            if profile.get("harness") != "codex":
                raise ValueError("Codex requires a Codex harness profile")
            settings = profile.get("codex") or {}
            model, _ = load_active_model_and_serving(repo_root=str(repo))
            model_name = os.environ.get("AB_MODEL_NAME") or model.name
            upstream = runtime_endpoint_for_container(os.environ.get("AB_SERVING_BASE_URL") or model.endpoint, self.container.host_address)
            adapter_port, token = pick_free_gateway_port(), "sk-agent-bench-" + secrets.token_hex(16)
            adapter_config = {"token": token, "base_url": upstream, "api_key": os.environ.get("AB_SERVING_API_KEY") or model.api_key or "EMPTY",
                              "model": model_name, "temperature": model.temperature, "max_tokens": model.max_tokens,
                              "upstream_api": settings.get("upstream_api", "responses")}
            if settings.get("chat_template_kwargs") is not None:
                adapter_config["chat_template_kwargs"] = settings["chat_template_kwargs"]
            adapter_config["require_done_marker"] = bool(settings.get("require_done_marker", False))
            (state / "adapter.json").write_text(json.dumps(adapter_config))
            (state / "adapter.json").chmod(0o600)
            host = adapter_name if self.container.backend_id == "docker" else "127.0.0.1"
            write_config(codex_state, model=model_name, base_url=f"http://{host}:{adapter_port}/v1", context_window=model.context_window, settings=settings)
            mounts = [Mount(app, workspace), Mount(logs, "/logs/verifier"), Mount(payload, PAYLOAD_PATH, "ro"), Mount(codex_state, STATE_PATH)]
            if self.container.backend_id == "singularity":
                mounts.append(Mount(tests, "/tests"))
            spec = ContainerSpec(name=task_name, image=task_image, mounts=tuple(mounts), env={"HOME": "/root"}, network=network,
                                 workdir=workspace, entrypoint="sleep", command=("infinity",), user="0", cpus=self.task_container_cpus)
            with self._task_specs_lock:
                self._task_specs[task_name] = spec
            task_handle = self.container.start(spec)
            adapter_spec = ContainerSpec(name=adapter_name, image=self.runtime_image, network=network,
                mounts=(Mount(state, "/opt/state"), Mount(repo / "agent_bench/harnesses/codex", "/opt/adapter", "ro")),
                entrypoint="python3", command=("/opt/adapter/responses_adapter.py", "--config", "/opt/state/adapter.json", "--state", "/opt/state", "--port", str(adapter_port)),
                env={"HOME": "/root"}, extra_hosts={"vllm-host": "host-gateway"})
            adapter_handle = self.container.start(adapter_spec)
            probe = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('sys').argv[1] + '/health',timeout=3).read()"
            deadline = time.monotonic() + 90
            while True:
                if not self.container.is_running(adapter_handle):
                    raise RuntimeError("Codex adapter exited: " + self.container.logs(adapter_handle)[-2000:])
                result = self.container.exec(adapter_handle, ["python3", "-c", probe, str(adapter_port)], check=False, timeout=10)
                if result.returncode == 0:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Codex adapter readiness timeout")
                time.sleep(1)
            env = {"HOME": "/root", "CODEX_HOME": STATE_PATH, "AB_CODEX_API_KEY": token, "AB_RUN_ID": run_id,
                   "PATH": f"{PAYLOAD_PATH}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
            version = self.container.exec(task_handle, [f"{PAYLOAD_PATH}/bin/codex", "--version"], env=env, check=True, timeout=30)
            return {"container_id": task_handle.id, "container_name": task_handle.name, "task_container_id": task_handle.id, "task_container_name": task_handle.name,
                "adapter_container_id": adapter_handle.id, "adapter_log_path": str(adapter_handle.log_path) if adapter_handle.log_path else None,
                "adapter_port": adapter_port, "network_name": network, "backend": self.container.backend_id,
                "harness": "codex", "workspace_root": workspace, "task_workspace_root": workspace, "workspace_host_path": str(app), "host_logs_dir": str(logs),
                "state_dir": str(state), "codex_state_dir": str(codex_state), "agent_env": env, "runtime_version": version.stdout.strip(), "upstream_api": adapter_config["upstream_api"],
                "prepare_phase_timing": {"workspace_prepare_start_ns": prepare_start, "workspace_prepare_end_ns": time.time_ns(), "workspace_prepare_seconds": (time.time_ns() - prepare_start) / 1e9}}
        except BaseException:
            with self._task_specs_lock:
                self._task_specs.pop(task_name, None)
            if adapter_port is not None:
                release_gateway_port(adapter_port)
            self._best_effort_stop(adapter_handle, task_handle)
            self.container.remove_network(network)
            raise

    def quiesce_agent_writers(self, env_info):
        name = env_info["task_container_name"]
        with self._task_specs_lock:
            spec = self._task_specs.get(name)
        if spec is None:
            raise RuntimeError("Missing Codex task spec for writer fencing")
        handle = ContainerHandle(env_info["task_container_id"], name, self.container.backend_id)
        self.container.stop(handle)
        if self.container.is_running(handle):
            raise RuntimeError("Codex task remained active after stop")
        adapter = ContainerHandle(env_info["adapter_container_id"], env_info["adapter_container_id"], self.container.backend_id)
        self.container.stop(adapter)
        if self.container.is_running(adapter):
            raise RuntimeError("Codex API recorder remained active after stop")
        restarted = self.container.start(spec)
        if not self.container.is_running(restarted):
            raise RuntimeError("Codex task restart failed")
        env_info["task_container_id"] = restarted.id
        env_info["roadmap_workspace_quiesced"] = True
        return {"workspaceQuiesced": True, "workspaceQuiescedNs": time.time_ns(),
                "quiescenceMethod": "stop-codex-task-and-restart", "codexWritersStopped": True, "taskRuntimeRestarted": True}

    def cleanup(self, container_id=None, *, adapter_container_id=None, adapter_port=None, **kwargs):
        if adapter_port is not None:
            release_gateway_port(adapter_port)
        if adapter_container_id:
            self._best_effort_stop(ContainerHandle(adapter_container_id, adapter_container_id, self.container.backend_id))
        super().cleanup(container_id, **kwargs)
