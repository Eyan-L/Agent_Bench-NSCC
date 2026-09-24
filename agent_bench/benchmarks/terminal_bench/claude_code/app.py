"""Terminal-Bench harness for an unmodified Claude Code runtime.

Claude Code must execute inside the task runtime so its Bash and file tools see
the same packages and filesystem as the verifier.  The separately prepared
runtime image therefore serves two purposes:

* it runs a LiteLLM Anthropic-to-OpenAI adapter beside SGLang; and
* it supplies a read-only, portable Claude Code payload mounted into the task
  runtime.

No verifier files are exposed until :meth:`run_tests` injects ``/tests``.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from agent_bench.benchmarks.conditions.gateway_helpers import (
    pick_free_gateway_port,
    release_gateway_port,
)
from agent_bench.benchmarks.swebench.app import _force_rmtree
from agent_bench.benchmarks.terminal_bench.app import (
    TerminalBenchOpenClawHarness,
    _docker_image_workdir,
    _normalize_container_path,
    _resolve_task_workspace,
    _seed_workspace_helpers,
)
from agent_bench.core.config import (
    load_active_model_and_serving,
    runtime_endpoint_for_container,
)
from agent_bench.deployments.containers import (
    Container,
    ContainerHandle,
    ContainerSpec,
    Mount,
)
from agent_bench.deployments.containers.limits import configured_task_container_cpus

logger = logging.getLogger(__name__)


class TerminalBenchClaudeCodeHarness(TerminalBenchOpenClawHarness):
    """Run Claude Code in the task instance with a sidecar protocol adapter."""

    backend_id = "container-agnostic"
    harness_name = "claude_code"
    _PAYLOAD_IN_TASK = "/tmp/agent-bench-claude-code"
    _STATE_IN_TASK = "/tmp/agent-bench-claude-state"
    _ADAPTER_READY_TIMEOUT_SECONDS = 180

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        *,
        container: Container,
        runtime_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ) -> None:
        self.benchmark_data_root = (
            Path(benchmark_data_root) if benchmark_data_root else None
        )
        self.container = container
        if self.container.backend_id == "singularity":
            self.runtime_image = runtime_image or os.environ.get(
                "AB_AGENT_RUNTIME_SIF", ""
            )
            if not self.runtime_image:
                raise RuntimeError(
                    "AB_AGENT_RUNTIME_SIF is required for Terminal-Bench "
                    "Claude Code runs with the singularity backend."
                )
        else:
            self.runtime_image = runtime_image or os.environ.get(
                "AB_AGENT_RUNTIME_IMAGE", "agent-bench/claude-code:2.1.278"
            )
        self.container.check_available()
        self.container.check_image(self.runtime_image)
        self.task_container_cpus = configured_task_container_cpus()
        self.workspace_base = Path(
            workspace_base
            or os.environ.get(
                "AB_HARNESS_WORKSPACE_BASE", "~/.cache/agent-bench/workspaces"
            )
        ).expanduser()
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base = self.workspace_base.parent / "states"
        self.state_base.mkdir(parents=True, exist_ok=True)
        self.payload_base = self.workspace_base.parent / "runtime-payloads"
        self.payload_base.mkdir(parents=True, exist_ok=True)
        self._task_specs: Dict[str, ContainerSpec] = {}
        self._task_specs_lock = threading.Lock()

    @staticmethod
    def _task_container_name(run_id: str) -> str:
        return f"cc-tb-task-{run_id[:12]}"

    @staticmethod
    def _adapter_container_name(run_id: str) -> str:
        return f"cc-adapter-{run_id[:12]}"

    def _payload_cache_key(self) -> str:
        image = Path(self.runtime_image)
        if image.is_file():
            stat = image.stat()
            identity = f"{image.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        else:
            identity = self.runtime_image
        return hashlib.sha256(identity.encode()).hexdigest()[:16]

    def _materialize_runtime_payload(self) -> Path:
        """Extract the immutable Claude payload once per runtime image."""
        payload = self.payload_base / f"claude-code-{self._payload_cache_key()}"
        marker = payload / ".agent-bench-payload-ready"
        lock_path = self.payload_base / f".{payload.name}.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if marker.is_file() and (payload / "bin" / "claude").exists():
                return payload
            temporary = payload.with_name(f".{payload.name}.{os.getpid()}.new")
            _force_rmtree(temporary, ignore_errors=True)
            self.container.copy_from_image(
                self.runtime_image,
                "/opt/claude-code",
                temporary,
            )
            claude = temporary / "bin" / "claude"
            if not claude.exists():
                _force_rmtree(temporary, ignore_errors=True)
                raise RuntimeError(
                    "Claude Code runtime payload is missing /opt/claude-code/bin/claude"
                )
            (temporary / ".agent-bench-payload-ready").write_text(
                f"source={self.runtime_image}\n", encoding="utf-8"
            )
            _force_rmtree(payload, ignore_errors=True)
            os.replace(temporary, payload)
        return payload

    def _write_adapter_config(
        self,
        state_dir: Path,
        *,
        adapter_token: str,
    ) -> tuple[Path, str]:
        model_cfg, _serving_cfg = load_active_model_and_serving(
            repo_root=os.environ.get("AB_REPO_ROOT", os.getcwd())
        )
        endpoint = (
            os.environ.get("AB_SERVING_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or model_cfg.endpoint
        )
        endpoint = runtime_endpoint_for_container(
            endpoint, self.container.host_address
        ).rstrip("/")
        model_name = os.environ.get("AB_MODEL_NAME") or model_cfg.name
        upstream_model = model_name
        if not upstream_model.startswith("openai/"):
            upstream_model = f"openai/{upstream_model}"
        config = {
            "model_list": [
                {
                    "model_name": model_name,
                    "litellm_params": {
                        "model": upstream_model,
                        "api_base": endpoint,
                        "api_key": model_cfg.api_key or "EMPTY",
                        "drop_params": True,
                    },
                }
            ],
            "litellm_settings": {
                "drop_params": True,
                "request_timeout": 600,
            },
            "general_settings": {"master_key": adapter_token},
        }
        path = state_dir / "litellm.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        path.chmod(0o600)
        return path, model_name

    def _wait_adapter_ready(
        self,
        handle: ContainerHandle,
        *,
        port: int,
        token: str,
    ) -> None:
        deadline = time.monotonic() + self._ADAPTER_READY_TIMEOUT_SECONDS
        last_error = ""
        probe = (
            "import sys,urllib.request; "
            "token=sys.argv[1]; port=sys.argv[2]; "
            "req=urllib.request.Request("
            "f'http://127.0.0.1:{port}/health/liveliness', "
            "headers={'Authorization':f'Bearer {token}'}); "
            "sys.exit(0 if urllib.request.urlopen(req,timeout=3).status < 500 else 1)"
        )
        while time.monotonic() < deadline:
            if not self.container.is_running(handle):
                raise RuntimeError(
                    "Claude Code LiteLLM adapter exited before readiness: "
                    + self.container.logs(handle)[-2000:]
                )
            try:
                result = self.container.exec(
                    handle,
                    [
                        "/opt/litellm/bin/python",
                        "-c",
                        probe,
                        token,
                        str(port),
                    ],
                    check=False,
                    timeout=10,
                )
                if result.returncode == 0:
                    return
                last_error = (result.stderr or result.stdout or "").strip()
            except Exception as exc:
                last_error = str(exc)
            time.sleep(1)
        raise RuntimeError(
            "Claude Code LiteLLM adapter did not become ready within "
            f"{self._ADAPTER_READY_TIMEOUT_SECONDS}s: {last_error}; "
            f"log tail={self.container.logs(handle)[-2000:]}"
        )

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
        if not instance_data or instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError(
                "TerminalBenchClaudeCodeHarness requires terminal_bench instance_data"
            )

        docker_image = instance_data["docker_image"]
        if self.container.backend_id == "docker":
            image_workdir = _docker_image_workdir(docker_image)
            task_workspace = _resolve_task_workspace(instance_data, image_workdir)
        else:
            task_workspace = _resolve_task_workspace(instance_data)
        task_image = self._resolve_task_image(docker_image)
        run_id = run_id or instance_id
        task_name = self._task_container_name(run_id)
        adapter_name = self._adapter_container_name(run_id)
        network_name = f"cc-tb-net-{run_id[:12]}"

        self._best_effort_stop(
            ContainerHandle(task_name, task_name, self.container.backend_id),
            ContainerHandle(adapter_name, adapter_name, self.container.backend_id),
        )
        self.container.remove_network(network_name)
        self.container.create_network(network_name)

        run_root = self.workspace_base / run_id
        state_dir = self.state_base / run_id
        _force_rmtree(run_root, ignore_errors=True)
        _force_rmtree(state_dir, ignore_errors=True)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        tests_dir = run_root / "tests"
        claude_state_dir = state_dir / "claude"
        for path in (app_dir, logs_dir, tests_dir, claude_state_dir):
            path.mkdir(parents=True, exist_ok=True)

        workspace_prepare_start_ns = time.time_ns()
        self.container.copy_from_image(task_image, task_workspace, app_dir)
        _seed_workspace_helpers(app_dir, instance_data, task_workspace)
        payload_dir = self._materialize_runtime_payload()
        workspace_prepare_end_ns = time.time_ns()

        adapter_port = pick_free_gateway_port()
        adapter_token = f"sk-agent-bench-{secrets.token_hex(16)}"
        adapter_config, model_name = self._write_adapter_config(
            state_dir, adapter_token=adapter_token
        )
        apt_config = state_dir / "apt-singularity.conf"
        if self.container.backend_id == "singularity":
            # Root-mapped namespaces on NSCC cannot switch to Debian's _apt
            # account. Keep APT in mapped root while writable-tmpfs supplies
            # an ephemeral package-install layer for agent and verifier use.
            apt_config.write_text('APT::Sandbox::User "root";\n', encoding="utf-8")
        model_cfg, _serving_cfg = load_active_model_and_serving(
            repo_root=os.environ.get("AB_REPO_ROOT", os.getcwd())
        )

        task_handle: Optional[ContainerHandle] = None
        adapter_handle: Optional[ContainerHandle] = None
        try:
            task_mounts = [
                Mount(app_dir, task_workspace, "rw"),
                Mount(logs_dir, "/logs/verifier", "rw"),
                Mount(payload_dir, self._PAYLOAD_IN_TASK, "ro"),
                Mount(claude_state_dir, self._STATE_IN_TASK, "rw"),
            ]
            if self.container.backend_id == "singularity":
                task_mounts.append(Mount(tests_dir, "/tests", "rw"))
                task_mounts.append(
                    Mount(
                        apt_config,
                        "/etc/apt/apt.conf.d/99agent-bench-singularity",
                        "ro",
                    )
                )
            task_spec = ContainerSpec(
                name=task_name,
                image=task_image,
                network=network_name,
                user="0",
                mounts=tuple(task_mounts),
                workdir=task_workspace,
                env={"HOME": "/root"},
                entrypoint="sleep",
                command=("infinity",),
                cpus=self.task_container_cpus,
                writable_tmpfs=self.container.backend_id == "singularity",
                fakeroot=self.container.backend_id == "singularity",
                labels={"claude-code.task": "1", "agent-bench.run_id": run_id},
            )
            with self._task_specs_lock:
                self._task_specs[task_name] = task_spec
            task_handle = self.container.start(task_spec)

            adapter_spec = ContainerSpec(
                name=adapter_name,
                image=self.runtime_image,
                network=network_name,
                user="0",
                mounts=(Mount(state_dir, "/etc/agent-bench", "rw"),),
                workdir="/tmp",
                env={
                    "HOME": "/root",
                    "LITELLM_MASTER_KEY": adapter_token,
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                },
                extra_hosts={"vllm-host": "host-gateway"},
                entrypoint="/opt/litellm/bin/litellm",
                command=(
                    "--config",
                    "/etc/agent-bench/litellm.json",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(adapter_port),
                ),
                labels={
                    "claude-code.adapter": "1",
                    "agent-bench.run_id": run_id,
                },
            )
            adapter_handle = self.container.start(adapter_spec)
            self._wait_adapter_ready(
                adapter_handle, port=adapter_port, token=adapter_token
            )

            if self.container.backend_id == "docker":
                adapter_host = adapter_handle.name
            else:
                adapter_host = "127.0.0.1"
            adapter_url = f"http://{adapter_host}:{adapter_port}"
            agent_env = {
                "HOME": "/root",
                "PATH": f"{self._PAYLOAD_IN_TASK}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "CLAUDE_CONFIG_DIR": self._STATE_IN_TASK,
                "ANTHROPIC_BASE_URL": adapter_url,
                "ANTHROPIC_AUTH_TOKEN": adapter_token,
                "ANTHROPIC_API_KEY": adapter_token,
                "ANTHROPIC_MODEL": model_name,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model_name,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model_name,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_name,
                "CLAUDE_CODE_SUBAGENT_MODEL": model_name,
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(model_cfg.max_tokens),
                "CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY": "16",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                "FORCE_AUTO_BACKGROUND_TASKS": "1",
                "ENABLE_BACKGROUND_TASKS": "1",
                "IS_SANDBOX": "1",
                "AB_RUN_ID": run_id,
            }
            return {
                "prepare_phase_timing": {
                    "workspace_prepare_start_ns": workspace_prepare_start_ns,
                    "workspace_prepare_end_ns": workspace_prepare_end_ns,
                    "workspace_prepare_seconds": max(
                        0.0,
                        (workspace_prepare_end_ns - workspace_prepare_start_ns)
                        / 1e9,
                    ),
                },
                "container_id": task_handle.id,
                "container_name": task_handle.name,
                "claude_container_id": task_handle.id,
                "task_container_id": task_handle.id,
                "task_container_name": task_handle.name,
                "adapter_container_id": adapter_handle.id,
                "adapter_container_name": adapter_handle.name,
                "adapter_port": adapter_port,
                "adapter_url": adapter_url,
                "backend": self.container.backend_id,
                "network_name": network_name,
                "workspace_root": task_workspace,
                "task_workspace_root": task_workspace,
                "workspace_host_path": str(app_dir),
                "host_logs_dir": str(logs_dir),
                "state_dir": str(state_dir),
                "claude_state_dir": str(claude_state_dir),
                "sandbox_image": docker_image,
                "runtime_image": self.runtime_image,
                "agent_env": agent_env,
                "tb_instruction": instance_data["instruction"],
                "tb_verifier_entrypoint": instance_data.get(
                    "verifier_entrypoint", "/tests/test.sh"
                ),
                "tb_reward_file": instance_data.get(
                    "reward_file", "/logs/verifier/reward.txt"
                ),
            }
        except BaseException:
            with self._task_specs_lock:
                self._task_specs.pop(task_name, None)
            release_gateway_port(adapter_port)
            self._best_effort_stop(adapter_handle, task_handle)
            self.container.remove_network(network_name)
            raise

    def cleanup(
        self,
        container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
        adapter_container_id: Optional[str] = None,
        adapter_container_name: Optional[str] = None,
        network_name: Optional[str] = None,
        adapter_port: Optional[int] = None,
        workspace_host_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        **_kw,
    ) -> None:
        if adapter_port is not None:
            release_gateway_port(adapter_port)
        if task_container_name:
            with self._task_specs_lock:
                self._task_specs.pop(task_container_name, None)
        for ident in dict.fromkeys(
            item
            for item in (
                adapter_container_id,
                adapter_container_name,
                container_id,
                task_container_id,
                task_container_name,
            )
            if item
        ):
            self._best_effort_stop(
                ContainerHandle(ident, ident, self.container.backend_id)
            )
        if network_name:
            self.container.remove_network(network_name)
        if os.environ.get("AB_HARNESS_KEEP_WORKSPACE", "0") != "1":
            if workspace_host_path:
                _force_rmtree(Path(workspace_host_path).parent, ignore_errors=True)
            if state_dir:
                _force_rmtree(Path(state_dir), ignore_errors=True)
