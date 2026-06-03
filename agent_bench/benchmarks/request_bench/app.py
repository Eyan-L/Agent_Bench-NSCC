"""Request-Bench harnesses.

Request-Bench is a diagnostic app for natural-language prompts. It keeps the
same benchmark/runtime shape as the other apps while avoiding Terminal-Bench or
SWE-bench semantics: the agent reads /workspace/PROMPT.md, writes
/workspace/answer.md, and the harness marks PASS when that artifact is
non-empty.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import string
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent_bench.benchmarks.conditions.gateway_helpers import (
    pick_free_gateway_port as _pick_free_gateway_port,
    release_gateway_port as _release_gateway_port,
)
from agent_bench.benchmarks.swebench.app import (
    DEFAULT_OPENCLAW_RUNTIME_IMAGE,
    _force_rmtree,
    _oc_sandbox_name,
)
from agent_bench.core.config import endpoint_for_container, load_active_model_and_serving
from agent_bench.core.profile_io import load_profile
from agent_bench.harnesses.hermes.dispatcher import HermesDispatcher
from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = os.environ.get(
    "REQUEST_BENCH_SANDBOX_IMAGE", "python:3.13-slim-bookworm"
)
DEFAULT_HM_DOCKER_IMAGE = os.environ.get(
    "HM_IMAGE", "agent-bench/hermes:v2026.5.16"
)
_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"
_WORKSPACE = "/workspace"
_STATE_DIR_IN_HERMES = "/opt/data"


def _docker_available() -> None:
    r = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"docker not available: {r.stderr.strip()}")


def _ensure_docker_image(image: str, *, pull: bool = True) -> None:
    r = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if r.returncode == 0:
        return
    if not pull:
        raise RuntimeError(f"docker image not found: {image}")
    logger.info("  docker image %s not local, pulling", image)
    r = subprocess.run(
        ["docker", "pull", image], capture_output=True, text=True, timeout=600
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"docker image pull failed: {image}\n{r.stderr.strip()}"
        )


def _request_task_name(run_id: str) -> str:
    return f"rb-sbx-{run_id[:12]}"


def _write_prompt_workspace(ws_host: Path, instance_data: Dict) -> None:
    prompt = instance_data.get("prompt") or instance_data.get("request") or ""
    description = instance_data.get("description") or "Natural request diagnostic task"
    instance_id = instance_data.get("instance_id", "")
    ws_host.mkdir(parents=True, exist_ok=True)
    (ws_host / "PROMPT.md").write_text(
        f"# Request-Bench task {instance_id}\n\n"
        f"description: {description}\n\n"
        "## Request\n\n"
        f"{prompt}\n",
        encoding="utf-8",
    )
    (ws_host / "AGENTS.md").write_text(
        "Complete the request in PROMPT.md. BOOTSTRAP.md is present in "
        "this workspace with the run contract. Write the final answer to "
        "/workspace/answer.md as Markdown.\n",
        encoding="utf-8",
    )
    (ws_host / "BOOTSTRAP.md").write_text(
        "# BOOTSTRAP.md\n\n"
        "1. Read /workspace/PROMPT.md for the user request.\n"
        "2. Produce a clear, complete Markdown answer.\n"
        "3. Write the final answer to /workspace/answer.md.\n"
        "4. Reply with DONE only after /workspace/answer.md exists.\n",
        encoding="utf-8",
    )

def build_answer_verdict(
    *,
    instance_id: str,
    answer_path: Path,
    output_dir: Path,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        answer_text = answer_path.read_text(encoding="utf-8")
    except OSError as exc:
        answer_text = ""
        error_message = f"answer.md missing: {exc}"
    else:
        error_message = "empty answer.md" if not answer_text.strip() else None

    host_answer = output_dir / "answer.md"
    if answer_text:
        host_answer.write_text(answer_text, encoding="utf-8")
    status = "PASS" if answer_text.strip() else "FAIL"
    return {
        "status": status,
        "test_pass_rate": 1.0 if status == "PASS" else 0.0,
        "passed": 1 if status == "PASS" else 0,
        "failed": 0 if status == "PASS" else 1,
        "answer_path": str(host_answer),
        "answer_text": answer_text,
        "evaluation_dir": str(output_dir),
        "error_message": error_message,
        "instance_id": instance_id,
    }


class RequestBenchHarness:
    """OpenClaw two-container Docker harness for Request-Bench."""

    backend_id = "docker"
    _GATEWAY_READY_TIMEOUT_SECONDS = 180

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        oc_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ):
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.oc_image = oc_image or DEFAULT_OPENCLAW_RUNTIME_IMAGE
        _docker_available()
        self.workspace_base = Path(
            workspace_base
            or os.environ.get(
                "OC_HOST_WORKSPACE_BASE",
                os.path.expanduser("~/.cache/oc-eval/workspaces"),
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base = self.workspace_base.parent / "states"
        self.state_base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _agent_id_from_harness_cfg() -> str:
        path = os.environ.get("AB_HARNESS_CONFIG")
        if path and os.path.isfile(path):
            try:
                cfg = load_profile(path)
                agents = ((cfg.get("openclaw") or {}).get("agents") or {}).get("list") or []
                for agent in agents:
                    if isinstance(agent, dict) and agent.get("id"):
                        return str(agent["id"])
            except Exception:
                pass
        return "request_writer"

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
            raise RuntimeError("RequestBenchHarness.prepare_environment requires instance_data")
        if instance_data.get("benchmark") != "request_bench":
            raise RuntimeError(
                "RequestBenchHarness only handles benchmark=request_bench "
                f"(got {instance_data.get('benchmark')!r})"
            )
        docker_image = container_image or instance_data.get("docker_image") or DEFAULT_SANDBOX_IMAGE
        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"rb-eval-net-{rid_short}"
        oc_name = f"oc-rb-{rid_short}"
        agent_id = self._agent_id_from_harness_cfg()
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        scope_key = f"agent:{agent_id}"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        _ensure_docker_image(self.oc_image, pull=False)
        _ensure_docker_image(docker_image)

        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        ws_host = run_root / "workspace"
        _write_prompt_workspace(ws_host, instance_data)

        dispatcher: Optional[OpenClawRemoteDispatcher] = None
        state_dir: Optional[Path] = None
        gateway_port: Optional[int] = None

        def rollback() -> None:
            if gateway_port is not None:
                _release_gateway_port(gateway_port)
            if dispatcher is not None:
                try:
                    dispatcher.stop()
                except Exception:
                    pass
            self._best_effort_cleanup(net_name, oc_name, task_container_name)

        try:
            r = subprocess.run(
                ["docker", "network", "create", net_name],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0 and "already exists" not in (r.stderr or ""):
                raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

            state_dir = self._build_state_dir(
                run_id=run_id,
                workspace_root=_WORKSPACE,
                sandbox_image=docker_image,
                sandbox_binds=[f"{ws_host}:{_WORKSPACE}:rw"],
                container_prefix=sandbox_prefix,
                agent_id=agent_id,
            )
            dispatcher = OpenClawRemoteDispatcher()
            dispatcher.start(state_dir=state_dir, backend="docker", target=task_container_name)

            r = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", task_container_name,
                    "--network", net_name,
                    "--user", "0",
                    "--label", "openclaw.sandbox=1",
                    "--label", f"openclaw.sessionKey={scope_key}",
                    "--label", f"oc-eval.run_id={run_id}",
                    "-v", f"{ws_host}:{_WORKSPACE}:rw",
                    "-w", _WORKSPACE,
                    "--entrypoint", "sleep",
                    docker_image,
                    "infinity",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(f"request sandbox start failed: {r.stderr.strip()}")
            task_container_id = r.stdout.strip()

            gateway_port = _pick_free_gateway_port()
            gateway_token = secrets.token_hex(24)
            bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
            bridge_args: List[str] = []
            for mount in bridge_mounts:
                bridge_args += ["-v", f"{mount.host}:{mount.container}:{mount.mode}"]
            for key, value in bridge_env.items():
                bridge_args += ["-e", f"{key}={value}"]

            r = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", oc_name,
                    "--network", net_name,
                    "--add-host", "vllm-host:host-gateway",
                    "--user", "0",
                    "-v", f"{state_dir}:/root/.openclaw:rw",
                    "-v", "/var/run/docker.sock:/var/run/docker.sock",
                    "-w", "/root",
                    "-e", f"OPENCLAW_GATEWAY_TOKEN={gateway_token}",
                    "-e", f"OPENCLAW_GATEWAY_OTEL_ENDPOINT={os.environ.get('OC_OTEL_ENDPOINT', 'http://vllm-host:4318')}",
                    "-e", f"OPENCLAW_RUN_ID={run_id}",
                    "-e", f"OTEL_RESOURCE_ATTRIBUTES=oc.run_id={run_id}",
                    "-e", "DOCKER_API_VERSION=1.44",
                    *bridge_args,
                    "--entrypoint", "/usr/local/bin/openclaw",
                    self.oc_image,
                    "gateway", "--allow-unconfigured",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(f"openclaw container start failed: {r.stderr.strip()}")
            oc_container_id = r.stdout.strip()
            self._wait_for_gateway_ready(oc_container_id)

            gw_env = (
                "export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:18789\n"
                f"export OPENCLAW_GATEWAY_TOKEN={gateway_token}\n"
                f"export OPENCLAW_RUN_ID={run_id}\n"
            )
            subprocess.run(
                [
                    "docker", "exec", oc_container_id, "bash", "-c",
                    f"cat > /tmp/oc_gateway_env <<'EOF'\n{gw_env}EOF",
                ],
                capture_output=True,
                text=True,
            )

            return {
                "container_id": oc_container_id,
                "container_name": oc_name,
                "oc_container_id": oc_container_id,
                "task_container_id": task_container_id,
                "task_container_name": task_container_name,
                "network_name": net_name,
                "workspace_root": _WORKSPACE,
                "workspace_host_path": str(ws_host),
                "state_dir": str(state_dir),
                "gateway_port": gateway_port,
                "gateway_token": gateway_token,
                "gateway_url": "ws://127.0.0.1:18789",
                "answer_file": instance_data.get("answer_file", "/workspace/answer.md"),
                **dispatcher.env_info(),
            }
        except Exception:
            rollback()
            raise

    def _build_state_dir(
        self,
        *,
        run_id: str,
        workspace_root: str,
        sandbox_image: str,
        sandbox_binds: List[str],
        container_prefix: str,
        agent_id: str,
    ) -> Path:
        repo_root = os.environ.get("AB_REPO_ROOT") or os.environ.get("KIT_ROOT") or os.getcwd()
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir, ignore_errors=True)
        state_dir.mkdir(parents=True)
        (state_dir / "agents" / agent_id).mkdir(parents=True, exist_ok=True)
        harness_path = os.environ.get(
            "AB_HARNESS_CONFIG",
            os.path.join(repo_root, "profiles/harnesses/openclaw_request_bench.yaml"),
        )
        if not os.path.isfile(harness_path):
            raise FileNotFoundError(f"request_bench harness config not found at {harness_path}")
        cfg = dict(load_profile(harness_path).get("openclaw") or {})
        if not cfg:
            raise ValueError(f"harness config {harness_path} has no openclaw block")
        model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
        providers = cfg.setdefault("models", {}).setdefault("providers", {})
        vllm = providers.setdefault("vllm", {})
        vllm["api"] = _OPENAI_CHAT_COMPLETIONS_API
        vllm["baseUrl"] = os.environ.get(
            "OC_VLLM_BASE_URL", endpoint_for_container(model_cfg.endpoint, "vllm-host")
        )
        vllm.setdefault("apiKey", model_cfg.api_key or "sk-dummy")
        vllm["models"] = [{
            "id": model_cfg.name,
            "name": model_cfg.name,
            "api": _OPENAI_CHAT_COMPLETIONS_API,
            "contextWindow": model_cfg.context_window,
            "maxTokens": model_cfg.max_tokens,
        }]
        cfg.setdefault("agents", {}).setdefault("defaults", {}).setdefault("model", {})[
            "primary"
        ] = f"vllm/{model_cfg.name}"
        agents_defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
        agents_defaults["workspace"] = workspace_root
        agents_defaults["sandbox"] = {
            "mode": "all",
            "scope": "agent",
            "backend": "docker",
            "workspaceAccess": "rw",
            "docker": {
                "image": sandbox_image,
                "containerPrefix": container_prefix,
                "workdir": workspace_root,
                "binds": list(sandbox_binds),
                "dangerouslyAllowExternalBindSources": True,
            },
        }
        cfg.setdefault("tools", {}).setdefault("exec", {})["host"] = "sandbox"
        otel_cfg = (cfg.get("diagnostics") or {}).get("otel") or {}
        if otel_cfg.get("enabled"):
            cfg["diagnostics"]["otel"]["endpoint"] = os.environ.get(
                "OC_OTEL_ENDPOINT", "http://vllm-host:4318"
            )
        (state_dir / "openclaw.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
        return state_dir

    def _wait_for_gateway_ready(self, oc_container_id: str) -> None:
        deadline = time.time() + self._GATEWAY_READY_TIMEOUT_SECONDS
        last_log_tail = ""
        while time.time() < deadline:
            r = subprocess.run(
                ["docker", "logs", oc_container_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            log_text = (r.stdout or "") + (r.stderr or "")
            if "ready" in log_text.lower() or "listening on" in log_text.lower():
                return
            last_log_tail = log_text[-1500:]
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", oc_container_id],
                capture_output=True,
                text=True,
            )
            if (r.stdout or "").strip() != "true":
                raise RuntimeError(
                    "openclaw container exited before gateway ready. "
                    f"last log:\n{last_log_tail}"
                )
            time.sleep(2)
        raise RuntimeError(
            f"gateway did not signal ready within {self._GATEWAY_READY_TIMEOUT_SECONDS}s. "
            f"last log:\n{last_log_tail}"
        )

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
        **_extra,
    ) -> Dict:
        run_id = run_id or instance_id
        host_run_root = self.workspace_base / run_id
        answer_name = Path((instance_data or {}).get("answer_file", "/workspace/answer.md")).name
        answer_path = host_run_root / "workspace" / answer_name
        output_dir = Path(run_dir) / "request_bench" if run_dir is not None else host_run_root / "verifier"
        return build_answer_verdict(
            instance_id=instance_id,
            answer_path=answer_path,
            output_dir=output_dir,
        )

    def cleanup(
        self,
        container_id: Optional[str] = None,
        oc_container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
        container_name: Optional[str] = None,
        network_name: Optional[str] = None,
        gateway_port: Optional[int] = None,
        workspace_host_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        bridge_pid: Optional[int] = None,
        bridge_socket: Optional[str] = None,
        **_extra,
    ) -> None:
        if gateway_port is not None:
            _release_gateway_port(gateway_port)
        OpenClawRemoteDispatcher.stop_external(pid=bridge_pid, socket_path=bridge_socket)
        uid, gid = os.getuid(), os.getgid()
        for cid in {oc_container_id, container_id}:
            if cid:
                subprocess.run(
                    ["docker", "exec", cid, "chown", "-R", f"{uid}:{gid}", "/root/.openclaw"],
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
        targets = [
            t for t in (oc_container_id, container_id, task_container_id, task_container_name, container_name) if t
        ]
        for cid in dict.fromkeys(targets):
            subprocess.run(["docker", "rm", "-f", "-v", cid], capture_output=True, check=False)
        if network_name:
            self._remove_network(network_name)
        self._remove_host_dirs(workspace_host_path=workspace_host_path, state_dir=state_dir)

    def _remove_network(self, network_name: str) -> None:
        r = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"network={network_name}"],
            capture_output=True,
            text=True,
        )
        for cid in (r.stdout or "").split():
            subprocess.run(["docker", "rm", "-f", "-v", cid], capture_output=True, check=False)
        subprocess.run(["docker", "network", "rm", network_name], capture_output=True, check=False)

    def _remove_host_dirs(self, *, workspace_host_path: Optional[str], state_dir: Optional[str]) -> None:
        keep = os.environ.get("OC_HOST_KEEP_WORKSPACE", "0") == "1"
        if keep:
            return
        if workspace_host_path and os.path.isdir(workspace_host_path):
            _force_rmtree(Path(workspace_host_path).parent, ignore_errors=True)
        if state_dir and os.path.isdir(state_dir):
            _force_rmtree(Path(state_dir), ignore_errors=True)

    def _best_effort_cleanup(self, net: str, oc: str, task: Optional[str] = None) -> None:
        for name in (oc, task):
            if name:
                subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True, check=False)
        self._remove_network(net)


class RequestBenchHermesHarness(RequestBenchHarness):
    """Hermes runtime container plus Request-Bench task sandbox."""

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        hm_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ):
        super().__init__(benchmark_data_root=benchmark_data_root, oc_image=DEFAULT_OPENCLAW_RUNTIME_IMAGE, workspace_base=workspace_base)
        self.hm_image = hm_image or DEFAULT_HM_DOCKER_IMAGE

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
            raise RuntimeError("RequestBenchHermesHarness.prepare_environment requires instance_data")
        if instance_data.get("benchmark") != "request_bench":
            raise RuntimeError(
                "RequestBenchHermesHarness only handles benchmark=request_bench "
                f"(got {instance_data.get('benchmark')!r})"
            )
        docker_image = container_image or instance_data.get("docker_image") or DEFAULT_SANDBOX_IMAGE
        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"hm-rb-net-{rid_short}"
        hm_name = f"hm-rb-{rid_short}"
        task_container_name = _request_task_name(run_id)

        _ensure_docker_image(self.hm_image, pull=False)
        _ensure_docker_image(docker_image)

        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        ws_host = run_root / "workspace"
        _write_prompt_workspace(ws_host, instance_data)
        state_dir = self._build_hermes_state_dir(run_id=run_id, sandbox_image=docker_image)
        dispatcher: Optional[HermesDispatcher] = None

        def rollback() -> None:
            if dispatcher is not None:
                try:
                    dispatcher.stop()
                except Exception:
                    pass
            self._best_effort_cleanup(net_name, hm_name, task_container_name)

        try:
            r = subprocess.run(
                ["docker", "network", "create", net_name],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0 and "already exists" not in (r.stderr or ""):
                raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")
            r = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", task_container_name,
                    "--network", net_name,
                    "--user", "0",
                    "--label", "hermes.sandbox=1",
                    "--label", f"oc-eval.run_id={run_id}",
                    "-v", f"{ws_host}:{_WORKSPACE}:rw",
                    "-w", _WORKSPACE,
                    "--entrypoint", "sleep",
                    docker_image,
                    "infinity",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(f"request sandbox start failed: {r.stderr.strip()}")
            task_container_id = r.stdout.strip()

            dispatcher = HermesDispatcher()
            dispatcher.start(state_dir=state_dir, backend="docker", target=task_container_name)
            model_cfg, _serving_cfg = load_active_model_and_serving()
            model_name = os.environ.get("HERMES_MODEL") or model_cfg.name
            provider = os.environ.get("HERMES_INFERENCE_PROVIDER") or model_cfg.provider
            base_url = endpoint_for_container(
                os.environ.get("OPENAI_BASE_URL") or model_cfg.endpoint, "vllm-host"
            )
            api_key = os.environ.get("OPENAI_API_KEY") or model_cfg.api_key
            bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
            docker_args: List[str] = []
            for mount in bridge_mounts:
                docker_args += ["-v", f"{mount.host}:{mount.container}:{mount.mode}"]
            hm_env = {
                "HERMES_HOME": _STATE_DIR_IN_HERMES,
                "HERMES_RUN_ID": run_id,
                "HERMES_INFERENCE_MODEL": model_name,
                "HERMES_INFERENCE_PROVIDER": provider,
                "HERMES_AGENT_BENCH_EVENTS": f"{_STATE_DIR_IN_HERMES}/events.jsonl",
                "OPENAI_API_KEY": api_key,
                "OPENAI_BASE_URL": base_url,
                "MINIMAX_API_KEY": api_key,
                "MINIMAX_BASE_URL": base_url,
                "TERMINAL_ENV": "docker",
                "TERMINAL_CWD": _WORKSPACE,
                "TERMINAL_DOCKER_IMAGE": docker_image,
                "OTEL_RESOURCE_ATTRIBUTES": f"agent.run_id={run_id}",
                **bridge_env,
            }
            env_args: List[str] = []
            for key, value in hm_env.items():
                env_args += ["-e", f"{key}={value}"]
            r = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", hm_name,
                    "--network", net_name,
                    "--add-host", "vllm-host:host-gateway",
                    "--user", "0",
                    "-v", f"{state_dir}:{_STATE_DIR_IN_HERMES}:rw",
                    "-v", f"{ws_host}:{_WORKSPACE}:rw",
                    *docker_args,
                    *env_args,
                    "-w", _WORKSPACE,
                    self.hm_image,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(f"hermes container start failed: {r.stderr.strip()}")
            hm_container_id = r.stdout.strip()
            return {
                "container_id": hm_container_id,
                "container_name": hm_name,
                "hm_container_id": hm_container_id,
                "task_container_id": task_container_id,
                "task_container_name": task_container_name,
                "network_name": net_name,
                "workspace_root": _WORKSPACE,
                "workspace_host_path": str(ws_host),
                "state_dir": str(state_dir),
                "agent_env": hm_env,
                "answer_file": instance_data.get("answer_file", "/workspace/answer.md"),
                **dispatcher.env_info(),
            }
        except Exception:
            rollback()
            raise

    def _build_hermes_state_dir(self, *, run_id: str, sandbox_image: str) -> Path:
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir, ignore_errors=True)
        state_dir.mkdir(parents=True)
        harness_cfg = {}
        harness_path = os.environ.get("AB_HARNESS_CONFIG")
        if harness_path and os.path.isfile(harness_path):
            harness_cfg = load_profile(harness_path)
        hm_cfg = harness_cfg.get("hermes") or {}
        model_cfg, _serving_cfg = load_active_model_and_serving()
        values = {
            "MODEL": os.environ.get("HERMES_MODEL") or model_cfg.name,
            "PROVIDER": os.environ.get("HERMES_INFERENCE_PROVIDER") or model_cfg.provider,
            "BASE_URL": endpoint_for_container(
                os.environ.get("OPENAI_BASE_URL") or model_cfg.endpoint, "vllm-host"
            ),
            "API_KEY": os.environ.get("OPENAI_API_KEY") or model_cfg.api_key,
            "TEMPERATURE": str(model_cfg.temperature),
            "MAX_TOKENS": str(model_cfg.max_tokens),
            "CONTEXT_LENGTH": str(model_cfg.context_window),
            "MAX_TURNS": str((hm_cfg.get("run") or {}).get("max_turns") or 20),
            "TERMINAL_TIMEOUT": str((hm_cfg.get("run") or {}).get("terminal_timeout") or 180),
            "SANDBOX_IMAGE": sandbox_image,
        }
        template = string.Template(
            "model:\n"
            "  default: \"$MODEL\"\n"
            "  provider: \"$PROVIDER\"\n"
            "  base_url: \"$BASE_URL\"\n"
            "  api_key: \"$API_KEY\"\n"
            "  api_mode: \"chat_completions\"\n"
            "  temperature: $TEMPERATURE\n"
            "  max_tokens: $MAX_TOKENS\n"
            "  context_length: $CONTEXT_LENGTH\n"
            "terminal:\n"
            "  backend: \"docker\"\n"
            "  cwd: \"/workspace\"\n"
            "  timeout: $TERMINAL_TIMEOUT\n"
            "  docker_image: \"$SANDBOX_IMAGE\"\n"
            "agent:\n"
            "  max_turns: $MAX_TURNS\n"
            "display:\n"
            "  streaming: false\n"
            "  compact: true\n"
            "platform_toolsets:\n"
            "  cli:\n"
            "    - terminal\n"
            "    - file\n"
            "    - code_execution\n"
        )
        (state_dir / "config.yaml").write_text(
            template.safe_substitute(values), encoding="utf-8"
        )
        return state_dir

    def exec_in_container(
        self,
        container_id: str,
        argv: List[str],
        *,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        cmd = ["docker", "exec"]
        if workdir:
            cmd += ["-w", workdir]
        for key, value in (env or {}).items():
            cmd += ["-e", f"{key}={value}"]
        cmd += [container_id, *argv]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
            )
        return proc

    def cleanup(
        self,
        container_id: Optional[str] = None,
        hm_container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
        container_name: Optional[str] = None,
        network_name: Optional[str] = None,
        workspace_host_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        bridge_pid: Optional[int] = None,
        bridge_socket: Optional[str] = None,
        **_extra,
    ) -> None:
        HermesDispatcher.stop_external(pid=bridge_pid, socket_path=bridge_socket)
        uid, gid = os.getuid(), os.getgid()
        for cid in {hm_container_id, container_id}:
            if cid:
                subprocess.run(
                    ["docker", "exec", cid, "chown", "-R", f"{uid}:{gid}", _STATE_DIR_IN_HERMES],
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
        targets = [
            t for t in (hm_container_id, container_id, task_container_id, task_container_name, container_name) if t
        ]
        for cid in dict.fromkeys(targets):
            subprocess.run(["docker", "rm", "-f", "-v", cid], capture_output=True, check=False)
        if network_name:
            self._remove_network(network_name)
        self._remove_host_dirs(workspace_host_path=workspace_host_path, state_dir=state_dir)
