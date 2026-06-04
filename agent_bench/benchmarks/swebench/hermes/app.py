"""SWE-bench application — Hermes Agent harness variant.

Layered model:

    host  ──── only vLLM runs here ────  http://localhost:8000/v1
                                   ▲
                                   │ OpenAI-compatible HTTP (vllm-host:8000)
                                   │
    ┌──────────────────────────────────────┐
    │ hm-<rid>  agent-bench/hermes:<v>      │
    │ - upstream Hermes Agent CLI           │
    │ - patched docker terminal env         │
    │ - hm-exec-bridge (mounted)            │
    └────────┬─────────────────────────────┘
             │ every tool/file/code exec via bridge
             │ (unix socket -> host server -> docker exec)
             ↓
    ┌──────────────────────────────────────┐
    │ oc-sbx-<rid>  swebench instance      │
    │ - per-instance python env + pytest   │
    │ - workspace bind-mounted at /testbed │
    └──────────────────────────────────────┘

Hermes stays on its normal docker terminal workflow. The baked patch checks
for HERMES_EXEC_BRIDGE and swaps docker exec for Agent_Bench's external
bridge, matching OpenClaw's patched-runtime approach.
"""
from __future__ import annotations

import json
import logging
import os
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from agent_bench.benchmarks.swebench.app import _force_rmtree, _oc_sandbox_name
from agent_bench.benchmarks.swebench.scoring import score_via_scoring_server, score_via_singularity
from agent_bench.deployments.containers import (
    Container,
    ContainerHandle,
    ContainerSpec,
    DockerContainer,
    Mount,
)
from agent_bench.benchmarks.conditions.agent_otel import otel_config_values, otel_env, otel_endpoint_for_container
from agent_bench.harnesses.hermes.dispatcher import HermesDispatcher
from agent_bench.core.config import endpoint_for_container, load_active_model_and_serving
from agent_bench.core.profile_io import load_profile

logger = logging.getLogger(__name__)

DEFAULT_HERMES_RUNTIME_IMAGE = "agent-bench/hermes:v2026.5.16"

_STATE_DIR_IN_HERMES = "/opt/data"
_WORKSPACE_IN_HERMES = "/workspace"
_TASK_WORKSPACE = "/testbed"


class SWEBenchHermesHarness:
    """Per-run harness for Hermes Agent x SWE-bench."""

    backend_id = "container-agnostic"

    _HERMES_SCAFFOLD_FILES = (
        "AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md "
        "BOOT.md MEMORY.md fix.patch"
    )
    _HERMES_SCAFFOLD_DIRS = ".hermes memory sessions .pytest_cache"

    def __init__(
        self,
        benchmark_data_root: Path,
        hm_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
        container: Optional[Container] = None,
    ):
        self.benchmark_data_root = Path(benchmark_data_root)
        self.container = container or DockerContainer()
        self.hm_image = hm_image or self._default_hm_image()

        try:
            import swebench  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "swebench python package not importable; "
                "install in the driver env: pip install swebench"
            ) from e

        self.container.check_available()
        self.container.check_image(self.hm_image)

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

    def _default_hm_image(self) -> str:
        if self.container.backend_id == "singularity":
            scratch = os.environ.get("SCRATCH") or os.path.expanduser("~")
            return (
                os.environ.get("AB_AGENT_RUNTIME_SIF")
                or os.environ.get("AB_AGENT_RUNTIME_IMAGE")
                or str(Path(scratch) / "sif_images" / "hermes_runtime.sif")
            )
        return os.environ.get("AB_AGENT_RUNTIME_IMAGE", DEFAULT_HERMES_RUNTIME_IMAGE)

    def _preflight_cleanup(self, net: str, hm: str, task: Optional[str]) -> None:
        logger.info(
            "preflight cleanup for Hermes run resources: hm=%s task=%s net=%s",
            hm, task, net,
        )
        self._best_effort_cleanup(net, hm, task)

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
        if "UNUSED" in container_image:
            container_image = (
                f"swebench/sweb.eval.x86_64.{instance_id.lower()}:latest"
                .replace("__", "_1776_")
            )
        task_image = self._resolve_task_image(container_image)

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"hm-eval-net-{rid_short}"
        hm_name = f"hm-{rid_short}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        scope_key = "agent:swebench"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        self._preflight_cleanup(net_name, hm_name, task_container_name)

        logger.info(
            "Preparing run %s [runtime=%s] (hm=%s, sandbox=%s, repo=%s@%s, task=%s)",
            run_id, self.container.backend_id, self.hm_image, container_image,
            repo_id, base_commit[:12], task_container_name,
        )

        ws_host = self.workspace_base / run_id
        if ws_host.exists():
            _force_rmtree(ws_host, ignore_errors=True)
        ws_host.mkdir(parents=True)
        self._prime_workspace(task_image, ws_host)

        state_dir = self._build_state_dir(
            run_id=run_id,
            sandbox_image=container_image,
        )

        self.container.create_network(net_name)

        task_path = (
            "/opt/miniconda3/envs/testbed/bin:"
            "/opt/miniconda3/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        task_spec = ContainerSpec(
            name=task_container_name,
            image=task_image,
            network=net_name,
            user="0",
            labels={
                "hermes.sandbox": "1",
                "oc-eval.run_id": run_id,
            },
            mounts=(Mount(host=ws_host, container=_TASK_WORKSPACE, mode="rw"),),
            workdir=_TASK_WORKSPACE,
            env={"PATH": task_path, "HOME": "/root"},
            entrypoint="sleep",
            command=("infinity",),
        )
        try:
            task_handle = self.container.start(task_spec)
        except BaseException:
            self._best_effort_cleanup(net_name, hm_name, task_container_name)
            raise
        task_container_id = task_handle.id

        dispatcher = HermesDispatcher()
        try:
            dispatcher.start(
                state_dir=state_dir,
                backend=self.container.backend_id,
                target=task_container_name,
            )
        except BaseException:
            self._best_effort_cleanup(net_name, hm_name, task_container_name)
            raise
        logger.info("  exec-bridge: pid=%s sock=%s", dispatcher.pid, dispatcher.socket_path)

        model_cfg, _serving_cfg = load_active_model_and_serving()
        model_name = os.environ.get("AB_MODEL_NAME") or model_cfg.name
        provider = os.environ.get("HERMES_INFERENCE_PROVIDER") or model_cfg.provider
        base_url = self._runtime_base_url(model_cfg.endpoint)
        api_key = os.environ.get("OPENAI_API_KEY") or model_cfg.api_key

        hm_mounts: list[Mount] = [
            Mount(host=state_dir, container=_STATE_DIR_IN_HERMES, mode="rw"),
            Mount(host=ws_host, container=_WORKSPACE_IN_HERMES, mode="rw"),
        ]
        b_mounts, b_env = dispatcher.agent_mounts_and_env()
        hm_mounts.extend(b_mounts)
        hm_env = {
            "HERMES_HOME": _STATE_DIR_IN_HERMES,
            "AB_RUN_ID": run_id,
            "HERMES_RUN_ID": run_id,
            "HERMES_INFERENCE_MODEL": model_name,
            "HERMES_INFERENCE_PROVIDER": provider,
            "AB_AGENT_RUNTIME_EVENTS_FILE": f"{_STATE_DIR_IN_HERMES}/events.jsonl",
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "MINIMAX_API_KEY": api_key,
            "MINIMAX_BASE_URL": base_url,
            # Hermes still thinks its terminal backend is docker; our patch
            # intercepts the docker exec path and routes through the bridge.
            "TERMINAL_ENV": "docker",
            "TERMINAL_CWD": _TASK_WORKSPACE,
            "TERMINAL_DOCKER_IMAGE": container_image,
            **otel_env(
                service_name="agent-bench-hermes",
                run_id=run_id,
                endpoint=otel_endpoint_for_container(self.container.host_address),
                extra_attrs={"hm.run_id": run_id},
            ),
            **b_env,
        }
        hm_spec = ContainerSpec(
            name=hm_name,
            image=self.hm_image,
            network=net_name,
            extra_hosts={"vllm-host": "host-gateway"},
            user="0",
            mounts=tuple(hm_mounts),
            workdir=_WORKSPACE_IN_HERMES,
            env=hm_env,
        )
        try:
            hm_handle = self.container.start(hm_spec)
        except BaseException:
            dispatcher.stop()
            self._best_effort_cleanup(net_name, hm_name, task_container_name)
            raise
        hm_container_id = hm_handle.id
        self._workspace_by_container[hm_container_id] = ws_host

        logger.info(
            "  ready: hm=%s task=%s bridge_pid=%s net=%s",
            hm_container_id[:12], task_container_id[:12],
            dispatcher.pid, net_name,
        )
        return {
            "container_id": hm_container_id,
            "container_name": hm_name,
            "backend": self.container.backend_id,
            "harness": "hermes",
            "hm_container_id": hm_container_id,
            "task_container_id": task_container_id,
            "task_container_name": task_container_name,
            "network_name": net_name,
            "workspace_root": _WORKSPACE_IN_HERMES,
            "task_workspace_root": _TASK_WORKSPACE,
            "workspace_host_path": str(ws_host),
            "state_dir": str(state_dir),
            "config_path_in_container": f"{_STATE_DIR_IN_HERMES}/config.yaml",
            "sandbox_image": container_image,
            "agent_env": hm_env,
            **dispatcher.env_info(),
        }

    def _load_harness_config(self) -> Dict:
        repo_root = os.environ.get("AB_REPO_ROOT", os.getcwd())
        harness_path = os.environ.get(
            "AB_HARNESS_CONFIG",
            os.path.join(repo_root, "profiles/harnesses/hermes_swebench.yaml"),
        )
        if not os.path.isfile(harness_path):
            raise FileNotFoundError(
                f"hermes harness config not found at {harness_path} "
                "(set AB_HARNESS_CONFIG)."
            )
        return load_profile(harness_path)

    def _build_state_dir(self, *, run_id: str, sandbox_image: str) -> Path:
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir)
        state_dir.mkdir(parents=True)

        repo_root = os.environ.get("AB_REPO_ROOT", os.getcwd())
        harness_cfg = self._load_harness_config()
        hm_cfg = harness_cfg.get("hermes") or {}
        model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)

        template_path = (
            Path(repo_root) / "agent_bench" / "harnesses" / "hermes" / "state" / "config.yaml.template"
        )
        if not template_path.is_file():
            raise FileNotFoundError(f"hermes config.yaml.template missing at {template_path}")

        values = {
            "AB_MODEL_NAME": os.environ.get("AB_MODEL_NAME") or model_cfg.name,
            "PROVIDER": os.environ.get("HERMES_INFERENCE_PROVIDER") or model_cfg.provider,
            "BASE_URL": self._runtime_base_url(model_cfg.endpoint),
            "API_KEY": os.environ.get("OPENAI_API_KEY") or model_cfg.api_key,
            "TEMPERATURE": str(model_cfg.temperature),
            "AB_MODEL_MAX_OUTPUT_TOKENS": str(model_cfg.max_tokens),
            "CONTEXT_LENGTH": str(model_cfg.context_window),
            "MAX_TURNS": str((hm_cfg.get("run") or {}).get("max_turns") or 90),
            "TERMINAL_TIMEOUT": str((hm_cfg.get("run") or {}).get("terminal_timeout") or 180),
            "TERMINAL_CWD": _TASK_WORKSPACE,
            "SANDBOX_IMAGE": sandbox_image,
            **otel_config_values(
                service_name="agent-bench-hermes",
                run_id=run_id,
                endpoint=otel_endpoint_for_container(self.container.host_address),
                extra_attrs={"hm.run_id": run_id},
                enabled=((harness_cfg.get("telemetry") or {}).get("otel") or {}).get("enabled", True) is not False,
            ),
        }
        rendered = string.Template(template_path.read_text()).safe_substitute(values)
        (state_dir / "config.yaml").write_text(rendered)
        return state_dir

    def _resolve_task_image(self, container_image: str) -> str:
        if self.container.backend_id == "docker":
            return container_image
        sif_dir = os.environ.get("AB_SCORING_IMAGE_DIR")
        if not sif_dir:
            raise RuntimeError(
                "AB_SCORING_IMAGE_DIR not set; the singularity backend needs "
                "the per-instance .sif directory."
            )
        tag = container_image.split(":")[0]
        slug = tag.split(".")[-1]
        instance_id = slug.replace("_1776_", "__")
        return str(Path(sif_dir) / f"{instance_id}.sif")

    def _prime_workspace(self, task_image: str, ws_host: Path) -> None:
        self.container.copy_from_image(task_image, _TASK_WORKSPACE, ws_host)

    def _runtime_base_url(self, configured: Optional[str]) -> str:
        raw = os.environ.get("OPENAI_BASE_URL") or configured or "http://localhost:8000/v1"
        return endpoint_for_container(raw, self.container.host_address)

    def _exec(
        self,
        container_id: str,
        argv: list[str],
        *,
        check: bool = True,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        h = ContainerHandle(
            id=container_id,
            name=container_id,
            backend=self.container.backend_id,
        )
        return self.container.exec(
            h, argv, check=check, timeout=timeout, env=env, workdir=workdir,
        )

    def exec_in_container(
        self,
        container_id: str,
        argv: list[str],
        *,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        return self._exec(
            container_id, argv, check=check, timeout=timeout, env=env, workdir=workdir,
        )

    def collect_patch(self, container_id: str, workspace_root: str) -> str:
        ws_host = self._workspace_by_container.get(container_id)
        if self.container.backend_id == "singularity" and ws_host:
            try:
                for name in self._HERMES_SCAFFOLD_FILES.split():
                    path = ws_host / name
                    if path.is_dir():
                        _force_rmtree(path, ignore_errors=True)
                    elif path.exists():
                        path.unlink()
                for name in self._HERMES_SCAFFOLD_DIRS.split():
                    _force_rmtree(ws_host / name, ignore_errors=True)
                r = subprocess.run(
                    ["git", "-C", str(ws_host), "diff", "HEAD"],
                    capture_output=True, text=True, check=True,
                )
                return r.stdout
            except Exception as e:
                logger.warning("host-side patch collection failed: %s", e)

        ws = workspace_root
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {self._HERMES_SCAFFOLD_FILES} && "
             f"rm -rf {self._HERMES_SCAFFOLD_DIRS}"],
            check=False,
        )
        self._exec(
            container_id,
            ["git", "config", "--global", "--add", "safe.directory", ws],
            check=False,
        )
        r = self._exec(container_id, ["git", "-C", ws, "diff", "HEAD"])
        return r.stdout

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
        **_kw,
    ) -> Dict:
        logger.info("Scoring %s", instance_id)
        if patch is None:
            if not container_id:
                return {
                    "status": "ERROR", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": "run_tests called with patch=None and no container_id",
                    "evaluation_dir": None,
                }
            patch = self.collect_patch(container_id, workspace_root)

        if not patch.strip():
            return {
                "status": "FAIL", "test_pass_rate": 0.0,
                "passed": 0, "failed": 1,
                "error_message": "Empty patch (agent made no changes)",
                "evaluation_dir": None,
            }

        if run_id is None:
            run_id = f"hm-eval-{instance_id}-{os.urandom(3).hex()}"

        singularity_verdict = score_via_singularity(
            instance_id=instance_id,
            patch=patch,
            model_name_or_path="hm-eval",
            timeout_seconds=timeout_seconds,
            run_id=run_id,
            benchmark_data_root=self.benchmark_data_root,
            instance_data=instance_data,
        )
        if singularity_verdict is not None:
            return singularity_verdict

        remote_verdict = score_via_scoring_server(
            instance_id=instance_id,
            patch=patch,
            model_name_or_path="hm-eval",
            timeout_seconds=timeout_seconds,
        )
        if remote_verdict is not None:
            return remote_verdict

        with tempfile.TemporaryDirectory(prefix="hm_pred_") as tmp:
            tmp_path = Path(tmp)
            pred_path = tmp_path / "predictions.json"
            pred_path.write_text(json.dumps([{
                "instance_id": instance_id,
                "model_name_or_path": "hm-eval",
                "model_patch": patch,
            }]))
            cmd = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "--dataset_name", "princeton-nlp/SWE-bench_Verified",
                "--predictions_path", str(pred_path),
                "--instance_ids", instance_id,
                "--max_workers", "1",
                "--run_id", run_id,
                "--cache_level", "env",
                "--timeout", str(timeout_seconds),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.benchmark_data_root),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds + 300,
                )
            except subprocess.TimeoutExpired:
                return {
                    "status": "TIMEOUT", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": "run_evaluation wall-clock timeout",
                    "evaluation_dir": None,
                }
            verdict = self._find_and_parse_verdict(run_id, instance_id)
            if verdict.get("status") in ("PASS", "FAIL"):
                if proc.returncode != 0:
                    logger.warning(
                        "run_evaluation exit %d but report.json present (status=%s)",
                        proc.returncode, verdict.get("status"),
                    )
                return verdict
            if proc.returncode != 0:
                return {
                    "status": "ERROR", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": f"run_evaluation exit {proc.returncode}: {proc.stderr[-500:]}",
                    "evaluation_dir": None,
                }
            return verdict

    def _find_and_parse_verdict(self, run_id: str, instance_id: str) -> Dict:
        report_path = (
            self.benchmark_data_root
            / "logs" / "run_evaluation" / run_id / "hm-eval" / instance_id
            / "report.json"
        )
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except Exception as e:
                return {
                    "status": "ERROR", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": f"failed to parse report.json: {e}",
                    "evaluation_dir": str(report_path.parent),
                }
            inst = report.get(instance_id, report)
            resolved = inst.get("resolved", False)
            ts = inst.get("tests_status", {})
            f2p = ts.get("FAIL_TO_PASS", {"success": [], "failure": []})
            p2p = ts.get("PASS_TO_PASS", {"success": [], "failure": []})
            passed = len(f2p.get("success", [])) + len(p2p.get("success", []))
            failed = len(f2p.get("failure", [])) + len(p2p.get("failure", []))
            total = passed + failed
            return {
                "status": "PASS" if resolved else "FAIL",
                "test_pass_rate": passed / total if total else 0.0,
                "passed": passed,
                "failed": failed,
                "evaluation_dir": str(report_path.parent),
                "raw_report_path": str(report_path),
            }
        return {
            "status": "ERROR", "test_pass_rate": 0.0,
            "passed": 0, "failed": 0,
            "error_message": f"report.json not found (run_id={run_id})",
            "evaluation_dir": None,
        }

    def cleanup(
        self,
        container_id: Optional[str] = None,
        hm_container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
        network_name: Optional[str] = None,
        workspace_host_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        bridge_pid: Optional[int] = None,
        bridge_socket: Optional[str] = None,
        **_extra,
    ):
        HermesDispatcher.stop_external(pid=bridge_pid, socket_path=bridge_socket)

        uid, gid = os.getuid(), os.getgid()
        for cid in {hm_container_id, container_id}:
            if not cid:
                continue
            self._workspace_by_container.pop(cid, None)
            try:
                self._exec(
                    cid,
                    ["chown", "-R", f"{uid}:{gid}", _STATE_DIR_IN_HERMES],
                    check=False,
                    timeout=30,
                )
            except Exception:
                pass

        targets = [t for t in (
            hm_container_id, container_id,
            task_container_id, task_container_name,
        ) if t]
        for cid in dict.fromkeys(targets):
            h = ContainerHandle(
                id=cid,
                name=cid,
                backend=self.container.backend_id,
            )
            try:
                self.container.stop(h)
            except Exception:
                pass

        if network_name:
            try:
                self.container.remove_network(network_name)
            except Exception:
                pass

        keep = os.environ.get("AB_HARNESS_KEEP_WORKSPACE", "0") == "1"
        if not keep and workspace_host_path and os.path.isdir(workspace_host_path):
            _force_rmtree(Path(workspace_host_path), ignore_errors=True)
        if not keep and state_dir and os.path.isdir(state_dir):
            _force_rmtree(Path(state_dir), ignore_errors=True)

    def _best_effort_cleanup(
        self, net: str, hm: str, task: Optional[str] = None,
    ) -> None:
        for name in (hm, task):
            if name:
                h = ContainerHandle(
                    id=name,
                    name=name,
                    backend=self.container.backend_id,
                )
                try:
                    self.container.stop(h)
                except Exception:
                    pass
        try:
            self.container.remove_network(net)
        except Exception:
            pass
