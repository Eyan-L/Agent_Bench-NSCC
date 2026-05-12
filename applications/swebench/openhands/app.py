"""SWE-bench application — OpenHands harness variant.

Sibling of `applications/swebench/app.py`. Same lifecycle contract (the
`SWEBenchOpenHandsHarness` class is interchangeable with `SWEBenchHarness`
from the driver's POV), different agent runtime: instead of an OpenClaw
gateway, this brings up the OpenHands SDK harness container and uses the
shared exec-bridge to route the agent's tool calls into the SWE-bench
sandbox container.

Layered model:

    host  ──── only vLLM runs here ────  http://localhost:8000/v1
                                   ▲
                                   │ OpenAI-compat HTTP (vllm-host:8000)
                                   │
    ┌──────────────────────────────────────┐
    │ oh-<rid>  agent-bench/openhands:<v>  │
    │ - openhands-sdk + openhands-tools    │
    │ - /opt/openhands_driver.py           │ ← invoked by the condition
    │ - oh-exec-bridge (mounted)           │ ← from dispatcher
    └────────┬─────────────────────────────┘
             │  every tool call via the bridge
             │  (unix socket → host server → docker exec)
             ↓
    ┌──────────────────────────────────────┐
    │ oc-sbx-<rid>  swebench instance      │
    │ - per-instance python env + pytest   │
    │ - workspace bind-mounted at /testbed │
    │ - tool exec from the agent lands here│
    └──────────────────────────────────────┘

Reuses the OC sandbox container name format so trace consumers can pivot on
`oc-sbx-*` regardless of which harness was active.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Shared helpers from the OpenClaw-flavoured SWE-bench app — these are
# harness-agnostic in spirit even though they live next to the openclaw
# class. Importing them lets us avoid duplicating ~80 lines.
from applications.swebench.app import (
    _force_rmtree,
    _oc_sandbox_name,
    _slugify_session_key,
)
from orchestration.dispatcher.openhands import OpenHandsDispatcher

logger = logging.getLogger(__name__)


DEFAULT_OH_IMAGE = os.environ.get(
    "OH_IMAGE", "agent-bench/openhands:1.21.0",
)

# Canonical in-container paths (must match orchestration/dispatcher/openhands.py
# and harnesses/openhands/docker/Dockerfile).
_STATE_DIR_IN_OH = "/etc/agent-bench/state"
_WORKSPACE_IN_OH = "/workspace"


class SWEBenchOpenHandsHarness:
    """Per-run harness for OpenHands × SWE-bench (docker backend).

    Constructor + prepare_environment + run_tests + cleanup mirror the
    `SWEBenchHarness` interface. The driver dispatches between the two based
    on the harness config's `"harness"` field.
    """

    backend_id = "docker"

    _OH_SCAFFOLD_FILES = ".openhands_instructions .openhands.json"
    _OH_SCAFFOLD_DIRS = ".openhands"

    def __init__(
        self,
        swebench_root: Path,
        oh_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ):
        self.swebench_root = Path(swebench_root)
        self.oh_image = oh_image or DEFAULT_OH_IMAGE

        try:
            import swebench  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "swebench python package not importable; "
                "install in the driver env: pip install swebench"
            ) from e

        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker not available: {r.stderr.strip()}")
        r = subprocess.run(
            ["docker", "image", "inspect", self.oh_image],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"openhands harness image not found: {self.oh_image}\n"
                "Build it first: bash harnesses/openhands/docker/build.sh"
            )

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"oh-eval-net-{rid_short}"
        oh_name = f"oh-{rid_short}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        scope_key = "agent:swebench"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        logger.info(
            "Preparing run %s (oh=%s, sandbox=%s, repo=%s@%s, task=%s)",
            run_id, self.oh_image, container_image, repo_id, base_commit[:12],
            task_container_name,
        )

        # 1. Host workspace, primed from the task image's /testbed.
        ws_host = self.workspace_base / run_id
        if ws_host.exists():
            _force_rmtree(ws_host, ignore_errors=True)
        ws_host.mkdir(parents=True)
        self._prime_workspace(container_image, ws_host)

        # 2. Per-run state dir with config.toml rendered from harness JSON.
        state_dir = self._build_state_dir(run_id=run_id)

        # 3. Docker network so OH harness can reach vLLM at vllm-host:8000.
        r = subprocess.run(
            ["docker", "network", "create", "--driver", "bridge", net_name],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0 and "already exists" not in r.stderr:
            raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

        # 4. Task (sandbox) container. The agent's tool calls land here via
        #    the bridge. SWE-bench images put the test env under
        #    /opt/miniconda3/envs/testbed — same PATH trick as OC variant.
        task_path = (
            "/opt/miniconda3/envs/testbed/bin:"
            "/opt/miniconda3/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        task_cmd = [
            "docker", "run", "-d",
            "--name", task_container_name,
            "--network", net_name,
            "--user", "0",
            "--label", "openhands.sandbox=1",
            "--label", f"oc-eval.run_id={run_id}",
            "-v", f"{ws_host}:/testbed:rw",
            "-w", "/testbed",
            "-e", f"PATH={task_path}",
            "--entrypoint", "sleep",
            container_image,
            "infinity",
        ]
        r = subprocess.run(task_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            self._best_effort_cleanup(net_name, oh_name, task_container_name)
            raise RuntimeError(f"task container start failed: {r.stderr.strip()}")
        task_container_id = r.stdout.strip()

        # 5. Exec-bridge dispatcher (host process). Server listens on a unix
        #    socket inside state_dir; the OH driver inside the harness
        #    container shells out to oh-exec-bridge → socket → docker exec
        #    into the sandbox we just started.
        dispatcher = OpenHandsDispatcher()
        try:
            dispatcher.start(
                state_dir=state_dir,
                backend="docker",
                target=task_container_name,
            )
        except Exception:
            self._best_effort_cleanup(net_name, oh_name, task_container_name)
            raise
        logger.info(
            "  exec-bridge: pid=%s sock=%s",
            dispatcher.pid, dispatcher.socket_path,
        )

        # 6. OpenHands harness container. `sleep infinity` + condition does
        #    `docker exec oh-<rid> python /opt/openhands_driver.py ...`.
        oh_cmd = [
            "docker", "run", "-d",
            "--name", oh_name,
            "--network", net_name,
            "--add-host", "vllm-host:host-gateway",
            "--user", "0",
            # state_dir holds the rendered config.toml (read-only-ish) *and*
            # events.jsonl which the OH driver appends to from inside the
            # container — so the mount must be rw.
            "-v", f"{state_dir}:{_STATE_DIR_IN_OH}:rw",
            "-v", f"{ws_host}:{_WORKSPACE_IN_OH}:rw",
            "-w", _WORKSPACE_IN_OH,
            "-e", f"OPENHANDS_RUN_ID={run_id}",
            "-e", f"OTEL_RESOURCE_ATTRIBUTES=oh.run_id={run_id}",
            # Sandbox routing — OH driver reads these to wrap tool executors.
            *dispatcher.agent_container_args(),
            self.oh_image,
            # No special entrypoint — image's default `tini sleep infinity`
            # keeps the container alive for `docker exec`.
        ]
        r = subprocess.run(oh_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oh_name, task_container_name)
            raise RuntimeError(f"oh container start failed: {r.stderr.strip()}")
        oh_container_id = r.stdout.strip()

        logger.info(
            "  ready: oh=%s task=%s bridge_pid=%s net=%s",
            oh_container_id[:12], task_container_id[:12],
            dispatcher.pid, net_name,
        )
        return {
            "container_id": oh_container_id,
            "container_name": oh_name,
            "backend": "docker",
            "harness": "openhands",
            "oh_container_id": oh_container_id,
            "task_container_id": task_container_id,
            "task_container_name": task_container_name,
            "network_name": net_name,
            "workspace_root": _WORKSPACE_IN_OH,
            "workspace_host_path": str(ws_host),
            "state_dir": str(state_dir),
            "config_path_in_container": f"{_STATE_DIR_IN_OH}/config.toml",
            "events_path_in_container": f"{_STATE_DIR_IN_OH}/events.jsonl",
            "events_path_on_host": str(state_dir / "events.jsonl"),
            "sandbox_image": container_image,
            **dispatcher.env_info(),
        }

    # ------------------------------------------------------------------

    def _prime_workspace(self, container_image: str, ws_host: Path) -> None:
        """Same recipe as SWEBenchHarness._prime_workspace_from_image, but
        local to avoid the cross-class private-method dependency."""
        scratch_name = f"oh-eval-prime-{ws_host.name[:12]}-{os.urandom(2).hex()}"
        r = subprocess.run(
            ["docker", "create", "--name", scratch_name,
             "--entrypoint", "true", container_image],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker create scratch failed: {r.stderr.strip()}")
        try:
            r = subprocess.run(
                ["docker", "cp", f"{scratch_name}:/testbed/.", str(ws_host) + "/"],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"docker cp from {container_image}:/testbed failed: "
                    f"{r.stderr.strip()}"
                )
        finally:
            subprocess.run(
                ["docker", "rm", "-f", scratch_name],
                capture_output=True, check=False,
            )

    # ------------------------------------------------------------------

    def _build_state_dir(self, *, run_id: str) -> Path:
        """Materialize <state_base>/<run_id>/ with the rendered config.toml.

        The harness config (`config/harness/openhands_*.json`, picked by the
        driver) carries an `openhands` block with model + agent + run
        knobs. We substitute those into harnesses/openhands/state/config.toml.template
        and drop the result at <state_dir>/config.toml. The OH driver reads
        it via `--config /etc/agent-bench/state/config.toml`.
        """
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir)
        state_dir.mkdir(parents=True)

        kit_root = os.environ.get("KIT_ROOT", os.getcwd())
        # Driver sets OC_HARNESS_YAML to the resolved harness config path
        # (kept name for back-compat with the openclaw side); same lookup.
        harness_path = os.environ.get(
            "OC_HARNESS_YAML",
            os.path.join(kit_root, "config/harness/openhands_27b.json"),
        )
        if not os.path.isfile(harness_path):
            raise FileNotFoundError(
                f"openhands harness config not found at {harness_path} "
                "(set OC_HARNESS_YAML)."
            )
        with open(harness_path) as fh:
            harness_cfg = json.load(fh) or {}

        oh_cfg = harness_cfg.get("openhands") or {}
        model_cfg = harness_cfg.get("model") or {}

        template_path = (
            Path(kit_root) / "harnesses" / "openhands" / "state" / "config.toml.template"
        )
        if not template_path.is_file():
            raise FileNotFoundError(
                f"openhands config.toml.template missing at {template_path}"
            )

        # OpenHands -> LiteLLM needs a provider prefix to route. vLLM speaks
        # OpenAI-compat, so wrap any bare/path-style model id with `openai/`.
        # Path-style (e.g. `/mnt/.../Qwen2.5-7B-Instruct`) is what we use when
        # `--model <local-dir>` is fed to vLLM; LiteLLM forwards that string
        # verbatim to the endpoint, so `openai/<path>` ends up correct.
        _model_raw = model_cfg.get("name") or "Qwen/Qwen3.6-35B-A3B"
        _LITELLM_PROVIDERS = (
            "openai/", "anthropic/", "vllm/", "hosted_vllm/", "azure/",
            "bedrock/", "gemini/", "groq/", "together_ai/", "huggingface/",
            "mistral/", "cohere/", "ollama/",
        )
        _model_resolved = (
            _model_raw if _model_raw.startswith(_LITELLM_PROVIDERS)
            else f"openai/{_model_raw}"
        )
        values = {
            "MODEL": _model_resolved,
            "BASE_URL": model_cfg.get("endpoint") or "http://vllm-host:8000/v1",
            "API_KEY": model_cfg.get("api_key") or "EMPTY",
            "TEMPERATURE": str(model_cfg.get("temperature") or 0.0),
            "MAX_TOKENS": str(model_cfg.get("max_tokens") or 4096),
            "MAX_ITERATIONS": str(
                (oh_cfg.get("run") or {}).get("max_iterations") or 100
            ),
            "AGENT_NAME": (oh_cfg.get("agent") or {}).get("name") or "CodeActAgent",
            "RUNTIME": (oh_cfg.get("run") or {}).get("runtime") or "process",
            "WORKSPACE": _WORKSPACE_IN_OH,
            "EVENTS_PATH": f"{_STATE_DIR_IN_OH}/events.jsonl",
        }
        rendered = string.Template(template_path.read_text()).safe_substitute(values)
        (state_dir / "config.toml").write_text(rendered)
        # Empty events file so the driver can append safely.
        (state_dir / "events.jsonl").touch()
        return state_dir

    # ------------------------------------------------------------------

    def _exec(
        self,
        container_id: str,
        argv: List[str],
        *,
        check: bool = True,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        r = subprocess.run(
            ["docker", "exec", container_id, *argv],
            capture_output=True, text=True, timeout=timeout,
        )
        if check and r.returncode != 0:
            raise RuntimeError(
                f"docker exec {argv[0]} failed (rc={r.returncode}): "
                f"{r.stderr.strip()}"
            )
        return r

    # ------------------------------------------------------------------
    # Patch collection — the agent's edits live in ws_host (bind-mounted at
    # /testbed in the sandbox and /workspace in the OH container).
    # ------------------------------------------------------------------

    def collect_patch(self, container_id: str, workspace_root: str) -> str:
        # Strip OH scaffold + diff against base_commit. We do this via the
        # task container (where /testbed has the right git env) — not the
        # OH harness container — because the OH image doesn't carry the
        # repo's python env. workspace_root here is the OH container's path
        # but the actual bind is the SAME host dir as the task's /testbed.
        # We'll use whichever container_id the harness handed us; the
        # driver passes the OH cid by convention, so route through there.
        # `git` is installed in the OH harness image.
        ws = workspace_root
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {self._OH_SCAFFOLD_FILES} && "
             f"rm -rf {self._OH_SCAFFOLD_DIRS}"],
            check=False,
        )
        # Git refuses to operate on a repo whose dir is owned by another uid
        # ("dubious ownership"). The /workspace bind is on the host file
        # system so ownership inside the container often differs from the
        # uid git is running as; tell git to trust this specific path.
        self._exec(
            container_id,
            ["git", "config", "--global", "--add", "safe.directory", ws],
            check=False,
        )
        self._exec(container_id, ["git", "-C", ws, "add", "-A"])
        r = self._exec(container_id, ["git", "-C", ws, "diff", "--cached"])
        return r.stdout

    # ------------------------------------------------------------------
    # Scoring — delegated to SWE-bench's run_evaluation. Identical to
    # SWEBenchHarness.run_tests; we duplicate here so the two harnesses
    # don't share a base class (composition would force a refactor of the
    # OpenClaw variant we don't want to touch in this commit).
    # ------------------------------------------------------------------

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
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
            run_id = f"oh-eval-{instance_id}-{os.urandom(3).hex()}"

        import tempfile  # local import to keep top-level slim
        with tempfile.TemporaryDirectory(prefix="oh_pred_") as tmp:
            tmp = Path(tmp)
            pred_path = tmp / "predictions.json"
            pred_path.write_text(json.dumps([{
                "instance_id": instance_id,
                "model_name_or_path": "oh-eval",
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
                    cmd, cwd=str(self.swebench_root),
                    capture_output=True, text=True,
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
        primary = (
            self.swebench_root
            / "logs" / "run_evaluation" / run_id / "oh-eval" / instance_id
            / "report.json"
        )
        if primary.exists():
            try:
                report = json.loads(primary.read_text())
            except Exception as e:
                return {
                    "status": "ERROR", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": f"failed to parse report.json: {e}",
                    "evaluation_dir": str(primary.parent),
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
                "evaluation_dir": str(primary.parent),
                "raw_report_path": str(primary),
            }
        return {
            "status": "ERROR", "test_pass_rate": 0.0,
            "passed": 0, "failed": 0,
            "error_message": f"report.json not found (run_id={run_id})",
            "evaluation_dir": None,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(
        self,
        container_id: Optional[str] = None,
        oh_container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
        network_name: Optional[str] = None,
        workspace_host_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        bridge_pid: Optional[int] = None,
        bridge_socket: Optional[str] = None,
        **_extra,
    ):
        # Bridge server first so it flushes any in-flight responses.
        OpenHandsDispatcher.stop_external(
            pid=bridge_pid, socket_path=bridge_socket,
        )

        # OH harness container ran as --user 0 → state_dir may have root-
        # owned files (events.jsonl), chown back so rmtree works.
        uid, gid = os.getuid(), os.getgid()
        for cid in {oh_container_id, container_id}:
            if not cid:
                continue
            subprocess.run(
                ["docker", "exec", cid, "chown", "-R",
                 f"{uid}:{gid}", _STATE_DIR_IN_OH],
                capture_output=True, check=False, timeout=30,
            )

        targets = [t for t in (
            oh_container_id, container_id,
            task_container_id, task_container_name,
        ) if t]
        for cid in dict.fromkeys(targets):
            subprocess.run(
                ["docker", "rm", "-f", "-v", cid],
                capture_output=True, check=False,
            )

        if network_name:
            r = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"network={network_name}"],
                capture_output=True, text=True,
            )
            for cid in (r.stdout or "").split():
                subprocess.run(["docker", "rm", "-f", "-v", cid],
                               capture_output=True, check=False)
            subprocess.run(["docker", "network", "rm", network_name],
                           capture_output=True, check=False)

        keep = os.environ.get("OC_HOST_KEEP_WORKSPACE", "0") == "1"
        if not keep and workspace_host_path and os.path.isdir(workspace_host_path):
            _force_rmtree(Path(workspace_host_path), ignore_errors=True)
        if not keep and state_dir and os.path.isdir(state_dir):
            _force_rmtree(Path(state_dir), ignore_errors=True)

    def _best_effort_cleanup(
        self, net: str, oh: str, task: Optional[str] = None,
    ) -> None:
        for name in (oh, task):
            if name:
                subprocess.run(
                    ["docker", "rm", "-f", "-v", name],
                    capture_output=True, check=False,
                )
        r = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"network={net}"],
            capture_output=True, text=True,
        )
        for cid in (r.stdout or "").split():
            subprocess.run(
                ["docker", "rm", "-f", "-v", cid],
                capture_output=True, check=False,
            )
        subprocess.run(
            ["docker", "network", "rm", net],
            capture_output=True, check=False,
        )
