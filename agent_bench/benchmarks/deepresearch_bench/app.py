"""DeepResearch-Bench harness.

Sibling of ``agent_bench/benchmarks/swebench/openclaw/app.py:SWEBenchHarness`` and
``agent_bench/benchmarks/terminal_bench/app.py:TerminalBenchDockerHarness``. Same two-
container model + exec-bridge dispatcher. The only DR-specific pieces:

  - single shared sandbox image (``hongrui0/dr-sandbox:1.0``) — no
    per-instance image because DR tasks have no per-task dependencies.
  - workspace at ``/workspace`` (rw) — the agent writes ``report.md``,
    optionally ``notes/`` and ``drafts/`` for its own scratch.
  - ``PROMPT.md`` is seeded into the sandbox before the agent starts so
    the agent reads the research question from the filesystem (mirrors
    Terminal-Bench's ``/tests`` + BOOTSTRAP.md pattern).
  - ``BRAVE_API_KEY`` + ``FIRECRAWL_API_KEY`` are passed into the OC
    container so OC's native ``web_search`` + ``web_fetch`` tools work.
    No new patch / no new bridge backend — see plan doc Q1 / Path A.
  - ``run_tests`` collects ``/workspace/report.md`` from the sandbox and
    shells out to the vendored RACE + FACT judge (DeepSeek-V4, host-side).

Architecture (docker mode):

    host  ── vLLM @ localhost:8000 / DeepSeek API (judge) / Brave / Jina
                                  ▲
                                  │
    ┌─────────────────────────────────────┐
    │ oc-<rid>   agent-bench/openclaw:<v> │  (optional; only when the
    │ - openclaw gateway                  │   harness config has an
    │ - exec-bridge client                │   `openclaw` block)
    │ - env: BRAVE_API_KEY,               │
    │        FIRECRAWL_API_KEY            │
    └────────┬────────────────────────────┘
             │
             ↓
    ┌─────────────────────────────────────┐
    │ openclaw-sandbox-<sess>             │
    │ image: hongrui0/dr-sandbox:1.0   │
    │ - /workspace      (rw)              │
    │   - PROMPT.md     (ro semantically) │
    │   - report.md     (agent writes)    │
    └─────────────────────────────────────┘
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent_bench.benchmarks.conditions.gateway_helpers import (
    pick_free_gateway_port as _pick_free_gateway_port,
    release_gateway_port as _release_gateway_port,
)
from agent_bench.benchmarks.swebench.app import (
    DEFAULT_OC_IMAGE,
    _force_rmtree,
    _oc_sandbox_name,
)
from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher
from agent_bench.core.config import endpoint_for_container, load_active_model_and_serving
from agent_bench.core.profile_io import load_profile

logger = logging.getLogger(__name__)


DEFAULT_SANDBOX_IMAGE = os.environ.get(
    "DR_SANDBOX_IMAGE", "hongrui0/dr-sandbox:1.0",
)
_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"

# Env vars whose values must reach the OC container so OC's native web
# tools authenticate. Currently only Brave (OC's web_fetch backend does
# not require a key in this deployment). Any unset key is silently
# skipped — the agent will see a missing-key error from OC at tool-call
# time, which is the correct behaviour.
_OC_PASSTHROUGH_ENV_KEYS = (
    "BRAVE_API_KEY",
)

# Workspace-scoped agent-context files. OC reads /workspace/AGENTS.md,
# SOUL.md, and USER.md at boot and injects them into the "Project Context"
# block of the system prompt (falling back to a generic stock template if
# absent — which silently shadows our DR-specific citation rules). The
# bootstrap message explicitly tells the agent to read BOOTSTRAP.md from
# the workspace. So all of these must live in /workspace, not in the
# agent-definition state_dir.
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_AGENT_FILES = ("AGENTS.md", "BOOTSTRAP.md", "SOUL.md", "USER.md")


class DeepResearchHarness:
    """Two-container docker harness for DeepResearch-Bench tasks.

    Mirrors ``TerminalBenchDockerHarness`` end-to-end: per-run network,
    pre-started sandbox container with the OC-predicted name, exec-bridge
    dispatcher on the host, OpenClaw container with state_dir + gateway.

    DR-specific pieces are documented inline at their occurrence — search
    for ``# DR:`` to find them.
    """

    backend_id = "docker"

    _GATEWAY_READY_RE = re.compile(
        r"\[gateway\][^\n]{0,80}(ready|listening on )", re.IGNORECASE,
    )
    _GATEWAY_READY_TIMEOUT_SECONDS = 180
    _DEFAULT_JUDGE_TIMEOUT_SECONDS = 1800

    def __init__(
        self,
        swebench_root: Optional[Path] = None,
        oc_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ):
        # swebench_root accepted for driver-API parity; unused here.
        self.swebench_root = Path(swebench_root) if swebench_root else None
        self.oc_image = oc_image or DEFAULT_OC_IMAGE

        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker not available: {r.stderr.strip()}")

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

    @staticmethod
    def _has_openclaw_block(harness_cfg_path: Optional[str]) -> bool:
        if not harness_cfg_path or not os.path.isfile(harness_cfg_path):
            return False
        try:
            return bool(load_profile(harness_cfg_path).get("openclaw"))
        except Exception:
            return False

    @staticmethod
    def _agent_id_from_harness_cfg() -> Optional[str]:
        """First registered agent id from the active harness config."""
        p = os.environ.get("AB_HARNESS_CONFIG")
        if not p or not os.path.isfile(p):
            return None
        try:
            cfg = load_profile(p)
        except Exception:
            return None
        agents = ((cfg.get("openclaw") or {}).get("agents") or {}).get("list") or []
        for a in agents:
            if isinstance(a, dict) and a.get("id"):
                return str(a["id"])
        return None

    # ------------------------------------------------------------------

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,                           # ignored — DR
        base_commit: str,                       # ignored — DR
        container_image: Optional[str] = None,  # optional override
        workspace_root: Optional[str] = None,   # ignored — DR uses /workspace
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        if instance_data is None:
            raise RuntimeError(
                "DeepResearchHarness.prepare_environment requires "
                "instance_data; the driver should pass it through."
            )
        if instance_data.get("benchmark") != "deepresearch_bench":
            raise RuntimeError(
                "DeepResearchHarness only handles benchmark=deepresearch_bench "
                f"(got benchmark={instance_data.get('benchmark')!r})"
            )

        docker_image = (
            container_image
            or instance_data.get("docker_image")
            or DEFAULT_SANDBOX_IMAGE
        )

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"oc-eval-net-{rid_short}"
        oc_name = f"oc-{rid_short}"

        # Predict OC's sandbox name (same algorithm as SWE-bench / TB).
        sandbox_scope = "agent"
        sandbox_agent_id = self._agent_id_from_harness_cfg() or "deep_researcher"
        scope_key = f"agent:{sandbox_agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        # Per-run host dirs: just /workspace; no /tests or /logs/verifier.
        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        ws_host = run_root / "workspace"
        ws_host.mkdir(parents=True, exist_ok=True)

        # DR: seed PROMPT.md so the agent reads the task spec from the
        # filesystem (consistent with the AGENTS.md instructions).
        self._seed_workspace(ws_host, instance_data)

        harness_cfg_path = os.environ.get("AB_HARNESS_CONFIG")
        use_openclaw = self._has_openclaw_block(harness_cfg_path)
        if use_openclaw:
            r = subprocess.run(
                ["docker", "image", "inspect", self.oc_image],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"openclaw image not found: {self.oc_image}\n"
                    "Build it first: bash agent_bench/harnesses/openclaw/docker/build.sh"
                )
        # Sandbox image is published on Docker Hub — pull on first use.
        r = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.info("  sandbox image %s not local, pulling…", docker_image)
            r = subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"sandbox image pull failed: {docker_image}\n"
                    f"{r.stderr.strip()}"
                )

        logger.info(
            "Preparing DR run %s (sandbox=%s, oc=%s, use_openclaw=%s)",
            run_id, docker_image,
            self.oc_image if use_openclaw else "(skipped)", use_openclaw,
        )

        # Track partial state for unified rollback on any failure below.
        state_dir: Optional[Path] = None
        dispatcher: Optional[OpenClawRemoteDispatcher] = None
        gateway_port: Optional[int] = None

        def _rollback() -> None:
            if gateway_port is not None:
                _release_gateway_port(gateway_port)
            if dispatcher is not None:
                try:
                    dispatcher.stop()
                except Exception:
                    pass
            self._best_effort_cleanup(net_name, oc_name, task_container_name)

        try:
            # 1. Per-run docker network.
            r = subprocess.run(
                ["docker", "network", "create", net_name],
                capture_output=True, text=True,
            )
            if r.returncode != 0 and "already exists" not in (r.stderr or ""):
                raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

            # 2. State dir + exec-bridge dispatcher (only when OC is in play).
            if use_openclaw:
                state_dir = self._build_state_dir(
                    run_id=run_id,
                    workspace_root="/workspace",
                    sandbox_image=docker_image,
                    sandbox_binds=[f"{ws_host}:/workspace:rw"],
                    container_prefix=sandbox_prefix,
                    scope=sandbox_scope,
                )
                dispatcher = OpenClawRemoteDispatcher()
                dispatcher.start(
                    state_dir=state_dir,
                    backend="docker",
                    target=task_container_name,
                )

            # 3. Start the sandbox container.
            task_run_cmd = [
                "docker", "run", "-d",
                "--name", task_container_name,
                "--network", net_name,
                "--user", "0",
                "--label", "openclaw.sandbox=1",
                "--label", f"openclaw.sessionKey={scope_key}",
                "--label", f"oc-eval.run_id={run_id}",
                "-v", f"{ws_host}:/workspace:rw",
                "-w", "/workspace",
                "--entrypoint", "sleep",
                docker_image,
                "infinity",
            ]
            r = subprocess.run(task_run_cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(
                    f"sandbox container start failed: {r.stderr.strip()}"
                )
            task_container_id = r.stdout.strip()

            env_info: Dict = {
                "container_id": task_container_id,
                "container_name": task_container_name,
                "backend": "docker",
                "task_container_id": task_container_id,
                "task_container_name": task_container_name,
                "network_name": net_name,
                "workspace_root": "/workspace",
                "workspace_host_path": str(ws_host),
                "sandbox_image": docker_image,
                "dr_prompt": instance_data["prompt"],
                "dr_language": instance_data.get("language", "en"),
                "dr_report_file": instance_data.get(
                    "report_file", "/workspace/report.md",
                ),
                "container_engine": "docker",
                "instance_name": task_container_name,
            }

            if not use_openclaw:
                logger.info(
                    "  DR sandbox up: %s (%s) — OC skipped (no `openclaw` block)",
                    task_container_name, task_container_id[:12],
                )
                return env_info

            # 4. Start the OC container.
            gateway_port = _pick_free_gateway_port()
            gateway_token = secrets.token_hex(24)
            assert dispatcher is not None and state_dir is not None

            # DR: forward Brave / firecrawl keys to OC so its native web tools
            # can call out. Skip silently when a key isn't set in host env —
            # the agent will get a runtime error from OC if it actually tries
            # to invoke an unconfigured tool.
            web_env_args: List[str] = []
            for key in _OC_PASSTHROUGH_ENV_KEYS:
                val = os.environ.get(key)
                if val:
                    web_env_args.extend(["-e", f"{key}={val}"])
                else:
                    logger.warning(
                        "  %s not set in host env — agent's web tool calls "
                        "depending on it will fail at runtime", key,
                    )

            cmd = [
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
                *web_env_args,
                *dispatcher.agent_container_args(),
                "--entrypoint", "/usr/local/bin/openclaw",
                self.oc_image,
                "gateway", "--allow-unconfigured",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"oc container start failed: {r.stderr.strip()}")
            oc_container_id = r.stdout.strip()

            self._wait_for_gateway_ready(oc_container_id)

            gw_env = (
                f"export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:18789\n"
                f"export OPENCLAW_GATEWAY_TOKEN={gateway_token}\n"
                f"export OPENCLAW_RUN_ID={run_id}\n"
            )
            subprocess.run(
                ["docker", "exec", oc_container_id, "bash", "-c",
                 f"cat > /tmp/oc_gateway_env <<'EOF'\n{gw_env}EOF"],
                capture_output=True, text=True,
            )

            logger.info(
                "  ready: oc=%s sandbox=%s bridge_pid=%s net=%s",
                oc_container_id[:12], task_container_id[:12],
                dispatcher.pid, net_name,
            )
            env_info.update({
                "container_id": oc_container_id,
                "container_name": oc_name,
                "oc_container_id": oc_container_id,
                "state_dir": str(state_dir),
                "gateway_port": gateway_port,
                "gateway_token": gateway_token,
                "gateway_url": "ws://127.0.0.1:18789",
                **dispatcher.env_info(),
            })
            return env_info
        except Exception:
            _rollback()
            raise

    # ------------------------------------------------------------------

    @staticmethod
    def _seed_workspace(ws_host: Path, instance_data: Dict) -> None:
        """Write PROMPT.md + agent-context files into /workspace.

        Beyond the task spec (PROMPT.md), this seeds AGENTS.md / BOOTSTRAP.md /
        SOUL.md / USER.md so OC's project-context injection picks up our
        DR-specific persona + strict citation rules instead of its stock
        generic workspace template.
        """
        prompt = instance_data.get("prompt", "")
        topic = instance_data.get("topic", "")
        language = instance_data.get("language", "en")
        instance_id = instance_data.get("instance_id", "")
        body = (
            f"# DeepResearch task {instance_id}\n"
            f"\n"
            f"topic: {topic}\n"
            f"language: {language}\n"
            f"\n"
            f"## Research question\n"
            f"\n"
            f"{prompt}\n"
        )
        (ws_host / "PROMPT.md").write_text(body, encoding="utf-8")

        src_dir = _THIS_DIR / "openclaw" / "agents"
        for name in _WORKSPACE_AGENT_FILES:
            src = src_dir / name
            if not src.is_file():
                continue
            try:
                shutil.copyfile(src, ws_host / name)
            except OSError as e:
                logger.warning("workspace seed %s failed: %s", name, e)

    # ------------------------------------------------------------------
    # State dir compose (DR flavour).
    # ------------------------------------------------------------------

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
        kit_root = os.environ.get("KIT_ROOT", os.getcwd())
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir)
        agent_id = self._agent_id_from_harness_cfg() or "deep_researcher"
        os.makedirs(state_dir / "agents" / agent_id, exist_ok=True)

        override = os.environ.get("OC_HOST_STATE_TEMPLATE")
        if override:
            if not os.path.isdir(override):
                raise FileNotFoundError(
                    f"OC_HOST_STATE_TEMPLATE points to {override} but is not a directory."
                )
            shutil.rmtree(state_dir)
            shutil.copytree(override, state_dir)
        else:
            harness_path = os.environ.get(
                "AB_HARNESS_CONFIG",
                os.path.join(kit_root, "profiles/harnesses/openclaw_deepresearch_bench_brave.yaml"),
            )
            if not os.path.isfile(harness_path):
                raise FileNotFoundError(
                    f"harness config not found at {harness_path} "
                    "(set AB_HARNESS_CONFIG or OC_HOST_STATE_TEMPLATE)."
                )
            harness_cfg = load_profile(harness_path)
            oc_cfg = dict(harness_cfg.get("openclaw") or {})
            if not oc_cfg:
                raise ValueError(
                    f"harness config {harness_path} has no `openclaw` block "
                    "(required when use_openclaw=True)."
                )
            model_cfg, _serving_cfg = load_active_model_and_serving(kit_root=kit_root)
            providers = oc_cfg.setdefault("models", {}).setdefault("providers", {})
            vllm_p = providers.setdefault("vllm", {})
            vllm_p["api"] = _OPENAI_CHAT_COMPLETIONS_API
            vllm_p.setdefault("apiKey", model_cfg.api_key or "sk-dummy")
            vllm_p["models"] = [{
                "id": model_cfg.name,
                "name": model_cfg.name,
                "api": _OPENAI_CHAT_COMPLETIONS_API,
                "contextWindow": model_cfg.context_window,
                "maxTokens": model_cfg.max_tokens,
            }]
            agents_d = oc_cfg.setdefault("agents", {}).setdefault("defaults", {})
            agents_d.setdefault("model", {})["primary"] = f"vllm/{model_cfg.name}"
            with open(state_dir / "openclaw.json", "w") as fh:
                json.dump(oc_cfg, fh, indent=2)

            dr_agents = os.path.join(kit_root, "agent_bench/benchmarks/deepresearch_bench/openclaw/agents")
            dr_skills = os.path.join(kit_root, "agent_bench/benchmarks/deepresearch_bench/openclaw/skills")

            if os.path.isdir(dr_agents):
                # BOOTSTRAP.md is workspace-scoped (seeded into /workspace by
                # _seed_workspace; OC's bootstrap message tells the agent to
                # read it from there). Don't also drop it into the agent-
                # definition state_dir.
                for f in os.listdir(dr_agents):
                    if f == "BOOTSTRAP.md":
                        continue
                    s = os.path.join(dr_agents, f)
                    if os.path.isfile(s):
                        shutil.copyfile(s, state_dir / "agents" / agent_id / f)
            if os.path.isdir(dr_skills):
                shutil.copytree(dr_skills, state_dir / "skills")

        cfg_path = state_dir / "openclaw.json"
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        agents_defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
        agents_defaults["workspace"] = workspace_root
        agents_defaults["sandbox"] = {
            "mode": "all",
            "scope": scope,
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
        if (state_dir / "skills").is_dir():
            cfg.setdefault("skills", {}).setdefault("load", {})["extraDirs"] = [
                "/root/.openclaw/skills"
            ]
        model_cfg, _serving_cfg = load_active_model_and_serving(kit_root=kit_root)
        vllm_endpoint = os.environ.get(
            "OC_VLLM_BASE_URL",
            endpoint_for_container(model_cfg.endpoint, "vllm-host"),
        )
        providers = cfg.setdefault("models", {}).setdefault("providers", {})
        vllm_provider = providers.setdefault("vllm", {})
        vllm_provider["api"] = _OPENAI_CHAT_COMPLETIONS_API
        vllm_provider["baseUrl"] = vllm_endpoint
        for model in vllm_provider.get("models") or []:
            if isinstance(model, dict):
                model["api"] = _OPENAI_CHAT_COMPLETIONS_API
        otel_cfg = (cfg.get("diagnostics") or {}).get("otel") or {}
        if otel_cfg.get("enabled"):
            cfg["diagnostics"]["otel"]["endpoint"] = os.environ.get(
                "OC_OTEL_ENDPOINT", "http://vllm-host:4318",
            )
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return state_dir

    # ------------------------------------------------------------------

    def _wait_for_gateway_ready(self, oc_container_id: str) -> None:
        deadline = time.time() + self._GATEWAY_READY_TIMEOUT_SECONDS
        last_log_tail = ""
        while time.time() < deadline:
            r = subprocess.run(
                ["docker", "logs", oc_container_id],
                capture_output=True, text=True, timeout=30,
            )
            log_text = (r.stdout or "") + (r.stderr or "")
            if self._GATEWAY_READY_RE.search(log_text):
                return
            last_log_tail = log_text[-1500:]
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", oc_container_id],
                capture_output=True, text=True,
            )
            if (r.stdout or "").strip() != "true":
                raise RuntimeError(
                    f"openclaw container exited before gateway ready. "
                    f"last log:\n{last_log_tail}"
                )
            time.sleep(2)
        raise RuntimeError(
            f"gateway did not signal ready within "
            f"{self._GATEWAY_READY_TIMEOUT_SECONDS}s. last log:\n{last_log_tail}"
        )

    # ------------------------------------------------------------------
    # exec_in_container — callable for conditions / cleanup paths.
    # ------------------------------------------------------------------

    def exec_in_container(
        self,
        container: str,
        argv: List[str],
        timeout_seconds: int = 300,
        stdin_text: Optional[str] = None,
    ) -> Dict:
        full = ["docker", "exec", "-i", container, *argv]
        try:
            r = subprocess.run(
                full,
                input=stdin_text,
                capture_output=True, text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            def _as_str(x):
                if x is None:
                    return ""
                return (x.decode("utf-8", errors="replace")
                        if isinstance(x, (bytes, bytearray)) else x)
            return {
                "rc": -1,
                "stdout": _as_str(e.stdout),
                "stderr": (_as_str(e.stderr)
                           + f"\n[harness] timeout after {timeout_seconds}s"),
                "timed_out": True,
            }
        return {
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "timed_out": False,
        }

    # ------------------------------------------------------------------

    def collect_report(
        self,
        task_container_name: str,
        report_file: str = "/workspace/report.md",
    ) -> str:
        """Read the agent-written report from the sandbox container."""
        r = self.exec_in_container(
            task_container_name, ["cat", report_file], timeout_seconds=30,
        )
        if r["rc"] != 0:
            raise RuntimeError(
                f"could not read {report_file} from {task_container_name}: "
                f"rc={r['rc']}, stderr={r['stderr'][-500:]}"
            )
        return r["stdout"]

    # ------------------------------------------------------------------

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,            # ignored
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        """Collect the agent's report and run RACE + FACT against it."""
        # Import lazily — judge_runner pulls in vendor/utils/api.py which
        # only resolves cleanly when DEEPSEEK_API_KEY etc are set.
        from agent_bench.benchmarks.deepresearch_bench.eval import judge_runner

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        sandbox_agent_id = self._agent_id_from_harness_cfg() or "deep_researcher"
        scope_key = f"agent:{sandbox_agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        report_file = (instance_data or {}).get(
            "report_file", "/workspace/report.md",
        )
        prompt = (instance_data or {}).get("prompt", "")
        language = (instance_data or {}).get("language", "en")

        # Step 1: lift the agent's report.
        host_run_root = self.workspace_base / run_id
        out_dir = host_run_root / "judge"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            article = self.collect_report(task_container_name, report_file)
        except RuntimeError as e:
            logger.warning("Report missing for %s: %s", instance_id, e)
            article = ""

        host_report = out_dir / "report.md"
        host_report.write_text(article, encoding="utf-8")

        if not article.strip():
            err_payload = {
                "instance_id": instance_id,
                "race": None, "fact": None,
                "race_error": "agent produced no report.md",
                "fact_error": "agent produced no report.md",
            }
            (out_dir / "score.json").write_text(
                json.dumps(err_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return _score_to_run_tests_result(
                err_payload, host_report, out_dir,
                error_message="empty report",
            )

        # DR_SKIP_EVAL=1 — collect-only mode (agent phase). The driver still
        # persists `host_report` to `run_dir/report.md` via verdict["report_path"];
        # re-grade later by running `eval/runner.py` over that sweep dir.
        if os.environ.get("DR_SKIP_EVAL") == "1":
            logger.info("DR_SKIP_EVAL=1: skipping RACE+FACT, report.md persisted")
            return {
                "status": "DEFERRED",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 0,
                "race": None,
                "fact": None,
                "race_error": "deferred — run eval phase separately",
                "fact_error": "deferred — run eval phase separately",
                "report_path": str(host_report),
                "evaluation_dir": str(out_dir),
                "error_message": "DR_SKIP_EVAL=1 (agent-only mode)",
            }

        # Step 2: run RACE + FACT on the host.
        try:
            result = judge_runner.run(
                instance_id=instance_id,
                prompt=prompt,
                language=language,
                article=article,
                output_dir=out_dir,
                run_id=run_id,
            )
        except Exception as e:
            logger.exception("judge_runner crashed for %s", instance_id)
            err_payload = {
                "instance_id": instance_id,
                "race": None, "fact": None,
                "race_error": f"judge_runner exception: {e}",
                "fact_error": f"judge_runner exception: {e}",
            }
            (out_dir / "score.json").write_text(
                json.dumps(err_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return _score_to_run_tests_result(
                err_payload, host_report, out_dir,
                error_message=str(e),
            )

        # judge_runner.run already wrote score.json
        from dataclasses import asdict
        payload = {
            "instance_id": result.instance_id,
            "race": asdict(result.race) if result.race else None,
            "fact": asdict(result.fact) if result.fact else None,
            "race_error": result.race_error,
            "fact_error": result.fact_error,
        }
        return _score_to_run_tests_result(payload, host_report, out_dir)

    # ------------------------------------------------------------------

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
    ):
        if gateway_port is not None:
            _release_gateway_port(gateway_port)
        OpenClawRemoteDispatcher.stop_external(
            pid=bridge_pid, socket_path=bridge_socket,
        )

        uid = os.getuid()
        gid = os.getgid()
        for cid in {oc_container_id, container_id}:
            if not cid:
                continue
            subprocess.run(
                ["docker", "exec", cid, "chown", "-R",
                 f"{uid}:{gid}", "/root/.openclaw"],
                capture_output=True, check=False, timeout=30,
            )

        targets = [t for t in (oc_container_id, container_id,
                               task_container_id, task_container_name,
                               container_name)
                   if t]
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
            run_root = Path(workspace_host_path).parent
            _force_rmtree(run_root, ignore_errors=True)
        if not keep and state_dir and os.path.isdir(state_dir):
            _force_rmtree(Path(state_dir), ignore_errors=True)

    # ------------------------------------------------------------------

    def _best_effort_cleanup(
        self,
        net: str,
        oc: str,
        task: Optional[str] = None,
    ) -> None:
        for name in (oc, task):
            if name:
                subprocess.run(["docker", "rm", "-f", "-v", name],
                               capture_output=True, check=False)
        r = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"network={net}"],
            capture_output=True, text=True,
        )
        for cid in (r.stdout or "").split():
            subprocess.run(["docker", "rm", "-f", "-v", cid],
                           capture_output=True, check=False)
        subprocess.run(["docker", "network", "rm", net],
                       capture_output=True, check=False)


# ----------------------------------------------------------------------

# RACE overall_score is in [0,1]; we PASS at >= 0.5 by default (the
# upstream criterion for "competitive with reference"). Tune via env.
_DR_PASS_THRESHOLD = float(os.environ.get("DR_PASS_THRESHOLD", "0.5"))


def _score_to_run_tests_result(
    payload: Dict,
    host_report: Path,
    judge_dir: Path,
    *,
    error_message: Optional[str] = None,
) -> Dict:
    """Squeeze the judge payload into the SWE-bench-shaped run_tests result."""
    race = payload.get("race") or {}
    fact = payload.get("fact") or {}
    overall = float(race.get("overall", 0.0)) if race else 0.0
    passed = (
        race is not None
        and overall >= _DR_PASS_THRESHOLD
        and payload.get("race_error") is None
    )
    status = (
        "PASS" if passed
        else ("ERROR" if (payload.get("race_error") and not race) else "FAIL")
    )
    return {
        "status": status,
        "test_pass_rate": overall,
        "passed": 1 if passed else 0,
        "failed": 0 if passed else 1,
        "race": race or None,
        "fact": fact or None,
        "race_error": payload.get("race_error"),
        "fact_error": payload.get("fact_error"),
        "report_path": str(host_report),
        "evaluation_dir": str(judge_dir),
        "error_message": error_message,
    }
