"""Terminal-Bench harnesses (docker + singularity).

Two backends, same surface as ``agent_bench/benchmarks/swebench/app.py``:

  - ``TerminalBenchDockerHarness``       — default; mirrors ``SWEBenchHarness``.
                                           Two containers (OC + task) on a per-run
                                           docker network, connected via the
                                           exec-bridge so an OpenClaw agent's
                                           sandbox tool execs land in the task
                                           container. Selected when
                                           ``AB_DEPLOYMENT_CONTAINER_BACKEND=docker`` (the
                                           default for ``run_tb``).
  - ``TerminalBenchSingularityHarness``  — HPC fallback. One singularity
                                           instance of the task .sif. No OC
                                           container; only the ``tb_react``
                                           condition drives it.

Architecture (docker mode):

    host  ──── only vLLM runs here ──── http://localhost:8000/v1
                                  ▲
                                  │
    ┌─────────────────────────────────────┐
    │ oc-<rid>   agent-bench/openclaw:<v> │  (optional; only when the
    │ - openclaw gateway                  │   harness config has an
    │ - exec-bridge client                │   `openclaw` block — i.e. the
    └────────┬────────────────────────────┘   condition is openclaw_agent)
             │
             ↓
    ┌─────────────────────────────────────┐
    │ openclaw-sandbox-<sess>             │
    │ image: <task.toml docker_image>     │
    │ - /app     (rw, agent workspace)    │
    │ - /tests   (ro, vendored tests)     │
    │ - /logs/verifier (rw, reward.txt)   │
    └─────────────────────────────────────┘

For the ``tb_react`` condition (no OC), the harness still brings up the task
container but skips the OC container + bridge; ``env_info["task_container_id"]``
is what the condition's docker-exec calls use.

Anti-cheat note:
    /tests is bound :ro so the agent can't rewrite the test file. /logs/verifier
    is bind-mounted rw — the verifier (`test.sh`) overwrites
    /logs/verifier/reward.txt as its last step.
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
    DEFAULT_OPENCLAW_RUNTIME_IMAGE,
    _force_rmtree,
    _oc_sandbox_name,
)
from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher
from agent_bench.core.config import runtime_endpoint_for_container, load_active_model_and_serving
from agent_bench.core.profile_io import load_profile

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Workspace seeding — drop TB-flavoured BOOTSTRAP.md + _oc_validate.sh
# from the per-app prompts/skills tree into the live /app workspace.
# ----------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BOOTSTRAP_SRC = _THIS_DIR / "openclaw" / "agents" / "BOOTSTRAP.md"
_VALIDATE_SRC = _THIS_DIR / "openclaw" / "skills" / "verify" / "_oc_validate.sh"
_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"


def _seed_workspace_helpers(app_dir: Path, instance_data: Dict) -> None:
    """Drop BOOTSTRAP.md + _oc_validate.sh into the per-run /app workspace.

    Sources of truth (edit these files, not inline strings):
      - `agent_bench/benchmarks/terminal_bench/openclaw/agents/BOOTSTRAP.md` — silences
        OC's `[Bootstrap pending] read BOOTSTRAP.md` injection
        (system-prompt-BDZ2g8e3.js:133-139) and replaces it with explicit
        container-path rules.
      - `agent_bench/benchmarks/terminal_bench/openclaw/skills/verify/_oc_validate.sh`
        — the helper the TB `verify` SKILL.md points at; without it the
        agent either fails the call or hallucinates a host path.

    Idempotent. Best-effort: warn on OSError; the run can proceed without
    these helpers, it'll just be more brittle.
    """
    try:
        shutil.copyfile(_BOOTSTRAP_SRC, app_dir / "BOOTSTRAP.md")
        dst_validate = app_dir / "_oc_validate.sh"
        shutil.copyfile(_VALIDATE_SRC, dst_validate)
        os.chmod(dst_validate, 0o755)
    except OSError as e:
        logger.warning("workspace seed (BOOTSTRAP.md/_oc_validate.sh) failed: %s", e)


# ----------------------------------------------------------------------
# Docker harness
# ----------------------------------------------------------------------


class TerminalBenchDockerHarness:
    """Two-container docker harness for Terminal-Bench tasks.

    Mirrors ``SWEBenchHarness`` end-to-end: per-run network, pre-started task
    container with the OC-predicted name, exec-bridge dispatcher on the host,
    optional OpenClaw container with state_dir + gateway. The only TB-specific
    pieces are the bind layout (``/app`` + ``/tests`` + ``/logs/verifier``
    instead of ``/testbed``) and the scoring path (``bash /tests/test.sh`` →
    ``reward.txt`` instead of SWE-bench's ``run_evaluation``).

    OC is **optional**: the harness only starts it when the active harness
    config carries an ``openclaw`` block (which signals an OC-driven condition
    like ``openclaw_agent``). The ``tb_react`` condition harness config has no
    such block, so the OC container + bridge are skipped and only the task
    container is brought up.
    """

    backend_id = "docker"

    _GATEWAY_READY_RE = re.compile(
        r"\[gateway\][^\n]{0,80}(ready|listening on )", re.IGNORECASE,
    )
    _GATEWAY_READY_TIMEOUT_SECONDS = 180
    _DEFAULT_VERIFIER_TIMEOUT_SECONDS = 900

    def __init__(self, benchmark_data_root: Optional[Path] = None,
                 oc_image: Optional[str] = None,
                 workspace_base: Optional[Path] = None):
        # benchmark_data_root accepted for driver-API parity; unused by TB.
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.oc_image = oc_image or DEFAULT_OPENCLAW_RUNTIME_IMAGE

        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker not available: {r.stderr.strip()}")

        # OC image is only required when the harness config asks for it; we
        # check lazily in prepare_environment so a pure tb_react run on a
        # machine without the OC image still works.

        self.workspace_base = Path(
            workspace_base
            or os.environ.get("AB_HARNESS_WORKSPACE_BASE",
                              os.path.expanduser("~/.cache/oc-eval/workspaces"))
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
        """Pull the first registered agent's id out of the active harness
        config's `openclaw.agents.list[]`. Returns None when the config has
        no openclaw block or no agents."""
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
        repo_id: str,                           # ignored — TB
        base_commit: str,                       # ignored — TB
        container_image: Optional[str] = None,  # ignored — read from task.toml
        workspace_root: Optional[str] = None,   # ignored — TB uses /app
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        if instance_data is None:
            raise RuntimeError(
                "TerminalBenchDockerHarness.prepare_environment requires "
                "instance_data; the driver should pass it through."
            )
        if instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError(
                "TerminalBenchDockerHarness only handles benchmark=terminal_bench "
                f"(got benchmark={instance_data.get('benchmark')!r})"
            )

        docker_image = instance_data["docker_image"]
        tests_host = Path(instance_data["tests_host_path"])
        if not tests_host.is_dir():
            raise FileNotFoundError(f"task tests dir missing: {tests_host}")

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"oc-eval-net-{rid_short}"
        oc_name = f"oc-{rid_short}"

        # Predict OC's sandbox container name — same algorithm as SWE-bench so
        # an OC-driven TB condition (openclaw_agent) finds the task container
        # as a "hot container" instead of spawning a new one.
        #
        # Agent id MUST match three places: (1) the harness config's
        # `openclaw.agents.list[0].id`, (2) the dirname under state_dir/agents/
        # where prompts are dropped, (3) what the condition passes to
        # `openclaw agent --agent <id>` (TerminalBenchOpenClawCondition uses
        # "terminal_bench"). Pull from OC config when present so a custom
        # harness can override.
        sandbox_scope = "agent"
        sandbox_agent_id = self._agent_id_from_harness_cfg() or "terminal_bench"
        scope_key = f"agent:{sandbox_agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        # Per-run host dirs: /app workspace + /logs/verifier (where reward.txt
        # lands) bind targets.
        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        app_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Whether to bring up the OC container + bridge — driven by whether
        # the active harness config has an `openclaw` block.
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

        logger.info(
            "Preparing TB run %s (task image=%s, oc=%s, use_openclaw=%s)",
            run_id, docker_image, self.oc_image if use_openclaw else "(skipped)",
            use_openclaw,
        )

        # 1. Per-run docker network.
        r = subprocess.run(
            ["docker", "network", "create", net_name],
            capture_output=True, text=True,
        )
        if r.returncode != 0 and "already exists" not in (r.stderr or ""):
            raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

        # 2. (Optional) prime /app from the task image so a pre-seeded image
        #    (Dockerfile that drops starter files at WORKDIR) isn't masked by
        #    the empty bind. kv-store-grpc's /app is empty by design so this
        #    is a no-op there.
        try:
            self._prime_app_from_image(docker_image, app_dir)
        except Exception as e:
            logger.warning("TB workspace prime skipped: %s", e)

        # 2b. Seed BOOTSTRAP.md + _oc_validate.sh into /app. BOOTSTRAP.md
        #     silences OC's "[Bootstrap pending] please read BOOTSTRAP.md"
        #     injection (system-prompt-BDZ2g8e3.js:133+) and replaces it
        #     with explicit TB path rules. _oc_validate.sh is what the
        #     `verify` skill claims is present — wire it for real.
        _seed_workspace_helpers(app_dir, instance_data)

        # 3. (Optional) per-run state_dir for OC — only when OC is in play.
        state_dir: Optional[Path] = None
        dispatcher: Optional[OpenClawRemoteDispatcher] = None
        if use_openclaw:
            state_dir = self._build_state_dir(
                run_id=run_id,
                workspace_root="/app",
                sandbox_image=docker_image,
                sandbox_binds=[
                    f"{app_dir}:/app:rw",
                    f"{tests_host}:/tests:ro",
                    f"{logs_dir}:/logs/verifier:rw",
                ],
                container_prefix=sandbox_prefix,
                scope=sandbox_scope,
            )

        # 4. Exec-bridge dispatcher (only meaningful when OC is up; we still
        #    start it so condition-side code that consults env_info["bridge_*"]
        #    has consistent shape).
        if use_openclaw:
            dispatcher = OpenClawRemoteDispatcher()
            try:
                dispatcher.start(
                    state_dir=state_dir,
                    backend="docker",
                    target=task_container_name,
                )
            except Exception:
                self._best_effort_cleanup(net_name, oc_name, task_container_name)
                raise

        # 5. Start the TB task container with the OC-predicted name and the
        #    TB bind layout. Use `--user 0` so pip installs in test.sh land
        #    in the system site-packages (kv-store-grpc tests rely on this).
        task_run_cmd = [
            "docker", "run", "-d",
            "--name", task_container_name,
            "--network", net_name,
            "--user", "0",
            "--label", "openclaw.sandbox=1",
            "--label", f"openclaw.sessionKey={scope_key}",
            "--label", f"oc-eval.run_id={run_id}",
            "-v", f"{app_dir}:/app:rw",
            "-v", f"{tests_host}:/tests:ro",
            "-v", f"{logs_dir}:/logs/verifier:rw",
            "-w", "/app",
            "--entrypoint", "sleep",
            docker_image,
            "infinity",
        ]
        r = subprocess.run(task_run_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            if dispatcher:
                dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise RuntimeError(
                f"task env container start failed: {r.stderr.strip()}"
            )
        task_container_id = r.stdout.strip()

        env_info: Dict = {
            # `container_id` is the field the driver passes positionally into
            # condition.run(). For tb_react this is the TB task container so
            # the existing handlers in driver.py just work; for openclaw_agent
            # we overwrite it below with the OC container id (same as SWE-bench).
            "container_id": task_container_id,
            "container_name": task_container_name,
            "backend": "docker",
            "task_container_id": task_container_id,
            "task_container_name": task_container_name,
            "network_name": net_name,
            "workspace_root": "/app",
            "workspace_host_path": str(app_dir),
            "host_logs_dir": str(logs_dir),
            "sandbox_image": docker_image,
            # TB-specific echo (so conditions don't re-parse task.toml)
            "tb_instruction": instance_data["instruction"],
            "tb_verifier_entrypoint": instance_data.get(
                "verifier_entrypoint", "/tests/test.sh"
            ),
            "tb_reward_file": instance_data.get(
                "reward_file", "/logs/verifier/reward.txt"
            ),
            # For react.py: tells it which engine to use for exec.
            "container_engine": "docker",
            "instance_name": task_container_name,
        }

        if not use_openclaw:
            logger.info(
                "  TB task container up: %s (%s) — OC skipped (no `openclaw` "
                "block in harness config)",
                task_container_name, task_container_id[:12],
            )
            return env_info

        # 6. Start the OC container (mirrors SWEBenchHarness).
        gateway_port = _pick_free_gateway_port()
        gateway_token = secrets.token_hex(24)
        assert dispatcher is not None and state_dir is not None  # for type-checkers

        # Wire the bridge explicitly, matching the SWE-bench harness pattern.
        # Without these env vars, OpenClaw falls back to its built-in Docker
        # tools and bridge-side tool-call logging never runs.
        bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
        bridge_args: List[str] = []
        for mount in bridge_mounts:
            bridge_args += ["-v", f"{mount.host}:{mount.container}:{mount.mode}"]
        for key, value in bridge_env.items():
            bridge_args += ["-e", f"{key}={value}"]

        cmd = [
            "docker", "run", "-d",
            "--name", oc_name,
            "--network", net_name,
            "--add-host", "vllm-host:host-gateway",
            "--user", "0",
            "-v", f"{state_dir}:/root/.openclaw:rw",
            # NB: don't bind {app_dir}:/app on the OC container — the OC image
            # ships `/usr/local/bin/openclaw -> /app/openclaw.mjs`, so a bind
            # over /app masks the binary and the entrypoint fails. The agent's
            # workspace lives in the task container (sandbox); OC reaches it
            # via the bridge, not via a host bind.
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "-w", "/root",
            "-e", f"OPENCLAW_GATEWAY_TOKEN={gateway_token}",
            "-e", f"OPENCLAW_GATEWAY_OTEL_ENDPOINT={os.environ.get('AB_TELEMETRY_ENDPOINT', 'http://vllm-host:4318')}",
            "-e", f"OPENCLAW_RUN_ID={run_id}",
            "-e", "OPENCLAW_RAW_RESPONSE_LOG=/root/.openclaw/logs/llm-raw-responses.jsonl",
            "-e", f"OTEL_RESOURCE_ATTRIBUTES=oc.run_id={run_id}",
            "-e", "DOCKER_API_VERSION=1.44",
            *bridge_args,
            "--entrypoint", "/usr/local/bin/openclaw",
            self.oc_image,
            "gateway", "--allow-unconfigured",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            _release_gateway_port(gateway_port)
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise RuntimeError(f"oc container start failed: {r.stderr.strip()}")
        oc_container_id = r.stdout.strip()

        try:
            self._wait_for_gateway_ready(oc_container_id)
        except Exception:
            _release_gateway_port(gateway_port)
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise

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
            "  ready: oc=%s task=%s bridge_pid=%s net=%s",
            oc_container_id[:12], task_container_id[:12],
            dispatcher.pid, net_name,
        )
        env_info.update({
            # OC takes the `container_id` slot so condition.run() (which uses
            # it for openclaw_agent's docker-exec) targets the OC container.
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

    # ------------------------------------------------------------------
    # State-dir compose (TB flavour). Falls back to SWE-bench prompts/skills
    # when the TB per-app dir is empty — harmless because the prompts are
    # generic ("you have a shell, do the task").
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
        repo_root = os.environ.get("AB_REPO_ROOT", os.getcwd())
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir)
        # Agent dirname must match the id the condition will invoke with
        # `openclaw agent --agent <id>` AND the openclaw.json `agents.list[].id`.
        # TerminalBenchOpenClawCondition uses "terminal_bench"; pull from OC
        # config when present so a custom harness can override.
        agent_id = self._agent_id_from_harness_cfg() or "terminal_bench"
        os.makedirs(state_dir / "agents" / agent_id, exist_ok=True)

        override = os.environ.get("AB_HARNESS_STATE_TEMPLATE")
        if override:
            if not os.path.isdir(override):
                raise FileNotFoundError(
                    f"AB_HARNESS_STATE_TEMPLATE points to {override} but is not a directory."
                )
            shutil.rmtree(state_dir)
            shutil.copytree(override, state_dir)
        else:
            harness_path = os.environ.get(
                "AB_HARNESS_CONFIG",
                os.path.join(repo_root, "profiles/harnesses/openclaw_terminal_bench.yaml"),
            )
            if not os.path.isfile(harness_path):
                raise FileNotFoundError(
                    f"harness config not found at {harness_path} "
                    "(set AB_HARNESS_CONFIG or AB_HARNESS_STATE_TEMPLATE)."
                )
            harness_cfg = load_profile(harness_path)
            oc_cfg = dict(harness_cfg.get("openclaw") or {})
            if not oc_cfg:
                raise ValueError(
                    f"harness config {harness_path} has no `openclaw` block "
                    "(required when use_openclaw=True)."
                )
            model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
            providers = oc_cfg.setdefault("models", {}).setdefault("providers", {})
            agent_model_p = providers.setdefault("agent_model", {})
            agent_model_p["api"] = _OPENAI_CHAT_COMPLETIONS_API
            agent_model_p.setdefault("apiKey", model_cfg.api_key or "sk-dummy")
            agent_model_p["models"] = [{
                "id": model_cfg.name,
                "name": model_cfg.name,
                "api": _OPENAI_CHAT_COMPLETIONS_API,
                "contextWindow": model_cfg.context_window,
                "maxTokens": model_cfg.max_tokens,
            }]
            agents_d = oc_cfg.setdefault("agents", {}).setdefault("defaults", {})
            agents_d.setdefault("model", {})["primary"] = f"agent_model/{model_cfg.name}"
            with open(state_dir / "openclaw.json", "w") as fh:
                json.dump(oc_cfg, fh, indent=2)

            tb_agents = os.path.join(repo_root, "agent_bench/benchmarks/terminal_bench/openclaw/agents")
            tb_skills = os.path.join(repo_root, "agent_bench/benchmarks/terminal_bench/openclaw/skills")
            swe_agents = os.path.join(repo_root, "agent_bench/benchmarks/swebench/openclaw/agents")
            swe_skills = os.path.join(repo_root, "agent_bench/benchmarks/swebench/openclaw/skills")

            # Prefer TB-specific prompts/skills; fall back to SWE-bench's when
            # the TB dir is empty (current state — see openclaw/agents/).
            src_agents = (
                tb_agents if (os.path.isdir(tb_agents) and os.listdir(tb_agents))
                else swe_agents
            )
            src_skills = (
                tb_skills if (os.path.isdir(tb_skills) and os.listdir(tb_skills))
                else swe_skills
            )
            if os.path.isdir(src_agents):
                # BOOTSTRAP.md is workspace-scoped (lives in /app at runtime,
                # see _seed_workspace_helpers). The file in agents/ is just
                # the source-of-truth — don't also copy it into
                # state_dir/agents/<id>/, where it'd be agent-persona context.
                for f in os.listdir(src_agents):
                    if f == "BOOTSTRAP.md":
                        continue
                    s = os.path.join(src_agents, f)
                    if os.path.isfile(s):
                        shutil.copyfile(s, state_dir / "agents" / agent_id / f)
            if os.path.isdir(src_skills):
                shutil.copytree(src_skills, state_dir / "skills")

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
        model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
        serving_endpoint = runtime_endpoint_for_container(
            model_cfg.endpoint, "vllm-host"
        )
        providers = cfg.setdefault("models", {}).setdefault("providers", {})
        agent_model_provider = providers.setdefault("agent_model", {})
        agent_model_provider["api"] = _OPENAI_CHAT_COMPLETIONS_API
        agent_model_provider["baseUrl"] = serving_endpoint
        for model in agent_model_provider.get("models") or []:
            if isinstance(model, dict):
                model["api"] = _OPENAI_CHAT_COMPLETIONS_API
        otel_cfg = (cfg.get("diagnostics") or {}).get("otel") or {}
        if otel_cfg.get("enabled"):
            cfg["diagnostics"]["otel"]["endpoint"] = os.environ.get(
                "AB_TELEMETRY_ENDPOINT", "http://vllm-host:4318",
            )
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return state_dir

    # ------------------------------------------------------------------

    def _prime_app_from_image(self, docker_image: str, app_dir: Path) -> None:
        """Best-effort copy of /app from the image into the host bind dir.
        No-op when the image's /app is empty (kv-store-grpc)."""
        scratch_name = f"oc-eval-tb-prime-{app_dir.name[:12]}-{os.urandom(2).hex()}"
        r = subprocess.run(
            ["docker", "create", "--name", scratch_name,
             "--entrypoint", "true", docker_image],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker create scratch failed: {r.stderr.strip()}")
        try:
            r = subprocess.run(
                ["docker", "cp", f"{scratch_name}:/app/.", str(app_dir) + "/"],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                # docker cp from an empty source dir succeeds with "no such
                # file" on some daemons; treat any failure as "nothing to prime".
                logger.debug(
                    "TB prime: docker cp returned %d (likely empty /app): %s",
                    r.returncode, r.stderr.strip(),
                )
        finally:
            subprocess.run(["docker", "rm", "-f", scratch_name],
                           capture_output=True, check=False)

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
    # exec_in_container — callable for conditions that want to drive the task
    # container directly (tb_react). Matches the singularity harness's shape.
    # ------------------------------------------------------------------

    def exec_in_container(
        self,
        container: str,                  # docker id or name
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

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,            # ignored — TB has no patch
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
        run_dir: Optional[Path] = None,
        **_kw,
    ) -> Dict:
        """Run /tests/test.sh inside the task container. Read reward.txt from
        the host bind target."""
        # Target the task container by name (deterministic from run_id + the
        # agent id used at prepare_environment time).
        run_id = run_id or instance_id
        rid_short = run_id[:12]
        sandbox_agent_id = self._agent_id_from_harness_cfg() or "terminal_bench"
        scope_key = f"agent:{sandbox_agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        host_logs_dir = self.workspace_base / run_id / "logs" / "verifier"

        timeout = timeout_seconds or self._DEFAULT_VERIFIER_TIMEOUT_SECONDS
        if instance_data:
            timeout = int(instance_data.get("verifier_timeout_sec", timeout))

        verifier = (instance_data or {}).get(
            "verifier_entrypoint", "/tests/test.sh"
        )
        logger.info(
            "scoring TB container %s (verifier=%s, timeout=%ds)",
            task_container_name, verifier, timeout,
        )
        result = self.exec_in_container(
            task_container_name,
            ["bash", verifier],
            timeout_seconds=timeout,
        )

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
        host_logs_dir: Optional[str] = None,
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

        keep = os.environ.get("AB_HARNESS_KEEP_WORKSPACE", "0") == "1"
        if not keep and workspace_host_path and os.path.isdir(workspace_host_path):
            # Workspace root is .../app — also clean the .../logs sibling and
            # the parent run dir so per-run dirs don't pile up.
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
# Singularity harness (HPC fallback) — unchanged from the original
# TerminalBenchHarness, just renamed.
# ----------------------------------------------------------------------


class TerminalBenchSingularityHarness:
    """Singularity-instance harness for TB tasks (HPC nodes with no docker)."""

    backend_id = "singularity"

    _PULL_TIMEOUT_SECONDS = 1800
    _INSTANCE_START_TIMEOUT_SECONDS = 60
    _DEFAULT_VERIFIER_TIMEOUT_SECONDS = 900

    def __init__(self, benchmark_data_root: Optional[Path] = None,
                 workspace_base: Optional[Path] = None):
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.workspace_base = Path(
            workspace_base or os.environ.get(
                "AB_HARNESS_WORKSPACE_BASE",
                os.path.expanduser("~/.cache/oc-eval/workspaces"),
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)

        self.sif_dir = Path(os.environ.get(
            "AB_DEPLOYMENT_TASK_IMAGE_DIR",
            str(self.workspace_base.parent / "tb-sif"),
        ))
        self.sif_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which("singularity"):
            raise RuntimeError("singularity not on PATH")

    @staticmethod
    def _sif_name_for(docker_image: str) -> str:
        return docker_image.replace("/", "__").replace(":", "__") + ".sif"

    def _ensure_sif(self, docker_image: str) -> Path:
        sif_path = self.sif_dir / self._sif_name_for(docker_image)
        if sif_path.exists() and sif_path.stat().st_size > 0:
            return sif_path
        logger.info("pulling %s -> %s", docker_image, sif_path)
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
                "TerminalBenchSingularityHarness.prepare_environment requires "
                "instance_data; the driver should pass it through."
            )
        if instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError(
                "TerminalBenchSingularityHarness only handles "
                "benchmark=terminal_bench instances "
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

        _seed_workspace_helpers(app_dir, instance_data)

        instance_name = self._instance_name_for(run_id)
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
            "container_id": None,
            "container_name": instance_name,
            "backend": self.backend_id,
            "workspace_root": "/app",
            "host_workspace_root": str(app_dir),
            "host_logs_dir": str(logs_dir),
            "task_sif": str(sif_path),
            "instance_name": instance_name,
            "container_engine": "singularity",
            "tb_instruction": instance_data["instruction"],
            "tb_verifier_entrypoint": instance_data.get(
                "verifier_entrypoint", "/tests/test.sh"
            ),
            "tb_reward_file": instance_data.get(
                "reward_file", "/logs/verifier/reward.txt"
            ),
        }

    def exec_in_container(
        self,
        instance_name: str,
        argv: List[str],
        timeout_seconds: int = 300,
        stdin_text: Optional[str] = None,
    ) -> Dict:
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
        instance_name = self._instance_name_for(run_id or instance_id)
        verifier = "/tests/test.sh"
        host_logs_dir = (
            self.workspace_base / (run_id or instance_id) / "logs" / "verifier"
        )

        timeout = timeout_seconds or self._DEFAULT_VERIFIER_TIMEOUT_SECONDS
        if instance_data:
            timeout = int(instance_data.get("verifier_timeout_sec", timeout))

        logger.info("scoring TB instance %s (verifier=%s, timeout=%ds)",
                    instance_name, verifier, timeout)
        result = self.exec_in_container(
            instance_name,
            ["bash", verifier],
            timeout_seconds=timeout,
        )

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

    def cleanup(self, container_id: Optional[str] = None, **kwargs) -> None:
        instance_name = kwargs.get("container_name") or kwargs.get("instance_name")
        if instance_name:
            self._try_stop_instance(instance_name)

    @staticmethod
    def _instance_name_for(run_id: str) -> str:
        safe = "".join(c if c.isalnum() or c == "-" else "-" for c in run_id)
        return f"tb-{safe}"[:32]

    @staticmethod
    def _try_stop_instance(name: str) -> None:
        try:
            subprocess.run(
                ["singularity", "instance", "stop", name],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            logger.debug("instance stop %s: ignored exception", name, exc_info=True)


# Default export used by the driver when no AB_DEPLOYMENT_CONTAINER_BACKEND override is set.
# Picks docker on a host with docker, singularity otherwise.
def _default_harness_cls():
    if shutil.which("docker"):
        return TerminalBenchDockerHarness
    if shutil.which("singularity"):
        return TerminalBenchSingularityHarness
    raise RuntimeError(
        "neither docker nor singularity is on PATH; "
        "cannot construct a TerminalBench harness"
    )


# Back-compat alias for `from agent_bench.benchmarks.terminal_bench.app import TerminalBenchHarness`.
# Driver selects via AB_DEPLOYMENT_CONTAINER_BACKEND now (see agent_bench/runtime/driver.py).
TerminalBenchHarness = TerminalBenchDockerHarness
