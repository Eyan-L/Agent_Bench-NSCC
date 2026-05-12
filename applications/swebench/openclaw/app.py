"""SWE-bench application — OpenClaw native sandbox docker harness.

Sibling of `applications/swebench/openhands/app.py`. Implements the same
lifecycle contract (`prepare_environment`, `run_tests`, `cleanup`,
`collect_patch`, ...) so the driver treats both harness classes
interchangeably — the choice is made per-sweep from the harness JSON.

Layered model (no host-mode anywhere):

    host  ──── only vLLM runs here ────  http://localhost:8000/v1
                                   ▲
                                   │ OpenAI-compat HTTP (vllm-host:8000)
                                   │
    ┌──────────────────────────────────────┐
    │ oc-<rid>  agent-bench/openclaw:<v>   │
    │ - openclaw gateway (long-running)    │
    │ - openclaw agent CLI (per session)   │
    │ - docker CLI + /var/run/docker.sock  │
    └────────┬─────────────────────────────┘
             │  spawns per-session via OpenClaw's
             │  sandbox.docker config:
             ↓
    ┌──────────────────────────────────────┐
    │ openclaw-sandbox-<sess>              │
    │ image: swebench/sweb.eval.<inst>     │
    │ - per-instance python env + pytest   │
    │ - workspace bind-mounted at /testbed │
    │ - tool exec from agent lands here    │
    └──────────────────────────────────────┘

The gateway routes the agent's `tools.exec` (bash, file edits) into the
sandbox container automatically — see `agents.defaults.sandbox.docker` in
the per-run openclaw.json. The agent never knows there are two containers;
it just sees a shell with /testbed cloned and a python env.

Scoring (`run_tests`) is independent of the sandbox — it shells out to
`python -m swebench.harness.run_evaluation` on the host; that command
spawns its own scoring containers and writes report.json.

`SingularityHarness` is the singularity counterpart (two .sif instances);
not yet implemented (NotImplementedError stub).
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from applications.conditions.gateway_helpers import (
    pick_free_gateway_port as _pick_free_gateway_port,
    release_gateway_port as _release_gateway_port,
)
from applications.swebench.app import (
    DEFAULT_OC_IMAGE,
    _force_rmtree,
    _oc_sandbox_name,
)
from orchestration.dispatcher.openclaw_remote import OpenClawRemoteDispatcher

logger = logging.getLogger(__name__)


# Static SWE-bench agent prompt filenames mounted from
# applications/swebench/openclaw/agents/.
_SWEBENCH_AGENT_PROMPTS = (
    "AGENTS.md", "SOUL.md", "IDENTITY.md", "USER.md", "TOOLS.md", "HEARTBEAT.md",
)


class SWEBenchHarness:
    """Two-container docker harness.

    `prepare_environment` brings up: a per-run docker network, a task
    container (the SWE-bench instance image holding /testbed and its
    python env), and an openclaw container (the patched OC runtime image
    holding the gateway + agent CLI). They share the workspace via
    bind-mount and a docker network so the agent's gateway is reachable
    over WS without exposing a host port.

    Conditions that need to invoke `openclaw agent` should `docker exec`
    into the openclaw container's id (`oc_container_id` in env_info).
    """

    backend_id = "docker"

    # OpenClaw 2026.4+ logs `[gateway] listening on ws://...` at readiness;
    # earlier versions said `[gateway] ... ready`.
    _GATEWAY_READY_RE = re.compile(
        r"\[gateway\][^\n]{0,80}(ready|listening on )", re.IGNORECASE,
    )
    _GATEWAY_READY_TIMEOUT_SECONDS = 180

    def __init__(self, swebench_root: Path, oc_image: Optional[str] = None,
                 workspace_base: Optional[Path] = None):
        self.swebench_root = Path(swebench_root)
        self.oc_image = oc_image or DEFAULT_OC_IMAGE

        # Pre-flight: swebench python pkg + docker daemon + oc image present.
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
            ["docker", "image", "inspect", self.oc_image],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"openclaw image not found: {self.oc_image}\n"
                "Build it first: bash harnesses/openclaw/docker/build.sh"
            )

        self.workspace_base = Path(
            workspace_base
            or os.environ.get("OC_HOST_WORKSPACE_BASE",
                              os.path.expanduser("~/.cache/oc-eval/workspaces"))
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        # Per-run state dir (mounted ro into the openclaw container at
        # /root/.openclaw/) lives next to workspaces.
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
        # Resolve the per-instance SWE-bench image when the sweep config uses
        # the `UNUSED` placeholder (multi-instance sweeps can't pin a single
        # image at sweep level). Mirrors swebench.harness TestSpec.instance_image_key
        # for remote images: `{namespace}/sweb.eval.{arch}.{instance_id_lower}:{tag}`
        # with `__` -> `_1776_` (docker tag-name encoding).
        if "UNUSED" in container_image:
            container_image = (
                f"swebench/sweb.eval.x86_64.{instance_id.lower()}:latest"
                .replace("__", "_1776_")
            )

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"oc-eval-net-{rid_short}"
        oc_name = f"oc-{rid_short}"

        # The task env container is started by THIS harness (not by OC), so
        # we own its full lifecycle. We still name it exactly what OC would
        # compute (containerPrefix + slugify(scopeKey)) — when OC's agent
        # later resolves its sandbox, it finds our container already running
        # and `docker exec`s into it instead of spawning a new one.
        #
        # Per-run containerPrefix guarantees the name is unique per run, so
        # a leaked container from a prior run can never be reused by mistake.
        sandbox_scope = "agent"
        sandbox_agent_id = "swebench"
        scope_key = f"agent:{sandbox_agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        logger.info(
            "Preparing run %s (sandbox image=%s, oc=%s, repo=%s@%s, task=%s)",
            run_id, container_image, self.oc_image, repo_id, base_commit[:12],
            task_container_name,
        )

        # 1. Per-run host workspace dir. We prime it from the task image's
        #    /testbed (pre-checked-out at base_commit), then bind it into the
        #    task container at /testbed. Living on the host lets us collect
        #    the patch via plain git after the run.
        ws_host = self.workspace_base / run_id
        if ws_host.exists():
            _force_rmtree(ws_host, ignore_errors=True)
        ws_host.mkdir(parents=True)

        # 2. Per-run state dir — built AFTER ws_host is known, so the per-run
        #    openclaw.json can encode the sandbox.docker prefix/scope/image.
        state_dir = self._build_state_dir(
            run_id=run_id,
            workspace_root=workspace_root,
            sandbox_image=container_image,
            sandbox_binds=[f"{ws_host}:/testbed:rw"],
            container_prefix=sandbox_prefix,
            scope=sandbox_scope,
        )

        # 3. Per-run docker network (for OC↔task container, if needed).
        r = subprocess.run(
            ["docker", "network", "create", net_name],
            capture_output=True, text=True,
        )
        if r.returncode != 0 and "already exists" not in (r.stderr or ""):
            raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

        # 4. Prime the host workspace from the task image — required because
        #    bind-mounting an empty <ws_host>:/testbed would mask the image's
        #    pre-checked-out repo and the agent would see /testbed as empty.
        try:
            self._prime_workspace_from_image(container_image, ws_host)
        except Exception:
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise

        # 4.5. Start the SWE-bench task env container ourselves with the
        #      name OC will look for. We mount <ws_host> at /testbed:rw so
        #      the agent's edits land on the host (for patch extraction)
        #      and `sleep infinity` keeps it alive for `docker exec` from OC.
        # SWE-bench images ship astropy (and the rest of the pinned dep set)
        # pre-installed in a conda env at /opt/miniconda3/envs/testbed. The
        # image default PATH points at the base conda env (no astropy), so
        # `docker exec ... python` would hit /opt/miniconda3/bin/python and
        # the agent would burn its budget on `pip install`. Pre-pending the
        # testbed env's bin to PATH (no `conda activate` needed) makes
        # `python` / `pip` resolve to the pre-built env directly.
        task_path = (
            "/opt/miniconda3/envs/testbed/bin:"
            "/opt/miniconda3/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        task_run_cmd = [
            "docker", "run", "-d",
            "--name", task_container_name,
            "--network", net_name,
            "--user", "0",
            "--label", "openclaw.sandbox=1",
            "--label", f"openclaw.sessionKey={scope_key}",
            "--label", f"oc-eval.run_id={run_id}",
            "-v", f"{ws_host}:/testbed:rw",
            "-w", "/testbed",
            "-e", f"PATH={task_path}",
            "--entrypoint", "sleep",
            container_image,
            "infinity",
        ]
        r = subprocess.run(task_run_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise RuntimeError(
                f"task env container start failed: {r.stderr.strip()}"
            )
        task_container_id = r.stdout.strip()

        # 4.6. Launch the exec-bridge dispatcher on the host. OC's patched
        #      runtime (see harnesses/openclaw/patch/openclaw-external-exec-bridge.patch)
        #      will hijack sandbox tool execs and route them through bridge.py
        #      inside the container → unix socket → this server → backend.
        #      The socket lives inside state_dir, which is already bind-mounted
        #      into OC at /root/.openclaw, so no extra mounts are needed.
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
        logger.info(
            "  exec-bridge: pid=%s sock=%s",
            dispatcher.pid, dispatcher.socket_path,
        )

        # 5. Start the openclaw container. Gateway runs here. OC's patched
        #    runtime sees OPENCLAW_EXEC_BRIDGE in env and re-routes every
        #    sandbox tool exec through bridge.py → unix socket → our
        #    host-side server. OC has no idea its execs aren't running
        #    locally — to it, agent commands "just work".
        #    We run as `--user 0` so the gateway can talk to docker.sock
        #    (host's docker.sock GID rarely matches `node`).
        gateway_port = _pick_free_gateway_port()
        gateway_token = secrets.token_hex(24)

        cmd = [
            "docker", "run", "-d",
            "--name", oc_name,
            "--network", net_name,
            "--add-host", "vllm-host:host-gateway",  # reach host vLLM
            "--user", "0",                            # docker.sock access
            # State dir mount — root's HOME is /root. Must be :rw so OpenClaw
            # can mkdir per-session sessions/, memory/, and write cache-trace
            # under agents/<id>/. The state_dir is per-run (rebuilt by
            # _build_state_dir on every prepare_environment) so there's no
            # cross-run leakage to defend against.
            "-v", f"{state_dir}:/root/.openclaw:rw",
            # Workspace bind kept for git/diff convenience inside oc; the
            # task container also sees the same dir at /testbed via OC's
            # sandbox.docker.binds.
            "-v", f"{ws_host}:/testbed:rw",
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "-w", "/testbed",
            "-e", f"OPENCLAW_GATEWAY_TOKEN={gateway_token}",
            "-e", f"OPENCLAW_GATEWAY_OTEL_ENDPOINT={os.environ.get('OC_OTEL_ENDPOINT', 'http://vllm-host:4318')}",
            "-e", f"OPENCLAW_RUN_ID={run_id}",
            "-e", f"OTEL_RESOURCE_ATTRIBUTES=oc.run_id={run_id}",
            # OC image ships docker CLI 20.10 (API 1.41). Modern docker daemons
            # (≥28.0) refuse anything below 1.44. Negotiate up; inspect/run/etc.
            # are wire-compatible across this range.
            "-e", "DOCKER_API_VERSION=1.44",
            # Bridge mount + OPENCLAW_EXEC_BRIDGE_* env. When this trio is
            # present, OC's patched spawnSpec returns the bridge argv
            # instead of `["docker", "exec", ...]`, sending each exec
            # through our host-side dispatcher over the socket.
            *dispatcher.agent_container_args(),
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

        # 6. Wait for gateway readiness.
        try:
            self._wait_for_gateway_ready(oc_container_id)
        except Exception:
            _release_gateway_port(gateway_port)
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise

        # 7. Drop a /tmp/oc_gateway_env so condition's docker-exec invocations
        #    can `source` it before `openclaw agent ...`.
        gw_env = (
            f"export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:18789\n"
            f"export OPENCLAW_GATEWAY_TOKEN={gateway_token}\n"
            f"export OPENCLAW_RUN_ID={run_id}\n"
        )
        self._exec(oc_container_id,
                   ["bash", "-c", f"cat > /tmp/oc_gateway_env <<'EOF'\n{gw_env}EOF"])

        logger.info(
            "  ready: oc=%s task=%s bridge_pid=%s net=%s",
            oc_container_id[:12], task_container_id[:12], dispatcher.pid, net_name,
        )
        return {
            "container_id": oc_container_id,
            "container_name": oc_name,
            "backend": "docker",
            "oc_container_id": oc_container_id,
            "task_container_id": task_container_id,
            "task_container_name": task_container_name,
            "network_name": net_name,
            "workspace_root": "/testbed",
            "workspace_host_path": str(ws_host),
            "state_dir": str(state_dir),
            "gateway_port": gateway_port,
            "gateway_token": gateway_token,
            "gateway_url": "ws://127.0.0.1:18789",
            "sandbox_image": container_image,
            **dispatcher.env_info(),
        }

    # ------------------------------------------------------------------
    # State-dir compose: config/harness/<name>.json `openclaw` block +
    # applications/swebench/openclaw/{agents,skills}
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
        os.makedirs(state_dir / "agents/swebench", exist_ok=True)

        override = os.environ.get("OC_HOST_STATE_TEMPLATE")
        if override:
            if not os.path.isdir(override):
                raise FileNotFoundError(
                    f"OC_HOST_STATE_TEMPLATE points to {override} but it is not a directory."
                )
            shutil.rmtree(state_dir)
            shutil.copytree(override, state_dir)
        else:
            # Compose the per-run openclaw.json from the active harness
            # config's `openclaw` block. The driver injects OC_HARNESS_YAML
            # (kept name for back-compat with shell scripts); for standalone
            # runs we fall back to the default 27B harness.
            harness_path = os.environ.get(
                "OC_HARNESS_YAML",
                os.path.join(kit_root, "config/harness/openclaw_27b.json"),
            )
            if not os.path.isfile(harness_path):
                raise FileNotFoundError(
                    f"harness config not found at {harness_path} "
                    "(set OC_HARNESS_YAML or OC_HOST_STATE_TEMPLATE)."
                )
            with open(harness_path) as fh:
                harness_cfg = json.load(fh) or {}
            oc_cfg = dict(harness_cfg.get("openclaw") or {})
            if not oc_cfg:
                raise ValueError(
                    f"harness config {harness_path} has no `openclaw` block "
                    "(required for the openclaw harness)."
                )
            # Auto-derive provider's `models[]` and agent's `model.primary`
            # from the harness config's top-level model.name so they can't
            # drift from what vLLM serves.
            model_name = (harness_cfg.get("model") or {}).get("name")
            if model_name:
                providers = oc_cfg.setdefault("models", {}).setdefault("providers", {})
                vllm_p = providers.setdefault("vllm", {})
                vllm_p["models"] = [{"id": model_name, "name": model_name}]
                agents_d = oc_cfg.setdefault("agents", {}).setdefault("defaults", {})
                agents_d.setdefault("model", {})["primary"] = f"vllm/{model_name}"
            with open(state_dir / "openclaw.json", "w") as fh:
                json.dump(oc_cfg, fh, indent=2)

            app_agents = os.path.join(kit_root, "applications/swebench/openclaw/agents")
            app_skills = os.path.join(kit_root, "applications/swebench/openclaw/skills")
            if not os.path.isdir(app_agents):
                raise FileNotFoundError(
                    f"per-app agents dir not found at {app_agents}"
                )
            for f in os.listdir(app_agents):
                src = os.path.join(app_agents, f)
                if os.path.isfile(src):
                    shutil.copyfile(src, state_dir / "agents/swebench" / f)
            if os.path.isdir(app_skills):
                shutil.copytree(app_skills, state_dir / "skills")

        # Per-run rewrites: workspace, sandbox, exec.host, skills extraDirs.
        cfg_path = state_dir / "openclaw.json"
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        agents_defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
        agents_defaults["workspace"] = workspace_root
        # Task container is pre-started by THIS harness with the exact name
        # OC will compute (containerPrefix + slugify(scopeKey)). OC's
        # `ensureSandboxContainer` finds our running container, treats the
        # configHash mismatch as "hot container" and uses ours instead of
        # spawning a new one. We keep the same `image`/`binds` here for the
        # registry/log paths, but only the prefix + scope actually drive
        # the name lookup.
        agents_defaults["sandbox"] = {
            "mode": "all",
            "scope": scope,
            "backend": "docker",
            "workspaceAccess": "rw",
            "docker": {
                "image": sandbox_image,
                "containerPrefix": container_prefix,
                # SWE-bench images mount the repo at /testbed; OC's default
                # workdir of /workspace doesn't exist in those images. The
                # bridge runtime reads `opts.sandbox.containerWorkdir` and
                # forwards it as --workdir to bridge.py, so this also feeds
                # the docker-backed exec invocation on the host.
                "workdir": workspace_root,
                "binds": list(sandbox_binds),
                "dangerouslyAllowExternalBindSources": True,
            },
        }
        # exec.host=sandbox routes the agent's exec tool INTO the task
        # container (the one we pre-started). "gateway"/"node" would exec
        # in OC's own Debian shell with no conda/astropy — wrong env.
        cfg.setdefault("tools", {}).setdefault("exec", {})["host"] = "sandbox"
        # Skills mount inside the oc container at /root/.openclaw/skills
        # (we run as --user 0 for docker socket access; HOME=/root).
        if (state_dir / "skills").is_dir():
            cfg.setdefault("skills", {}).setdefault("load", {})["extraDirs"] = [
                "/root/.openclaw/skills"
            ]
        # vLLM lives on the host. Inside the oc container, the host is reachable
        # at the docker `host-gateway` synthetic hostname we add via --add-host.
        # Harness yamls ship with `baseUrl: http://localhost:8000/v1` (matches
        # what the user sees on the host); rewrite per-run to the in-container
        # form so OC actually reaches the host's vLLM.
        vllm_endpoint = os.environ.get(
            "OC_VLLM_BASE_URL", "http://vllm-host:8000/v1"
        )
        providers = cfg.setdefault("models", {}).setdefault("providers", {})
        if "vllm" in providers:
            providers["vllm"]["baseUrl"] = vllm_endpoint
        # OTel collector lives on the host too — same rewrite story as vLLM.
        # The harness yaml's `diagnostics.otel.endpoint=http://localhost:4318`
        # is the host-perspective form; inside the OC container we need the
        # `host-gateway` alias so OTLP exports actually reach the collector.
        otel_cfg = (cfg.get("diagnostics") or {}).get("otel") or {}
        if otel_cfg.get("enabled"):
            cfg["diagnostics"]["otel"]["endpoint"] = os.environ.get(
                "OC_OTEL_ENDPOINT", "http://vllm-host:4318",
            )
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return state_dir

    # ------------------------------------------------------------------
    # Workspace priming
    # ------------------------------------------------------------------

    def _prime_workspace_from_image(self, container_image: str, ws_host: Path) -> None:
        """Copy /testbed out of the task image into ws_host, so the
        bind-mount overlay isn't empty when the real task container starts."""
        scratch_name = f"oc-eval-prime-{ws_host.name[:12]}-{os.urandom(2).hex()}"
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
                raise RuntimeError(f"docker cp from {container_image}:/testbed failed: {r.stderr.strip()}")
        finally:
            subprocess.run(["docker", "rm", "-f", scratch_name],
                           capture_output=True, check=False)

    # ------------------------------------------------------------------
    # Gateway readiness
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
            # Container died?
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
    # Container exec helpers
    # ------------------------------------------------------------------

    def _exec(
        self,
        container_id: str,
        argv: List[str],
        check: bool = True,
        timeout: Optional[int] = None,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        cmd = ["docker", "exec", container_id] + argv
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, input=input_text
        )
        if check and r.returncode != 0:
            raise RuntimeError(
                f"docker exec failed ({argv[0]}): rc={r.returncode}\n"
                f"stderr: {r.stderr[-500:]}"
            )
        return r

    # ------------------------------------------------------------------
    # Patch application + collection
    # ------------------------------------------------------------------

    def apply_patch(
        self,
        container_id: str,
        patch_path: Path,
        workspace_root: str,
    ) -> bool:
        """Apply a patch file via the task container (or any container that
        has /testbed bind-mounted to the same host workspace)."""
        logger.info("Applying patch %s in %s", patch_path, container_id[:12])
        target = f"{workspace_root}/oc_candidate.patch"
        r = subprocess.run(
            ["docker", "cp", str(patch_path), f"{container_id}:{target}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.error("docker cp failed: %s", r.stderr)
            return False
        dry = self._exec(
            container_id,
            ["git", "-C", workspace_root, "apply", "--check", target],
            check=False,
        )
        if dry.returncode != 0:
            logger.warning("Patch does not apply: %s", dry.stderr[-300:])
            return False
        self._exec(container_id, ["git", "-C", workspace_root, "apply", target])
        return True

    # OpenClaw scaffolding files agent might write into the workspace; strip
    # before diffing so they don't bleed into the SWE-bench patch.
    _OC_SCAFFOLD_FILES = (
        "AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md "
        "BOOT.md MEMORY.md fix.patch _oc_validate.sh"
    )
    _OC_SCAFFOLD_DIRS = ".openclaw memory sessions"

    def collect_patch(self, container_id: str, workspace_root: str) -> str:
        ws = workspace_root
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {self._OC_SCAFFOLD_FILES} && "
             f"rm -rf {self._OC_SCAFFOLD_DIRS}"],
            check=False,
        )
        self._exec(container_id, ["git", "-C", ws, "add", "-A"])
        r = self._exec(container_id, ["git", "-C", ws, "diff", "--cached"])
        return r.stdout

    # ------------------------------------------------------------------
    # Scoring via SWE-bench's run_evaluation
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
                    "status": "ERROR",
                    "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": "run_tests called with patch=None and no container_id",
                    "evaluation_dir": None,
                }
            patch = self.collect_patch(container_id, workspace_root)

        if not patch.strip():
            return {
                "status": "FAIL",
                "test_pass_rate": 0.0,
                "passed": 0, "failed": 1,
                "error_message": "Empty patch (agent made no changes)",
                "evaluation_dir": None,
            }

        if run_id is None:
            run_id = f"oc-eval-{instance_id}-{os.urandom(3).hex()}"

        with tempfile.TemporaryDirectory(prefix="oc_pred_") as tmp:
            tmp = Path(tmp)
            pred_path = tmp / "predictions.json"
            pred_path.write_text(json.dumps([{
                "instance_id": instance_id,
                "model_name_or_path": "oc-eval",
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
                    "status": "TIMEOUT",
                    "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": "run_evaluation wall-clock timeout",
                    "evaluation_dir": None,
                }
            # `run_evaluation` sometimes returns non-zero AFTER successfully
            # grading the instance — e.g. its container teardown races with
            # a parallel `docker rm` and emits docker.errors.NotFound, then
            # the harness exits 1 even though report.json was already written
            # with resolved=true. So always look for the verdict file first.
            verdict = self._find_and_parse_verdict(run_id, instance_id)
            if verdict.get("status") in ("PASS", "FAIL"):
                # Verdict file present → trust it. Log non-zero exits as a
                # warning so they don't get silently swept under the rug.
                if proc.returncode != 0:
                    logger.warning(
                        "run_evaluation exit %d but report.json present (status=%s) — "
                        "trusting report. stderr tail:\n%s",
                        proc.returncode, verdict.get("status"),
                        proc.stderr[-1500:],
                    )
                return verdict
            # No verdict file → genuine failure.
            if proc.returncode != 0:
                logger.error("run_evaluation stderr:\n%s", proc.stderr[-2000:])
                return {
                    "status": "ERROR",
                    "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": f"run_evaluation exit {proc.returncode}: {proc.stderr[-500:]}",
                    "evaluation_dir": None,
                }
            return verdict

    def _find_and_parse_verdict(self, run_id: str, instance_id: str) -> Dict:
        primary = (
            self.swebench_root
            / "logs" / "run_evaluation" / run_id / "oc-eval" / instance_id
            / "report.json"
        )
        if primary.exists():
            return self._parse_verdict(primary, instance_id)
        legacy = (
            self.swebench_root
            / "evaluation_results" / run_id / instance_id / "report.json"
        )
        if legacy.exists():
            return self._parse_verdict(legacy, instance_id)
        for root in (
            self.swebench_root / "logs" / "run_evaluation" / run_id,
            self.swebench_root / "evaluation_results" / run_id,
        ):
            if root.exists():
                matches = list(root.rglob("report.json"))
                if matches:
                    return self._parse_verdict(matches[0], instance_id)
        return {
            "status": "ERROR",
            "test_pass_rate": 0.0,
            "passed": 0, "failed": 0,
            "error_message": f"report.json not found for {instance_id} (run_id={run_id})",
            "evaluation_dir": str(primary.parent),
        }

    def _parse_verdict(self, report_path: Path, instance_id: str) -> Dict:
        try:
            report = json.loads(report_path.read_text())
        except Exception as e:
            return {
                "status": "ERROR",
                "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": f"Failed to parse {report_path}: {e}",
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

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(
        self,
        container_id: Optional[str] = None,
        oc_container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
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
        # Bridge server is just a Python subprocess on the host; SIGTERM is
        # enough (server.py installs a SIGTERM handler that os._exit(0)s
        # after unlinking the socket).
        OpenClawRemoteDispatcher.stop_external(
            pid=bridge_pid, socket_path=bridge_socket,
        )

        # Chown state_dir contents back to host UID from inside the still-
        # alive OC container — OC runs as root and writes per-session files
        # owned by root; without chown the host-side rmtree below hits EACCES.
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
        # Tear down ALL containers we created (OC + task env). Both names
        # are deterministic — no `--filter` scan needed. This is the
        # invariant that guarantees "previous run cannot affect next run":
        # cleanup always rms BOTH containers it brought up.
        targets = [t for t in (oc_container_id, container_id,
                               task_container_id, task_container_name)
                   if t]
        for cid in dict.fromkeys(targets):  # de-dupe preserving order
            subprocess.run(
                ["docker", "rm", "-f", "-v", cid],
                capture_output=True, check=False,
            )
        # Sweep the per-run network (in case the task container or OC left
        # anything attached) and remove it.
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
        self,
        net: str,
        oc: str,
        task: Optional[str] = None,
    ) -> None:
        for name in (oc, task):
            if name:
                subprocess.run(["docker", "rm", "-f", "-v", name],
                               capture_output=True, check=False)
        # Sweep anything left dangling on this network, then drop the network.
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
# SingularityHarness — placeholder for the two-instance singularity model.
# Will mirror SWEBenchHarness with `singularity instance start` for both
# the openclaw .sif (built from agent-bench/openclaw image) and the per-task
# .sif (built from the SWE-bench instance image).
# ----------------------------------------------------------------------


class SingularityHarness:
    backend_id = "singularity"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SingularityHarness is being rewritten for the two-instance model "
            "(openclaw sif + task sif). Use HARNESS_BACKEND=docker until then."
        )
