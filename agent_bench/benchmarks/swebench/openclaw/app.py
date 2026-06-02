"""SWE-bench application — OpenClaw harness.

Sibling of `agent_bench/benchmarks/swebench/openhands/app.py`. Implements the same
lifecycle contract (`prepare_environment`, `run_tests`, `cleanup`,
`collect_patch`, ...) so the driver treats both harness classes
interchangeably — the choice is made per-sweep from the harness JSON.

Layered model (no host-mode anywhere):

    host  ──── only vLLM/sglang runs here ────  http://localhost:8000/v1
                                   ▲
                                   │ OpenAI-compat HTTP
                                   │  (docker: vllm-host:8000 via --add-host)
                                   │  (singularity: localhost:8000, shared net)
    ┌──────────────────────────────────────┐
    │ oc-<rid>  agent-bench/openclaw:<v>   │
    │ - openclaw gateway (long-running)    │
    │ - openclaw agent CLI (per session)   │
    └────────┬─────────────────────────────┘
             │  agent tool.exec is hijacked by the OC patches and
             │  routed through the exec-bridge unix socket back to
             │  the host, where server.py docker-execs or
             │  singularity-execs into the task container:
             ↓
    ┌──────────────────────────────────────┐
    │ task-<rid>  swebench/sweb.eval.<inst>│
    │ - per-instance python env + pytest   │
    │ - workspace bind-mounted at /testbed │
    └──────────────────────────────────────┘

The same `SWEBenchHarness` drives both docker and singularity flows — it
talks to a runtime-agnostic `Container` injected by the driver. See
`agent_bench/deployments/containers/` for the two implementations.

Scoring (`run_tests`) is independent of the sandbox — it shells out to
`python -m swebench.harness.run_evaluation` on the host; that command
spawns its own scoring containers and writes report.json.
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
import threading
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
from agent_bench.benchmarks.swebench.scoring import score_via_scoring_server, score_via_singularity
from agent_bench.deployments.containers import (
    Container,
    ContainerHandle,
    ContainerSpec,
    Mount,
)
from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher
from agent_bench.core.config import endpoint_for_container, load_active_model_and_serving
from agent_bench.core.profile_io import load_profile

logger = logging.getLogger(__name__)


# Gateway warmup is serialized across concurrent runs. OC npm-installs ~11
# bundled plugins *in parallel* between container start and "[gateway] ready".
# In singularity mode every run shares the host net + filesystems, so N runs
# warming up at once fire ~11*N parallel npm installs that race on shared
# transitive deps (e.g. typebox) and crash with ENOTEMPTY. Limiting how many
# gateways warm up at a time (default 1 = one-by-one) keeps each warmup at
# single-run load, which is reliable; the agent loops still run fully
# concurrent afterward, and the agent CLI reuses the gateway's now-complete
# plugin-runtime-deps (same instance) so it won't re-install/race.
_OPENCLAW_WARMUP_SEM = threading.BoundedSemaphore(
    int(os.environ.get("AB_HARNESS_WARMUP_CONCURRENCY", "1"))
)


# Static SWE-bench agent prompt filenames mounted from
# agent_bench/benchmarks/swebench/openclaw/agents/.
_SWEBENCH_AGENT_PROMPTS = (
    "AGENTS.md", "SOUL.md", "IDENTITY.md", "USER.md", "TOOLS.md", "HEARTBEAT.md",
)

_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"


class SWEBenchHarness:
    """Two-container OC harness, runtime-agnostic.

    `prepare_environment` brings up: (optionally) a per-run docker network,
    a task container (the SWE-bench instance image holding /testbed and its
    python env), and an openclaw container (the patched OC runtime image
    holding the gateway + agent CLI). They share the workspace via
    bind-mount; the agent's `tools.exec` is hijacked by the OC patches and
    routed through the host-side exec-bridge into the task container — the
    `Container` injected by the driver picks docker or singularity for the
    actual `exec` call.
    """

    backend_id = "container-agnostic"

    # OpenClaw 2026.4+ logs `[gateway] listening on ws://...` at readiness;
    # earlier versions said `[gateway] ... ready`.
    _GATEWAY_READY_RE = re.compile(
        r"\[gateway\][^\n]{0,80}(ready|listening on )", re.IGNORECASE,
    )
    _GATEWAY_READY_TIMEOUT_SECONDS = 600

    def __init__(
        self,
        benchmark_data_root: Path,
        *,
        container: Container,
        oc_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ):
        self.benchmark_data_root = Path(benchmark_data_root)
        self.container = container
        # `oc_image` is the OC runtime container reference. docker: image tag
        # (e.g. `agent-bench/openclaw:2026.4.22`). singularity: absolute path
        # to oc_runtime.sif. Driver/PBS passes the right one via env when
        # building this harness.
        self.oc_image = (
            oc_image
            or os.environ.get("AB_AGENT_RUNTIME_SIF")
            or DEFAULT_OPENCLAW_RUNTIME_IMAGE
        )

        # Pre-flight: swebench python pkg + container runtime + OC image.
        try:
            import swebench  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "swebench python package not importable; "
                "install in the driver env: pip install swebench"
            ) from e
        self.container.check_available()
        self.container.check_image(self.oc_image)

        self.workspace_base = Path(
            workspace_base
            or os.environ.get("AB_HARNESS_WORKSPACE_BASE",
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

        # The task container image. docker: the resolved docker tag.
        # singularity: an absolute path to the per-instance .sif. The harness
        # asks the active Container to translate.
        task_image = self._resolve_task_image(container_image)

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
            "Preparing run %s [runtime=%s] (sandbox image=%s, oc=%s, repo=%s@%s, task=%s)",
            run_id, self.container.backend_id, container_image, self.oc_image,
            repo_id, base_commit[:12], task_container_name,
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

        # 3. Per-run docker network (no-op for singularity).
        self.container.create_network(net_name)

        # 4. Prime the host workspace from the task image — required because
        #    bind-mounting an empty <ws_host>:/testbed would mask the image's
        #    pre-checked-out repo and the agent would see /testbed as empty.
        try:
            self._prime_workspace_from_image(task_image, ws_host)
        except Exception:
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise

        # 4.5. Start the SWE-bench task env container ourselves with the
        #      name OC will look for. We mount <ws_host> at /testbed:rw so
        #      the agent's edits land on the host (for patch extraction)
        #      and the container stays alive for exec from the bridge.
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
        task_spec = ContainerSpec(
            name=task_container_name,
            image=task_image,
            network=net_name,
            user="0",
            labels={
                "openclaw.sandbox": "1",
                "openclaw.sessionKey": scope_key,
                "oc-eval.run_id": run_id,
            },
            mounts=(Mount(host=ws_host, container="/testbed", mode="rw"),),
            workdir="/testbed",
            env={"PATH": task_path, "HOME": "/root"},
            entrypoint="sleep",
            command=("infinity",),
        )
        try:
            task_handle = self.container.start(task_spec)
        except Exception:
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise
        task_container_id = task_handle.id

        # 4.6. Launch the exec-bridge dispatcher on the host. OC's patched
        #      runtime hijacks sandbox tool execs and routes them through
        #      bridge.py inside the container → unix socket → this server →
        #      Container backend (docker exec or singularity exec).
        dispatcher = OpenClawRemoteDispatcher()
        try:
            dispatcher.start(
                state_dir=state_dir,
                backend=self.container.backend_id,
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
        #    sandbox tool exec through the bridge → unix socket → our
        #    host-side server. OC has no idea its execs aren't running
        #    locally — to it, agent commands "just work".
        gateway_port = _pick_free_gateway_port()
        gateway_token = secrets.token_hex(24)

        oc_mounts: list[Mount] = [
            # State dir mount — root's HOME is /root. Must be :rw so OpenClaw
            # can mkdir per-session sessions/, memory/, and write cache-trace
            # under agents/<id>/. Per-run state_dir (rebuilt by
            # _build_state_dir on every prepare_environment).
            Mount(host=state_dir, container="/root/.openclaw", mode="rw"),
            # Workspace bind for git/diff convenience inside oc; task
            # container sees the same dir at /testbed via OC sandbox binds.
            Mount(host=ws_host, container="/testbed", mode="rw"),
        ]
        # docker.sock — only useful in docker mode where OC's docker plugin
        # may still try to ping the daemon at init. Singularity has nothing
        # to mount here.
        if self.container.backend_id == "docker":
            oc_mounts.append(Mount(
                host=Path("/var/run/docker.sock"),
                container="/var/run/docker.sock",
                mode="rw",
            ))

        oc_env = {
            # State dir is bind-mounted at /root/.openclaw; OC reads its
            # config from $HOME/.openclaw. docker runs the container as
            # --user 0 so HOME is /root by default; singularity passes the
            # host user's HOME (e.g. /home/users/ntu/$USER), which would
            # send OC looking for state in the wrong place. Pin HOME=/root
            # so both runtimes land on the same path.
            "HOME": "/root",
            # Bind the gateway to its per-run port (reserved by
            # pick_free_gateway_port; the browser sidecar takes port+2, also
            # reserved). Docker gives each container its own net namespace, so
            # the default 18789 never collides there. Singularity shares the
            # host net namespace — concurrent runs MUST each bind a distinct
            # host port, otherwise all but the first gateway die with
            # EADDRINUSE on 18789 and their agents never start.
            "OPENCLAW_GATEWAY_PORT": str(gateway_port),
            "OPENCLAW_GATEWAY_TOKEN": gateway_token,
            "OPENCLAW_GATEWAY_OTEL_ENDPOINT": os.environ.get(
                "AB_TELEMETRY_ENDPOINT",
                f"http://{self.container.host_address}:4318",
            ),
            "OPENCLAW_RUN_ID": run_id,
            # pi-ai openai-completions patch tees every LLM HTTP response
            # into this file; collected back via the state_dir mount.
            "OPENCLAW_RAW_RESPONSE_LOG": "/root/.openclaw/logs/llm-raw-responses.jsonl",
            "OTEL_RESOURCE_ATTRIBUTES": f"oc.run_id={run_id}",
            # OC image ships docker CLI 20.10 (API 1.41). Modern daemons
            # (≥28.0) refuse anything below 1.44. Harmless in singularity.
            "DOCKER_API_VERSION": "1.44",
            # NSCC blocks egress to registry.npmjs.org. OC plugins (e.g.
            # browser, mcp-*) `npm install` runtime deps on first use; without
            # a reachable registry they hang ~10min retrying. Route npm at the
            # Alibaba mirror (registry.npmmirror.com — current Taobao successor)
            # which IS reachable from NSCC, plus fast-fail knobs as backstop.
            "NPM_CONFIG_REGISTRY": "https://registry.npmmirror.com",
            "NPM_CONFIG_FETCH_RETRIES": "1",
            "NPM_CONFIG_FETCH_TIMEOUT": "15000",
            "NODE_OPTIONS": "--dns-result-order=ipv4first",
            # OC's bonjour/mDNS advertiser tries to announce a service via
            # UDP multicast on the host net interface. In docker mode that
            # multicast stays inside the container's own net namespace;
            # in singularity mode the container shares the host net, so
            # the multicast hits the actual NIC and NSCC's compute-node
            # firewall drops it. The advertiser's `advertise()` await then
            # never returns → gateway stays in "starting channels and
            # sidecars" forever and never enters the HTTP accept loop.
            # OC reads OPENCLAW_DISABLE_BONJOUR=1 as a hard kill-switch
            # (dist/server.impl-*.js: `bonjourEnabled = mdnsMode!=="off" &&
            # process.env.OPENCLAW_DISABLE_BONJOUR!=="1" && !VITEST`).
            "OPENCLAW_DISABLE_BONJOUR": "1",
        }
        # Bridge wiring (mount + env). Runtime-agnostic.
        b_mounts, b_env = dispatcher.agent_mounts_and_env()
        oc_mounts.extend(b_mounts)
        oc_env.update(b_env)
        # In singularity mode there's no docker daemon to inspect / talk to,
        # but OC's stock sandbox.backend=docker factory calls dockerImageExists
        # / dockerContainerState / docker run during lane init. The OC patch
        # `openclaw-ensure-sandbox-bridge-bypass` short-circuits
        # ensureSandboxContainer when this env is set: returns the precomputed
        # containerName (we pre-started the task sif under exactly that name)
        # without touching docker. Tool exec still flows through the
        # OPENCLAW_EXEC_BRIDGE patch into bridge.py → singularity exec.
        if self.container.backend_id == "singularity":
            oc_env["OPENCLAW_BYPASS_ENSURE_SANDBOX"] = "1"
            # Plugin runtime-deps install defaults to <state_dir>/
            # plugin-runtime-deps, i.e. /root/.openclaw/... which singularity
            # mode backs with networked scratch. The install's final step
            # (bundled-runtime-deps.ts replaceNodeModulesDir) does
            # rmSync(node_modules) → rename. On a networked FS, rmSync of a
            # tree whose files are still held open by the already-running
            # gateway triggers NFS silly-rename (.nfs* placeholders left
            # behind) → rmdir sees a non-empty dir → ENOTEMPTY → the agent
            # CLI's plugin load throws → agent never starts (no session log,
            # "No patch generated"). Non-deterministic: it only fires when the
            # gateway still holds those files, which concurrent-run I/O load
            # makes more likely. Redirect the whole install to a node-local,
            # per-run /tmp dir (OPENCLAW_PLUGIN_STAGE_DIR, read in
            # bundled-runtime-deps.ts:resolveBundledRuntimeDepsExternalBaseDir):
            # local FS unlinks open files cleanly, so rmSync/rename can't
            # ENOTEMPTY. Per-run subdir keeps concurrent runs from colliding on
            # the shared host /tmp.
            oc_env["OPENCLAW_PLUGIN_STAGE_DIR"] = f"/tmp/oc-plugin-deps/{run_id}"

        oc_spec = ContainerSpec(
            name=oc_name,
            image=self.oc_image,
            network=net_name,
            extra_hosts={"vllm-host": "host-gateway"},  # docker only
            user="0",                                   # docker only
            mounts=tuple(oc_mounts),
            workdir="/testbed",
            env=oc_env,
            entrypoint="/usr/local/bin/openclaw",
            command=("gateway", "--allow-unconfigured"),
        )
        # Serialize the gateway warmup window (start → plugin npm install →
        # "[gateway] ready") via _OPENCLAW_WARMUP_SEM so concurrent runs don't fire
        # ~11*N parallel npm installs that ENOTEMPTY-race. See the semaphore
        # definition above. Agent loops run concurrently outside this block.
        logger.info("  [warmup] waiting for gateway-warmup slot (run=%s)", run_id)
        with _OPENCLAW_WARMUP_SEM:
            logger.info("  [warmup] slot acquired; starting gateway (run=%s)", run_id)
            try:
                oc_handle = self.container.start(oc_spec)
            except Exception:
                _release_gateway_port(gateway_port)
                dispatcher.stop()
                self._best_effort_cleanup(net_name, oc_name, task_container_name)
                raise
            oc_container_id = oc_handle.id

            # 6. Wait for gateway readiness (plugin install completes here).
            try:
                self._wait_for_gateway_ready(oc_handle)
            except Exception:
                _release_gateway_port(gateway_port)
                dispatcher.stop()
                self._best_effort_cleanup(net_name, oc_name, task_container_name)
                raise
        logger.info("  [warmup] gateway ready; slot released (run=%s)", run_id)

        # 7. Drop a /tmp/oc_gateway_env so the condition layer can `source`
        #    it before `openclaw agent ...`.
        gw_env = (
            f"export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:{gateway_port}\n"
            f"export OPENCLAW_GATEWAY_TOKEN={gateway_token}\n"
            f"export OPENCLAW_RUN_ID={run_id}\n"
        )
        if self.container.backend_id == "singularity":
            # Give the agent CLI its OWN plugin stage dir, distinct from the
            # gateway's (oc_env sets the gateway's to
            # /tmp/oc-plugin-deps/<run_id>). Both processes mirror bundled
            # plugins via mirrorBundledPluginRuntimeRoot
            # (bundled-runtime-root.ts), which does an UNGUARDED
            # rmSync+renameSync of dist/extensions/<id> — no lock, no
            # skip-if-present. Sharing one stage dir lets the gateway (it
            # mirrors e.g. vllm — deps=0, so no npm log — when resolving the
            # provider on the first model call) and the CLI (its startup plugin
            # load) rename dist/extensions/vllm concurrently → "ENOTEMPTY:
            # rename ... -> .../vllm" → CLI plugin load throws → agent never
            # starts (no session log, "No patch generated"). Non-deterministic;
            # concurrent-run I/O load makes the overlap likelier. This env is
            # sourced from /tmp/oc_gateway_env right before `openclaw agent`
            # (conditions/openclaw_agent.py), so it overrides the gateway value
            # the singularity exec env-replay would otherwise carry in. Separate
            # per-process roots remove the shared path so the mirrors can't
            # collide.
            gw_env += (
                f"export OPENCLAW_PLUGIN_STAGE_DIR=/tmp/oc-plugin-deps/{run_id}-cli\n"
            )
        self.container.exec(
            oc_handle,
            ["bash", "-c", f"cat > /tmp/oc_gateway_env <<'EOF'\n{gw_env}EOF"],
        )

        logger.info(
            "  ready: oc=%s task=%s bridge_pid=%s net=%s",
            oc_container_id[:12], task_container_id[:12], dispatcher.pid, net_name,
        )
        return {
            "container_id": oc_container_id,
            "container_name": oc_name,
            "backend": self.container.backend_id,
            "oc_container_id": oc_container_id,
            "task_container_id": task_container_id,
            "task_container_name": task_container_name,
            "network_name": net_name,
            "workspace_root": "/testbed",
            "workspace_host_path": str(ws_host),
            "state_dir": str(state_dir),
            "gateway_port": gateway_port,
            "gateway_token": gateway_token,
            "gateway_url": f"ws://127.0.0.1:{gateway_port}",
            "sandbox_image": container_image,
            **dispatcher.env_info(),
        }

    # ------------------------------------------------------------------
    # Container image resolution
    # ------------------------------------------------------------------

    def _resolve_task_image(self, container_image: str) -> str:
        """Translate a docker tag like `swebench/sweb.eval.x86_64.<id>:latest`
        into the runtime's expected reference.

        docker: pass through unchanged.
        singularity: a sif filename under `$AB_SCORING_IMAGE_DIR`. The on-disk
        naming convention is `<instance_id>.sif` (see
        agent_bench/deployments/pbs/pull_images.sh:pull_one). We recover the
        instance_id from the docker tag (last `.` segment before `:`).
        """
        if self.container.backend_id == "docker":
            return container_image
        sif_dir = os.environ.get("AB_SCORING_IMAGE_DIR")
        if not sif_dir:
            raise RuntimeError(
                "AB_SCORING_IMAGE_DIR not set; the singularity backend needs "
                "the per-instance .sif directory (where pull_images.sh "
                "wrote the pulled SWE-bench task images)."
            )
        # `swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest`
        #  → tag = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
        #  → last segment after the final `.` = the instance_id slug
        #    (with `__` already turned into `_1776_` by docker tag encoding).
        # pull_images.sh names sifs as `<instance_id>.sif` using the original
        # instance_id (with `__`), so we have to reverse the encoding.
        tag = container_image.split(":")[0]
        slug = tag.split(".")[-1]                  # astropy_1776_astropy-12907
        instance_id = slug.replace("_1776_", "__") # astropy__astropy-12907
        sif_path = Path(sif_dir) / f"{instance_id}.sif"
        return str(sif_path)

    # ------------------------------------------------------------------
    # State-dir compose: profiles/harnesses/<name>.yaml `openclaw` block +
    # agent_bench/benchmarks/swebench/openclaw/{agents,skills}
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
        os.makedirs(state_dir / "agents/swebench", exist_ok=True)

        override = os.environ.get("AB_HARNESS_STATE_TEMPLATE")
        if override:
            if not os.path.isdir(override):
                raise FileNotFoundError(
                    f"AB_HARNESS_STATE_TEMPLATE points to {override} but it is not a directory."
                )
            shutil.rmtree(state_dir)
            shutil.copytree(override, state_dir)
        else:
            # Compose the per-run openclaw.json from the active harness
            # config's `openclaw` block. Model identity and endpoint come
            # from the separate model/serving axes.
            harness_path = os.environ.get(
                "AB_HARNESS_CONFIG",
                os.path.join(repo_root, "profiles/harnesses/openclaw_swebench.yaml"),
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
                    "(required for the openclaw harness)."
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

            app_agents = os.path.join(repo_root, "agent_bench/benchmarks/swebench/openclaw/agents")
            app_skills = os.path.join(repo_root, "agent_bench/benchmarks/swebench/openclaw/skills")
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
        # vLLM/sglang lives on the host. The container reaches the host via
        # `self.container.host_address` (docker: `vllm-host` synthetic
        # hostname; singularity: `127.0.0.1` shared net namespace). Harness
        # yamls ship with `baseUrl: http://localhost:8000/v1` (host
        # perspective); rewrite per-run.
        model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
        # The serving config is static, but PBS may pick a per-job dynamic
        # LiteLLM port and publish it through OPENAI_BASE_URL/AB_SERVING_BASE_URL.
        # Use those runtime values so OpenClaw follows the same endpoint as
        # OpenHands/Hermes.
        runtime_endpoint = (
            os.environ.get("AB_SERVING_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("AB_SERVING_BASE_URL")
            or model_cfg.endpoint
        )
        serving_endpoint = endpoint_for_container(
            runtime_endpoint,
            self.container.host_address,
        )
        providers = cfg.setdefault("models", {}).setdefault("providers", {})
        agent_model_provider = providers.setdefault("agent_model", {})
        agent_model_provider["api"] = _OPENAI_CHAT_COMPLETIONS_API
        agent_model_provider["baseUrl"] = serving_endpoint
        for model in agent_model_provider.get("models") or []:
            if isinstance(model, dict):
                model["api"] = _OPENAI_CHAT_COMPLETIONS_API
        # API-backed providers (e.g. DeepSeek) need a real key; inject it
        # from env so the secret isn't committed in the harness config.
        # sglang/local leaves AB_SERVING_API_KEY unset and keeps the dummy key.
        _api_key = os.environ.get("AB_SERVING_API_KEY")
        if _api_key:
            agent_model_provider["apiKey"] = _api_key
        # OTel collector lives on the host too — same rewrite story.
        otel_cfg = (cfg.get("diagnostics") or {}).get("otel") or {}
        if otel_cfg.get("enabled"):
            cfg["diagnostics"]["otel"]["endpoint"] = os.environ.get(
                "AB_TELEMETRY_ENDPOINT", f"http://{self.container.host_address}:4318",
            )
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return state_dir

    # ------------------------------------------------------------------
    # Workspace priming
    # ------------------------------------------------------------------

    def _prime_workspace_from_image(self, task_image: str, ws_host: Path) -> None:
        """Copy /testbed out of the task image into ws_host, so the
        bind-mount overlay isn't empty when the real task container starts."""
        self.container.copy_from_image(task_image, "/testbed", ws_host)

    # ------------------------------------------------------------------
    # Gateway readiness
    # ------------------------------------------------------------------

    def _wait_for_gateway_ready(self, oc_handle: ContainerHandle) -> None:
        self.container.wait_log_pattern(
            oc_handle,
            self._GATEWAY_READY_RE,
            timeout_s=self._GATEWAY_READY_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # Container exec helpers
    # ------------------------------------------------------------------

    def _exec(
        self,
        container_handle_or_id,
        argv: List[str],
        check: bool = True,
        timeout: Optional[int] = None,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Back-compat helper; new call sites should use self.container.exec.

        Accepts either a `ContainerHandle` (preferred) or a bare id (for the
        legacy condition-layer calls that only know the id string).
        """
        if isinstance(container_handle_or_id, ContainerHandle):
            h = container_handle_or_id
        else:
            h = ContainerHandle(
                id=str(container_handle_or_id),
                name=str(container_handle_or_id),
                backend=self.container.backend_id,
            )
        return self.container.exec(
            h, argv, check=check, timeout=timeout, input_text=input_text,
        )

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
        h = ContainerHandle(
            id=container_id, name=container_id, backend=self.container.backend_id,
        )
        try:
            self.container.copy_into(h, Path(patch_path), target)
        except Exception as e:
            logger.error("copy_into failed: %s", e)
            return False
        dry = self._exec(
            h,
            ["git", "-C", workspace_root, "apply", "--check", target],
            check=False,
        )
        if dry.returncode != 0:
            logger.warning("Patch does not apply: %s", dry.stderr[-300:])
            return False
        self._exec(h, ["git", "-C", workspace_root, "apply", target])
        return True

    # OpenClaw scaffolding files agent might write into the workspace; strip
    # before diffing so they don't bleed into the SWE-bench patch.
    _OPENCLAW_SCAFFOLD_FILES = (
        "AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md "
        "BOOT.md MEMORY.md fix.patch _oc_validate.sh"
    )
    _OPENCLAW_SCAFFOLD_DIRS = ".openclaw memory sessions"

    def collect_patch(self, container_id: str, workspace_root: str) -> str:
        ws = workspace_root
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {self._OPENCLAW_SCAFFOLD_FILES} && "
             f"rm -rf {self._OPENCLAW_SCAFFOLD_DIRS}"],
            check=False,
        )
        # SWE-bench's setup runs `git commit -am SWE-bench` after install
        # (see swebench/harness/test_spec/python.py), so HEAD is the
        # post-setup baseline. Diffing against HEAD captures every agent
        # edit to a tracked file and naturally excludes untracked artifacts
        # (build/, *.egg-info, __pycache__, .pytest_cache) that would
        # otherwise turn into giant "new file" hunks and trip the eval
        # container's patch -R heuristic.
        r = self._exec(container_id, ["git", "-C", ws, "diff", "HEAD"])
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
        instance_data: Optional[Dict] = None,
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

        singularity_verdict = score_via_singularity(
            instance_id=instance_id,
            patch=patch,
            model_name_or_path="oc-eval",
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
            model_name_or_path="oc-eval",
            timeout_seconds=timeout_seconds,
        )
        if remote_verdict is not None:
            return remote_verdict

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
                    cmd, cwd=str(self.benchmark_data_root),
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
            self.benchmark_data_root
            / "logs" / "run_evaluation" / run_id / "oc-eval" / instance_id
            / "report.json"
        )
        if primary.exists():
            return self._parse_verdict(primary, instance_id)
        legacy = (
            self.benchmark_data_root
            / "evaluation_results" / run_id / instance_id / "report.json"
        )
        if legacy.exists():
            return self._parse_verdict(legacy, instance_id)
        for root in (
            self.benchmark_data_root / "logs" / "run_evaluation" / run_id,
            self.benchmark_data_root / "evaluation_results" / run_id,
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
        # alive OC container — OC runs as root (in docker) and writes
        # per-session files owned by root; without chown the host-side
        # rmtree below hits EACCES. Singularity already runs as host UID,
        # so the chown is a no-op there but harmless.
        uid = os.getuid()
        gid = os.getgid()
        for cid in {oc_container_id, container_id}:
            if not cid:
                continue
            h = ContainerHandle(
                id=cid, name=cid, backend=self.container.backend_id,
            )
            try:
                self.container.exec(
                    h,
                    ["chown", "-R", f"{uid}:{gid}", "/root/.openclaw"],
                    check=False, timeout=30,
                )
            except Exception:
                pass

        # Tear down ALL containers we created (OC + task env). Both names
        # are deterministic — no `--filter` scan needed. This is the
        # invariant that guarantees "previous run cannot affect next run":
        # cleanup always rms BOTH containers it brought up.
        targets = [t for t in (oc_container_id, container_id,
                               task_container_id, task_container_name)
                   if t]
        for cid in dict.fromkeys(targets):  # de-dupe preserving order
            h = ContainerHandle(
                id=cid, name=cid, backend=self.container.backend_id,
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
        self,
        net: str,
        oc: str,
        task: Optional[str] = None,
    ) -> None:
        for name in (oc, task):
            if name:
                h = ContainerHandle(
                    id=name, name=name, backend=self.container.backend_id,
                )
                try:
                    self.container.stop(h)
                except Exception:
                    pass
        try:
            self.container.remove_network(net)
        except Exception:
            pass
