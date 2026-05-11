"""OpenClaw-side glue for SWE-bench.

Layer-2 helpers extracted from applications/swebench/app.py so the application
file (Layer 1) is no longer responsible for OpenClaw's host-package autodetect,
gateway port management, container mount construction, or the
singularity-exec wrapper. The classes in app.py call into this module.
"""
from __future__ import annotations

import logging
import os
import secrets
import shutil
import socket
import subprocess
import threading
from typing import Dict, List, Optional, Set, Tuple  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- pkg autodetect

def _autodetect_openclaw_pkg() -> str:
    """Find the openclaw package dir. Prefers a user-level npm install whose
    `dist/extensions/*/node_modules` are already populated — required because
    we mount the package read-only, so OpenClaw cannot npm-install plugin
    runtime deps at startup. Falls back to the system path with a clear error
    surface in the constructor below.
    """
    env = os.environ.get("OC_HOST_OPENCLAW")
    if env:
        return env
    bin_path = shutil.which("openclaw")
    if bin_path:
        real = os.path.realpath(bin_path)
        if os.path.basename(real) in ("openclaw.mjs", "openclaw"):
            cand = os.path.dirname(real)
            if os.path.exists(os.path.join(cand, "dist", "extensions")):
                return cand
        prefix = os.path.dirname(os.path.dirname(bin_path))
        cand = os.path.join(prefix, "lib", "node_modules", "openclaw")
        if os.path.isdir(cand):
            return cand
    return "/usr/local/lib/node_modules/openclaw"


HOST_OPENCLAW_PKG = _autodetect_openclaw_pkg()


# ---------------------------------------------------------------------- gateway port pool

# Per-container Gateway port range. Avoids 18789 (host's other gateway) and
# leaves headroom for sidecars (Gateway picks port+2 for the browser plugin
# control surface). Stride of 4 → up to ~25 concurrent runs.
GATEWAY_PORT_MIN = int(os.environ.get("OC_GATEWAY_PORT_MIN", "18800"))
GATEWAY_PORT_MAX = int(os.environ.get("OC_GATEWAY_PORT_MAX", "18899"))
GATEWAY_PORT_STRIDE = 4

# Process-global reservation — the bind probe releases immediately, so two
# concurrent threads can race on the same port. Hold a logical reservation
# from `pick_free_gateway_port` until `release_gateway_port`.
_GATEWAY_PORT_LOCK = threading.Lock()
_GATEWAY_PORTS_INUSE: Set[int] = set()


def pick_free_gateway_port(attempts: int = 25) -> int:
    """Reserve a free port in [GATEWAY_PORT_MIN, GATEWAY_PORT_MAX].

    Range stride accounts for Gateway opening a sidecar at port+2 (browser
    plugin control surface). Reservation is process-global so concurrent
    callers don't collide.
    """
    import random
    candidates = list(range(GATEWAY_PORT_MIN, GATEWAY_PORT_MAX + 1, GATEWAY_PORT_STRIDE))
    random.shuffle(candidates)
    with _GATEWAY_PORT_LOCK:
        for port in candidates[:attempts]:
            if port in _GATEWAY_PORTS_INUSE or (port + 2) in _GATEWAY_PORTS_INUSE:
                continue
            try:
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port))
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port + 2))
            except OSError:
                continue
            _GATEWAY_PORTS_INUSE.add(port)
            _GATEWAY_PORTS_INUSE.add(port + 2)
            return port
    raise RuntimeError(
        f"no free Gateway port in [{GATEWAY_PORT_MIN},{GATEWAY_PORT_MAX}]"
    )


def release_gateway_port(port: int) -> None:
    with _GATEWAY_PORT_LOCK:
        _GATEWAY_PORTS_INUSE.discard(port)
        _GATEWAY_PORTS_INUSE.discard(port + 2)


# ---------------------------------------------------------------------- mounts

# Static SWE-bench agent prompt filenames mounted from
# applications/swebench/openclaw/agents/ into /root/.openclaw/agents/swebench/.
SWEBENCH_AGENT_PROMPT_FILES = (
    "AGENTS.md", "SOUL.md", "IDENTITY.md", "USER.md", "TOOLS.md", "HEARTBEAT.md",
)


def compose_oc_mounts(
    *,
    host_node_bin: str,
    host_npm_pkg: str,
    oc_config_file: str,
    oc_agents_dir: str,
    oc_plugin_cache: str,
    oc_skills_dir: Optional[str] = None,
) -> List[str]:
    """Build the OpenClaw-related `-v` flags for `docker run`.

    Returns a flat list ready to splice into the docker argv. Caller supplies
    the resolved host paths; this function only knows the layout of the
    container side. If `oc_skills_dir` is given, the per-app skills directory
    is mounted at /root/.openclaw/skills (the path openclaw.json registers via
    skills.load.extraDirs).
    """
    mounts: List[str] = [
        "-v", f"{host_node_bin}:/usr/local/bin/node:ro",
        "-v", f"{host_npm_pkg}:/usr/local/lib/node_modules/npm:ro",
        "-v", f"{HOST_OPENCLAW_PKG}:/opt/openclaw:ro",
        "-v", f"{oc_config_file}:/root/.openclaw/openclaw.json:ro",
    ]
    # agents/ cannot be mounted wholesale: OpenClaw mkdirs sessions/,
    # memory/ etc. under the agent's dir at runtime. Mount only the
    # static prompt files, leaving the parent dir writable.
    for f in SWEBENCH_AGENT_PROMPT_FILES:
        mounts += ["-v", f"{oc_agents_dir}/{f}:/root/.openclaw/agents/swebench/{f}:ro"]
    # Per-app skills (e.g. the `verify` skill that wraps the validate bridge).
    # ro because OpenClaw only reads SKILL.md to inject into the system prompt.
    if oc_skills_dir:
        mounts += ["-v", f"{oc_skills_dir}:/root/.openclaw/skills:ro"]
    # Plugin runtime deps cache — shared across runs to avoid re-downloading
    # hundreds of MB of Node modules on every container start. First container
    # to start populates it; subsequent containers reuse. rw because OpenClaw
    # may touch lockfiles etc. even on reuse.
    mounts += ["-v", f"{oc_plugin_cache}:/root/.openclaw/plugin-runtime-deps:rw"]
    return mounts


# ---------------------------------------------------------------------- gateway lifecycle

def start_gateway_in_container(
    container_id: str,
    *,
    container_name: str,
    kit_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    """Pick a free port + token, then source harnesses/openclaw/gateway_up.sh
    inside the container. Returns gateway connection info:

        {"port": "<n>", "token": "<hex>", "url": "ws://127.0.0.1:<n>"}

    Raises FileNotFoundError if gateway_up.sh is missing, RuntimeError if it
    fails to start the gateway.
    """
    kit_root = kit_root or os.environ.get("KIT_ROOT", os.getcwd())
    gw_up = f"{kit_root}/harnesses/openclaw/gateway_up.sh"
    if not os.path.exists(gw_up):
        raise FileNotFoundError(f"missing helper: {gw_up}")

    port = pick_free_gateway_port()
    token = secrets.token_hex(24)
    url = f"ws://127.0.0.1:{port}"

    argv = ["bash", gw_up, container_id, str(port), token]
    if run_id:
        argv.append(run_id)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        # Caller is expected to release the port + tear down the container.
        release_gateway_port(port)
        logger.error("gateway-up failed: %s", r.stderr.strip())
        raise RuntimeError(
            f"openclaw_gateway_up.sh failed (rc={r.returncode}): {r.stderr.strip()}"
        )
    logger.info("  gateway ready at %s in container %s", url, container_name)
    return {"port": port, "token": token, "url": url}


# ---------------------------------------------------------------------- singularity

def wrap_singularity_argv(
    argv: List[str], env: Dict, binds: List, sif: str, overlay: Optional[str] = None,
) -> Tuple[List[str], Dict]:
    """Prefix `argv` with `singularity exec [--overlay <ov>:ro] --bind ... <sif>`.

    Caller-supplied `sif` and `overlay` (no env-var sniffing) so call sites
    that already know they're on the singularity backend can wrap explicitly.

    `binds` is a list of (src, dst) path pairs. Use src==dst when possible so
    absolute paths in env vars (OPENCLAW_STATE_DIR etc) resolve identically
    inside the container.

    OPENCLAW_*, OTEL_*, OC_HOST_* env vars are forwarded into the container
    via the SINGULARITYENV_ prefix.
    """
    bind_args: List[str] = []
    for src, dst in binds:
        if src and dst:
            bind_args += ["--bind", f"{src}:{dst}"]

    overlay_args: List[str] = []
    if overlay:
        # `:ro` keeps the patched layer immutable across concurrent runs.
        overlay_args = ["--overlay", f"{overlay}:ro"]

    new_argv = (
        ["singularity", "exec"] + overlay_args + bind_args + [sif] + argv
    )
    new_env = dict(env)
    for k in list(new_env.keys()):
        if k.startswith(("OPENCLAW_", "OTEL_", "OC_HOST_")):
            new_env[f"SINGULARITYENV_{k}"] = new_env[k]
    return new_argv, new_env
