"""Shared OpenClaw-runtime helpers used by harnesses + the OC agent condition.

Benchmark-agnostic (lives under ``applications/conditions/`` next to the OC
agent base class, ``_llm_client``, and ``util_sampler``). Two concerns:

  1. **Gateway port pool** — every OC container needs a unique loopback port
     for its WebSocket gateway; this picks one + reserves it process-globally
     so concurrent runs don't collide. Each app's harness calls
     ``pick_free_gateway_port()`` / ``release_gateway_port(port)`` in its
     ``prepare_environment`` / ``cleanup``.

  2. **Singularity argv wrapper** — when the OC condition runs in singularity
     mode (HPC nodes without docker), the openclaw CLI must be wrapped in
     ``singularity exec --bind ... <sif> openclaw ...``. The base condition
     class calls ``wrap_singularity_argv`` inside ``_invoke_agent`` when
     ``env_info["backend"] == "singularity"``.

These functions are NOT SWE-bench specific despite their pre-2026.5 home
under ``applications/swebench/openclaw/``. Both SWE-bench and TB harnesses
reach for them.
"""
from __future__ import annotations

import os
import socket
import threading
from typing import Dict, List, Optional, Set, Tuple


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
