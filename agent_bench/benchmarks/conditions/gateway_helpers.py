"""Shared OpenClaw-runtime helpers used by harnesses + the OC agent condition.

Benchmark-agnostic (lives under ``agent_bench/benchmarks/conditions/`` next to the OC
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

RoadmapBench additionally requires a real authenticated Gateway RPC before the
agent may start. This prevents a transient ``[gateway] ready`` log line from
being mistaken for an operational Gateway and rejects OpenClaw's automatic
embedded fallback. On firewalled Singularity deployments we also restore the
historical offline pricing settings before the Gateway instance is started.

These functions are NOT SWE-bench specific despite their pre-2026.5 home
under ``agent_bench/benchmarks/swebench/openclaw/``. Both SWE-bench and TB harnesses
reach for them.
"""
from __future__ import annotations

import os
import shlex
import socket
import threading
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------- gateway port pool

# Per-container Gateway port range. Avoids 18789 (host's other gateway) and
# leaves headroom for sidecars (Gateway picks port+2 for the browser plugin
# control surface). Stride of 4 → up to ~25 concurrent runs.
GATEWAY_PORT_MIN = int(os.environ.get("AB_HARNESS_GATEWAY_PORT_MIN", "18800"))
GATEWAY_PORT_MAX = int(os.environ.get("AB_HARNESS_GATEWAY_PORT_MAX", "18899"))
GATEWAY_PORT_STRIDE = int(os.environ.get("AB_HARNESS_GATEWAY_PORT_STRIDE", "4"))

# Process-global reservation — the bind probe releases immediately, so two
# concurrent threads can race on the same port. Hold a logical reservation
# from `pick_free_gateway_port` until `release_gateway_port`.
_GATEWAY_PORT_LOCK = threading.Lock()
_GATEWAY_PORTS_INUSE: Set[int] = set()


def _configure_firewalled_singularity_env() -> None:
    """Restore OpenClaw's offline model-cost settings on Singularity HPC.

    ``TerminalBenchOpenClawHarness.prepare_environment`` calls
    ``pick_free_gateway_port`` immediately before it starts the OpenClaw
    runtime. ``SingularityContainer`` inherits parent ``SINGULARITYENV_*``
    variables for that background gateway process, so setting these here is
    early enough to prevent the runtime from attempting public pricing/model
    catalog fetches on NSCC while leaving Docker/local runs unchanged.
    """
    backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "").lower()
    if backend != "singularity":
        return
    os.environ.setdefault("SINGULARITYENV_OPENCLAW_DISABLE_PRICING_FETCH", "1")
    os.environ.setdefault("SINGULARITYENV_LITELLM_LOCAL_MODEL_COST_MAP", "True")


def pick_free_gateway_port(attempts: int = 25) -> int:
    """Reserve a free port in [GATEWAY_PORT_MIN, GATEWAY_PORT_MAX].

    Range stride accounts for Gateway opening a sidecar at port+2 (browser
    plugin control surface). Reservation is process-global so concurrent
    callers don't collide.
    """
    _configure_firewalled_singularity_env()
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

def _is_roadmap_agent(argv: List[str]) -> bool:
    if len(argv) < 2 or argv[:2] != ["openclaw", "agent"]:
        return False
    try:
        agent_index = argv.index("--agent")
    except ValueError:
        return False
    return agent_index + 1 < len(argv) and argv[agent_index + 1] == "roadmap_bench"


def _roadmap_gateway_guarded_argv(argv: List[str], env: Dict) -> List[str]:
    """Require the exact Roadmap Gateway RPC and reject embedded fallback.

    OpenClaw v2026.4.22 logs ``[gateway] ready`` before every configured
    startup dependency is necessarily usable. The low-level ``gateway call``
    command performs an authenticated WebSocket RPC against the exact URL,
    avoiding false positives from another localhost Gateway. We retry for a
    short bounded window because channels/sidecars may still be settling.

    The upstream CLI can automatically continue in embedded mode after a
    Gateway error. We capture stderr to a file, replay it to Agent_Bench after
    the command, and force a non-zero exit when that fallback marker appears.
    Direct redirection avoids a process-substitution flush race before grep.
    """
    gateway_url = str(env.get("OPENCLAW_GATEWAY_URL") or "")
    gateway_token = str(env.get("OPENCLAW_GATEWAY_TOKEN") or "")
    if not gateway_url or not gateway_token:
        raise RuntimeError(
            "RoadmapBench gateway-only mode requires OPENCLAW_GATEWAY_URL and "
            "OPENCLAW_GATEWAY_TOKEN"
        )

    ready_timeout_s = max(
        1, int(os.environ.get("AB_OPENCLAW_GATEWAY_RPC_READY_TIMEOUT_SECONDS", "120"))
    )
    probe_timeout_ms = max(
        1000, int(os.environ.get("AB_OPENCLAW_GATEWAY_RPC_PROBE_TIMEOUT_MS", "10000"))
    )
    command = shlex.join(argv)
    script = f"""
set -u
ab_probe_log=$(mktemp /tmp/agent-bench-gateway-probe.XXXXXX)
ab_deadline=$((SECONDS + {ready_timeout_s}))
while ! openclaw gateway call status \
  --url {shlex.quote(gateway_url)} \
  --token {shlex.quote(gateway_token)} \
  --timeout {probe_timeout_ms} --json >\"$ab_probe_log\" 2>&1; do
  if (( SECONDS >= ab_deadline )); then
    echo '[agent-bench] Roadmap gateway RPC readiness failed; refusing embedded fallback' >&2
    cat \"$ab_probe_log\" >&2
    rm -f \"$ab_probe_log\"
    exit 86
  fi
  sleep 2
done
rm -f \"$ab_probe_log\"

ab_stderr=$(mktemp /tmp/agent-bench-openclaw-stderr.XXXXXX)
{command} 2>\"$ab_stderr\"
ab_rc=$?
cat \"$ab_stderr\" >&2
if grep -Fqi 'falling back to embedded' \"$ab_stderr\"; then
  echo '[agent-bench] embedded fallback detected; Roadmap gateway-only run is invalid' >&2
  rm -f \"$ab_stderr\"
  exit 87
fi
rm -f \"$ab_stderr\"
exit \"$ab_rc\"
""".strip()
    return ["bash", "-lc", script]


def wrap_singularity_argv(
    argv: List[str], env: Dict, binds: List, sif: str, overlay: Optional[str] = None,
) -> Tuple[List[str], Dict]:
    """Prefix `argv` with `singularity exec [--overlay <ov>:ro] --bind ... <sif>`.

    Caller-supplied `sif` and `overlay` (no env-var sniffing) so call sites
    that already know they're on the singularity backend can wrap explicitly.

    `binds` is a list of (src, dst) path pairs. Use src==dst when possible so
    absolute paths in env vars (OPENCLAW_STATE_DIR etc) resolve identically
    inside the container.

    OPENCLAW_*, LITELLM_*, OTEL_*, and AB_* env vars are forwarded into the
    container via the SINGULARITYENV_ prefix.
    """
    bind_args: List[str] = []
    for src, dst in binds:
        if src and dst:
            bind_args += ["--bind", f"{src}:{dst}"]

    overlay_args: List[str] = []
    if overlay:
        # `:ro` keeps the patched layer immutable across concurrent runs.
        overlay_args = ["--overlay", f"{overlay}:ro"]

    new_env = dict(env)
    # The CLI may instantiate model metadata independently of the already
    # running Gateway, so keep it offline too. Parent SINGULARITYENV_* values
    # set by pick_free_gateway_port cover the Gateway process itself.
    new_env.setdefault("OPENCLAW_DISABLE_PRICING_FETCH", "1")
    new_env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    inner_argv = (
        _roadmap_gateway_guarded_argv(argv, new_env)
        if _is_roadmap_agent(argv)
        else argv
    )
    new_argv = (
        ["singularity", "exec"] + overlay_args + bind_args + [sif] + inner_argv
    )
    for k in list(new_env.keys()):
        if k.startswith(("OPENCLAW_", "LITELLM_", "OTEL_", "AB_")):
            new_env[f"SINGULARITYENV_{k}"] = str(new_env[k])
    return new_argv, new_env
