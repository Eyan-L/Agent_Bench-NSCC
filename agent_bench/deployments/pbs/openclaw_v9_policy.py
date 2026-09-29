"""Canonical prebuilt OpenClaw v2026.9.6 policy for PBS execution.

The v9 Agent_Bench runtime is intentionally built and audited off-node.
Formal NSCC jobs may consume that sealed candidate, but must never silently
fall back to the legacy v2026.4.22 builder when the candidate is absent.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from agent_bench.core.spec import ConfigError

OPENCLAW_VERSION = "2026.9.6"
OPENCLAW_ACTIVE_SIF = f"openclaw_{OPENCLAW_VERSION}_agentbench.sif"


def openclaw_v9_runtime_defaults(
    repo_root: str,
    scratch: str,
) -> dict[str, str]:
    """Return the canonical prebuilt OpenClaw v9 PBS runtime tuple."""
    from agent_bench.deployments.pbs import stages

    version = stages._version_from_file(
        Path(repo_root)
        / f"agent_bench/harnesses/openclaw/patch/VERSION.v{OPENCLAW_VERSION}",
        "upstream_version",
        OPENCLAW_VERSION,
    )
    return {
        "AB_AGENT_RUNTIME_VERSION": version,
        "AB_AGENT_RUNTIME_BASE_SIF": (
            f"{scratch}/sif_images/{OPENCLAW_ACTIVE_SIF}"
        ),
        "AB_AGENT_RUNTIME_SIF": (
            f"{scratch}/sif_images/{OPENCLAW_ACTIVE_SIF}"
        ),
        "AB_AGENT_RUNTIME_DOCKER": (
            f"docker://ghcr.io/openclaw/openclaw:{version}"
        ),
        "AB_AGENT_RUNTIME_REBUILD": "0",
        "AB_AGENT_RUNTIME_OVERLAY": "",
        "AB_AGENT_RUNTIME_INSIDE_PKG": "/app",
        "AB_AGENT_RUNTIME_BASE_DOCKER": (
            f"docker://ghcr.io/openclaw/openclaw:{version}"
        ),
    }


def require_prebuilt_openclaw_v9_candidate(
    env: Mapping[str, str],
) -> list:
    """Validate the prebuilt deployment slot and return no build commands."""
    candidate = Path(str(env.get("AB_AGENT_RUNTIME_SIF") or "")).expanduser()
    if env.get("AB_AGENT_RUNTIME_REBUILD") == "1":
        raise ConfigError(
            f"OpenClaw v{OPENCLAW_VERSION} PBS jobs require a prebuilt audited candidate; "
            "AB_AGENT_RUNTIME_REBUILD=1 is not supported. Build with "
            "agent_bench/deployments/pbs/build_oc_v2026_9_6_sif.sh off-node."
        )
    try:
        usable = candidate.is_file() and candidate.stat().st_size > 0
    except OSError:
        usable = False
    if not usable:
        raise ConfigError(
            f"missing prebuilt OpenClaw v{OPENCLAW_VERSION} candidate SIF: "
            f"{candidate}. Refusing legacy pull/build fallback."
        )
    return []


def preflight_openclaw_v9_candidate_for_harness(
    harness: str,
    env: Mapping[str, str],
) -> None:
    """Fail before expensive PBS stages when the sealed candidate is absent."""
    if not env.get("PBS_JOBID") or harness != "openclaw":
        return
    explicit_version = str(env.get("AB_AGENT_RUNTIME_VERSION") or "").strip()
    if explicit_version and explicit_version != OPENCLAW_VERSION:
        # Explicit legacy experiments retain their historical runtime path.
        return

    user = env.get("USER") or os.environ.get("USER") or "user"
    scratch = env.get("SCRATCH") or f"/scratch/users/ntu/{user}"
    effective = dict(env)
    effective.setdefault(
        "AB_AGENT_RUNTIME_SIF",
        f"{scratch}/sif_images/{OPENCLAW_ACTIVE_SIF}",
    )
    effective.setdefault("AB_AGENT_RUNTIME_REBUILD", "0")
    require_prebuilt_openclaw_v9_candidate(effective)


def _config_arg(argv: Sequence[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg == "--config" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def preflight_openclaw_v9_pbs_candidate(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Resolve the requested harness and preflight v9 inside PBS jobs."""
    source_env = os.environ if env is None else env
    if not source_env.get("PBS_JOBID"):
        return
    config = _config_arg(list(sys.argv[1:] if argv is None else argv))
    if not config:
        return

    from agent_bench.core.spec import load_resolved_experiment

    resolved = load_resolved_experiment(config)
    harness = str(
        resolved.harness.get("harness")
        or resolved.harness.get("name")
        or resolved.spec.harness_name
    )
    preflight_openclaw_v9_candidate_for_harness(harness, source_env)


def install_openclaw_v9_pbs_policy() -> None:
    """Make the pinned v9 release the default and fail closed on missing SIF."""
    from agent_bench.deployments.pbs import stages

    current_defaults = stages._agent_runtime_defaults
    if not getattr(current_defaults, "_openclaw_v9_default_policy", False):
        def runtime_defaults(
            harness: str, repo_root: str, scratch: str, *, roadmapbench: bool = False
        ):
            if harness == "openclaw":
                return openclaw_v9_runtime_defaults(repo_root, scratch)
            return current_defaults(
                harness, repo_root, scratch, roadmapbench=roadmapbench
            )

        setattr(runtime_defaults, "_openclaw_v9_default_policy", True)
        stages._agent_runtime_defaults = runtime_defaults

    current_commands = stages._openclaw_runtime_commands
    if not getattr(current_commands, "_openclaw_v9_prebuilt_policy", False):
        def runtime_commands(spec, env, repo_root):
            version = str(env.get("AB_AGENT_RUNTIME_VERSION") or "").strip()
            if version == OPENCLAW_VERSION:
                return require_prebuilt_openclaw_v9_candidate(env)
            # Explicit legacy experiments retain their historical builder.
            return current_commands(spec, env, repo_root)

        setattr(runtime_commands, "_openclaw_v9_prebuilt_policy", True)
        stages._openclaw_runtime_commands = runtime_commands
