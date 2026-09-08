"""Canonical prebuilt OpenClaw v2026.9.2 policy for PBS execution.

The v2026.9.2 Agent_Bench runtime is intentionally built and audited off-node.
Formal NSCC jobs may consume that sealed candidate, but must never silently
fall back to the legacy v2026.4.22 builder when the candidate is absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from agent_bench.core.spec import ConfigError

OPENCLAW_VERSION = "2026.9.2"


def openclaw_v9_runtime_defaults(
    repo_root: str,
    scratch: str,
) -> dict[str, str]:
    """Return the canonical OpenClaw v2026.9.2 PBS runtime tuple."""
    from agent_bench.deployments.pbs import stages

    version = stages._version_from_file(
        Path(repo_root)
        / "agent_bench/harnesses/openclaw/patch/VERSION.v2026.9.2",
        "upstream_version",
        OPENCLAW_VERSION,
    )
    return {
        "AB_AGENT_RUNTIME_VERSION": version,
        "AB_AGENT_RUNTIME_BASE_SIF": (
            f"{scratch}/sif_images/openclaw_{version}_base.sif"
        ),
        "AB_AGENT_RUNTIME_SIF": (
            f"{scratch}/sif_images/openclaw_{version}_agentbench.sif"
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
    """Validate the sealed v9.2 candidate and return no build commands."""
    candidate = Path(str(env.get("AB_AGENT_RUNTIME_SIF") or "")).expanduser()
    if env.get("AB_AGENT_RUNTIME_REBUILD") == "1":
        raise ConfigError(
            "OpenClaw v2026.9.2 PBS jobs require a prebuilt audited candidate; "
            "AB_AGENT_RUNTIME_REBUILD=1 is not supported. Build with "
            "agent_bench/deployments/pbs/build_oc_v2026_9_2_sif.sh off-node."
        )
    try:
        usable = candidate.is_file() and candidate.stat().st_size > 0
    except OSError:
        usable = False
    if not usable:
        raise ConfigError(
            "missing prebuilt OpenClaw v2026.9.2 candidate SIF: "
            f"{candidate}. Refusing legacy pull/build fallback."
        )
    return []


def install_openclaw_v9_pbs_policy() -> None:
    """Make v2026.9.2 the OpenClaw default and fail closed on missing SIF."""
    from agent_bench.deployments.pbs import stages

    current_defaults = stages._agent_runtime_defaults
    if not getattr(current_defaults, "_openclaw_v9_default_policy", False):
        def runtime_defaults(harness: str, repo_root: str, scratch: str):
            if harness == "openclaw":
                return openclaw_v9_runtime_defaults(repo_root, scratch)
            return current_defaults(harness, repo_root, scratch)

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
