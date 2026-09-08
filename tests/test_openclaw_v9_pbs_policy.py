from pathlib import Path

import pytest

from agent_bench.core.spec import ConfigError
from agent_bench.deployments.pbs.openclaw_v9_policy import (
    OPENCLAW_VERSION,
    openclaw_v9_runtime_defaults,
    require_prebuilt_openclaw_v9_candidate,
)


def test_openclaw_v9_runtime_defaults_use_sealed_candidate_names(tmp_path: Path):
    defaults = openclaw_v9_runtime_defaults(str(Path.cwd()), str(tmp_path))

    assert defaults["AB_AGENT_RUNTIME_VERSION"] == OPENCLAW_VERSION
    assert defaults["AB_AGENT_RUNTIME_BASE_SIF"].endswith(
        "/openclaw_2026.9.2_base.sif"
    )
    assert defaults["AB_AGENT_RUNTIME_SIF"].endswith(
        "/openclaw_2026.9.2_agentbench.sif"
    )
    assert defaults["AB_AGENT_RUNTIME_DOCKER"].endswith(":2026.9.2")
    assert defaults["AB_AGENT_RUNTIME_REBUILD"] == "0"


def test_openclaw_v9_candidate_missing_fails_closed(tmp_path: Path):
    missing = tmp_path / "openclaw_2026.9.2_agentbench.sif"

    with pytest.raises(ConfigError, match="Refusing legacy pull/build fallback"):
        require_prebuilt_openclaw_v9_candidate(
            {
                "AB_AGENT_RUNTIME_SIF": str(missing),
                "AB_AGENT_RUNTIME_REBUILD": "0",
            }
        )


def test_openclaw_v9_candidate_exists_needs_no_build_command(tmp_path: Path):
    candidate = tmp_path / "openclaw_2026.9.2_agentbench.sif"
    candidate.write_bytes(b"sealed-candidate")

    assert require_prebuilt_openclaw_v9_candidate(
        {
            "AB_AGENT_RUNTIME_SIF": str(candidate),
            "AB_AGENT_RUNTIME_REBUILD": "0",
        }
    ) == []


def test_openclaw_v9_rebuild_is_rejected_even_with_candidate(tmp_path: Path):
    candidate = tmp_path / "openclaw_2026.9.2_agentbench.sif"
    candidate.write_bytes(b"sealed-candidate")

    with pytest.raises(ConfigError, match="AB_AGENT_RUNTIME_REBUILD=1"):
        require_prebuilt_openclaw_v9_candidate(
            {
                "AB_AGENT_RUNTIME_SIF": str(candidate),
                "AB_AGENT_RUNTIME_REBUILD": "1",
            }
        )
