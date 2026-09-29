from pathlib import Path

import pytest

from agent_bench.core.spec import ConfigError
from agent_bench.deployments.pbs.openclaw_v9_policy import (
    OPENCLAW_VERSION,
    install_openclaw_v9_pbs_policy,
    openclaw_v9_runtime_defaults,
    preflight_openclaw_v9_candidate_for_harness,
    require_prebuilt_openclaw_v9_candidate,
)


def test_openclaw_v9_policy_preserves_roadmap_hermes_runtime(tmp_path: Path):
    from agent_bench.deployments.pbs import stages

    original_defaults = stages._agent_runtime_defaults
    original_commands = stages._openclaw_runtime_commands
    try:
        install_openclaw_v9_pbs_policy()
        defaults = stages._agent_runtime_defaults(
            "hermes", str(Path.cwd()), str(tmp_path), roadmapbench=True
        )
        assert defaults["AB_AGENT_RUNTIME_SIF"].endswith(
            "/hermes_roadmap_v2026.5.16.sif"
        )
    finally:
        stages._agent_runtime_defaults = original_defaults
        stages._openclaw_runtime_commands = original_commands


def test_openclaw_v9_runtime_defaults_use_sealed_candidate_names(tmp_path: Path):
    defaults = openclaw_v9_runtime_defaults(str(Path.cwd()), str(tmp_path))

    assert defaults["AB_AGENT_RUNTIME_VERSION"] == OPENCLAW_VERSION
    assert defaults["AB_AGENT_RUNTIME_BASE_SIF"].endswith(
        f"/openclaw_{OPENCLAW_VERSION}_agentbench.sif"
    )
    assert defaults["AB_AGENT_RUNTIME_SIF"].endswith(
        f"/openclaw_{OPENCLAW_VERSION}_agentbench.sif"
    )
    assert defaults["AB_AGENT_RUNTIME_DOCKER"].endswith(f":{OPENCLAW_VERSION}")
    assert defaults["AB_AGENT_RUNTIME_REBUILD"] == "0"


def test_openclaw_v9_candidate_missing_fails_closed(tmp_path: Path):
    missing = tmp_path / f"openclaw_{OPENCLAW_VERSION}_agentbench.sif"

    with pytest.raises(ConfigError, match="Refusing legacy pull/build fallback"):
        require_prebuilt_openclaw_v9_candidate(
            {
                "AB_AGENT_RUNTIME_SIF": str(missing),
                "AB_AGENT_RUNTIME_REBUILD": "0",
            }
        )


def test_openclaw_v9_candidate_exists_needs_no_build_command(tmp_path: Path):
    candidate = tmp_path / f"openclaw_{OPENCLAW_VERSION}_agentbench.sif"
    candidate.write_bytes(b"sealed-candidate")

    assert require_prebuilt_openclaw_v9_candidate(
        {
            "AB_AGENT_RUNTIME_SIF": str(candidate),
            "AB_AGENT_RUNTIME_REBUILD": "0",
        }
    ) == []


def test_openclaw_v9_rebuild_is_rejected_even_with_candidate(tmp_path: Path):
    candidate = tmp_path / f"openclaw_{OPENCLAW_VERSION}_agentbench.sif"
    candidate.write_bytes(b"sealed-candidate")

    with pytest.raises(ConfigError, match="AB_AGENT_RUNTIME_REBUILD=1"):
        require_prebuilt_openclaw_v9_candidate(
            {
                "AB_AGENT_RUNTIME_SIF": str(candidate),
                "AB_AGENT_RUNTIME_REBUILD": "1",
            }
        )


def test_openclaw_v9_pbs_preflight_fails_before_stages_when_candidate_missing(
    tmp_path: Path,
):
    with pytest.raises(ConfigError, match="missing prebuilt OpenClaw"):
        preflight_openclaw_v9_candidate_for_harness(
            "openclaw",
            {"PBS_JOBID": "123.pbs101", "SCRATCH": str(tmp_path)},
        )


def test_openclaw_v9_pbs_preflight_accepts_existing_candidate(tmp_path: Path):
    sif_dir = tmp_path / "sif_images"
    sif_dir.mkdir()
    (sif_dir / f"openclaw_{OPENCLAW_VERSION}_agentbench.sif").write_bytes(
        b"sealed-candidate"
    )

    preflight_openclaw_v9_candidate_for_harness(
        "openclaw",
        {"PBS_JOBID": "123.pbs101", "SCRATCH": str(tmp_path)},
    )


def test_openclaw_v9_preflight_does_not_affect_other_harnesses(tmp_path: Path):
    preflight_openclaw_v9_candidate_for_harness(
        "openhands",
        {"PBS_JOBID": "123.pbs101", "SCRATCH": str(tmp_path)},
    )


def test_openclaw_v9_preflight_preserves_explicit_legacy_runtime(tmp_path: Path):
    preflight_openclaw_v9_candidate_for_harness(
        "openclaw",
        {
            "PBS_JOBID": "123.pbs101",
            "SCRATCH": str(tmp_path),
            "AB_AGENT_RUNTIME_VERSION": "2026.4.22",
        },
    )
