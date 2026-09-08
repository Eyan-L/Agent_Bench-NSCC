from pathlib import Path

import pytest

from agent_bench.core.spec import ConfigError
from agent_bench.deployments.pbs.openclaw_v9_policy import (
    preflight_openclaw_v9_candidate_for_harness,
)


def test_openclaw_v9_pbs_preflight_fails_before_stages_when_candidate_missing(
    tmp_path: Path,
):
    with pytest.raises(ConfigError, match="missing prebuilt OpenClaw"):
        preflight_openclaw_v9_candidate_for_harness(
            "openclaw",
            {
                "PBS_JOBID": "123.pbs101",
                "SCRATCH": str(tmp_path),
            },
        )


def test_openclaw_v9_pbs_preflight_accepts_existing_candidate(tmp_path: Path):
    sif_dir = tmp_path / "sif_images"
    sif_dir.mkdir()
    candidate = sif_dir / "openclaw_2026.9.2_agentbench.sif"
    candidate.write_bytes(b"sealed-candidate")

    preflight_openclaw_v9_candidate_for_harness(
        "openclaw",
        {
            "PBS_JOBID": "123.pbs101",
            "SCRATCH": str(tmp_path),
        },
    )


def test_openclaw_v9_preflight_does_not_affect_other_harnesses(tmp_path: Path):
    preflight_openclaw_v9_candidate_for_harness(
        "openhands",
        {
            "PBS_JOBID": "123.pbs101",
            "SCRATCH": str(tmp_path),
        },
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
