"""Ensure runtime metadata stays consistent across durable accounting outputs."""
import json

import pytest

from agent_bench.benchmarks.roadmap_bench.conditions.measurement_compat import (
    _rewrite_accounting_artifacts,
)


@pytest.mark.parametrize(
    "recorded,configured,expected,source",
    [
        (None, "2026.9.2", "v2026.9.2", "AB_AGENT_RUNTIME_VERSION"),
        (None, " v2026.9.2 ", "v2026.9.2", "AB_AGENT_RUNTIME_VERSION"),
        ("v2026.9.6", "2026.9.2", "v2026.9.6", "runtime_observability"),
        ("2026.9.2", None, "v2026.9.2", "runtime_observability"),
        (None, None, None, "unknown"),
    ],
)
def test_recorded_version_is_preserved_and_all_outputs_agree(
    tmp_path, monkeypatch, recorded, configured, expected, source
):
    if configured is None:
        monkeypatch.delenv("AB_AGENT_RUNTIME_VERSION", raising=False)
    else:
        monkeypatch.setenv("AB_AGENT_RUNTIME_VERSION", configured)
    runtime = {
        "stuck_diagnostic_policy": {"runtime_version": "v2026.4.22", "is_failure": False},
        "announce_attribution": {"limitation": "Legacy fixed version description"},
    }
    if recorded:
        runtime["runtime_version"] = recorded
    result = {"runtime_observability": runtime, "reward": 0.5,
              "agent_execution_seconds": 10, "serving_tokens": {"input_tokens": 12}}
    _rewrite_accounting_artifacts(tmp_path, result, [])
    observations = [
        result["runtime_observability"],
        json.loads((tmp_path / "runtime_observability.json").read_text()),
        json.loads((tmp_path / "token_accounting.json").read_text())["runtime_observability"],
    ]
    for observation in observations:
        assert observation["runtime_version"] == expected
        assert observation["runtime_version_source"] == source
        assert observation["stuck_diagnostic_policy"]["runtime_version"] == expected
        assert observation["stuck_diagnostic_policy"]["is_failure"] is False
        assert "2026.4.22" not in observation["announce_attribution"]["limitation"]
    assert result["reward"] == 0.5
    assert result["agent_execution_seconds"] == 10
    assert result["serving_tokens"] == {"input_tokens": 12}
