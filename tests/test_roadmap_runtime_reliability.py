from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_bench.benchmarks.roadmap_bench.conditions.measurement_compat import (
    augment_roadmap_measurement,
    stderr_recovery_events,
)
from agent_bench.benchmarks.roadmap_bench.conditions.runtime_policy import (
    repair_roadmap_terminal_evidence,
)
from agent_bench.deployments.pbs.job_spec import _run_stamp
from agent_bench.harnesses.openclaw.runtime_compat_patch import (
    LITELLM_OFFLINE_URL,
    LITELLM_PRICING_URL,
    OPENROUTER_OFFLINE_URL,
    OPENROUTER_PRICING_URL,
    patch_runtime_tree,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_measurement_compat_forwards_execution_bounds(tmp_path):
    from agent_bench.benchmarks.conditions.openclaw_measurement import (
        build_openclaw_measurement,
    )

    result = build_openclaw_measurement(
        tmp_path,
        agent_id="roadmap_bench",
        execution_start_ns=1_000_000_000,
        execution_end_ns=3_000_000_000,
    )

    assert result["runtime_observability"]["execution_time"][
        "agent_end_to_end_seconds"
    ] == 2.0


def test_runtime_patch_relaxes_only_sessions_patch_and_disables_public_pricing(
    tmp_path,
):
    bundle = tmp_path / "dist" / "chunk.js"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(
        "const childPatch={method:'sessions.patch',params:{key:k},timeoutMs:1e4};\n"
        "const other={method:'agent.wait',timeoutMs:10000};\n"
        f"const openrouter='{OPENROUTER_PRICING_URL}';\n"
        f"const litellm='{LITELLM_PRICING_URL}';\n"
    )

    status = patch_runtime_tree(tmp_path, session_patch_timeout_ms=45_000)
    patched = bundle.read_text()

    assert "method:'sessions.patch',params:{key:k},timeoutMs:45000" in patched
    assert "method:'agent.wait',timeoutMs:10000" in patched
    assert OPENROUTER_PRICING_URL not in patched
    assert LITELLM_PRICING_URL not in patched
    assert OPENROUTER_OFFLINE_URL in patched
    assert LITELLM_OFFLINE_URL in patched
    assert status["sessions_patch_timeout_replacements"] == 1
    assert status["openrouter_pricing_url_replacements"] == 1
    assert status["litellm_pricing_url_replacements"] == 1
    assert status["pricing_mode"] == "offline_data_urls"


def test_runtime_patch_fails_closed_when_pricing_anchors_disappear(tmp_path):
    bundle = tmp_path / "dist" / "chunk.js"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(
        "const childPatch={method:'sessions.patch',params:{key:k},timeoutMs:10000};\n"
    )

    with pytest.raises(RuntimeError, match="pricing endpoints"):
        patch_runtime_tree(tmp_path, session_patch_timeout_ms=45_000)


def test_compaction_counts_only_completed_operations_and_emits_truncation(tmp_path):
    (tmp_path / "openclaw_stderr.log").write_text(
        "2026-09-05T02:30:00Z context overflow detected; attempting auto-compaction\n"
        "2026-09-05T02:30:01Z compaction starting for parent session\n"
        "2026-09-05T02:30:05Z auto-compaction succeeded; retrying prompt\n"
        "2026-09-05T02:30:06Z post-compaction guard armed after compaction completed\n"
        "2026-09-05T02:35:00Z context overflow detected; attempting auto-compaction\n"
        "2026-09-05T02:35:05Z compaction completed successfully\n"
        "2026-09-05T02:35:06Z compaction guard armed\n"
        "2026-09-05T02:40:00Z context overflow detected; attempting auto-compaction\n"
        "2026-09-05T02:40:04Z successfully compacted context; retrying prompt\n"
        "2026-09-05T02:45:00Z tool-result truncation removed 318 tool results after context overflow\n"
        "2026-09-05T02:50:57Z maximum context length exceeded\n"
    )

    events = stderr_recovery_events(tmp_path)
    compactions = [event for event in events if event["event_type"] == "compaction"]
    truncations = [
        event
        for event in events
        if event["event_type"] == "tool_result_truncation"
    ]

    assert len(compactions) == 3
    assert all(event["outcome"] == "completed" for event in compactions)
    assert len(truncations) == 1
    assert truncations[0]["truncated_tool_results"] == 318


def test_recovery_events_are_collected_from_gateway_log_and_deduplicated(tmp_path):
    line = "2026-09-07T10:23:00Z tool-result truncation removed 300 tool results after context overflow\n"
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-09-07T10:22:55Z auto-compaction succeeded; retrying prompt\n" + line
    )
    # Some deployments mirror the same recovery diagnostic to stderr. It must
    # not count twice in runtime_observability.
    (tmp_path / "openclaw_stderr.log").write_text(line)

    events = stderr_recovery_events(tmp_path)
    compactions = [event for event in events if event["event_type"] == "compaction"]
    truncations = [
        event
        for event in events
        if event["event_type"] == "tool_result_truncation"
    ]

    assert len(compactions) == 1
    assert compactions[0]["source"] == "openclaw_gateway"
    assert len(truncations) == 1
    assert truncations[0]["truncated_tool_results"] == 300


def test_announce_success_is_inferred_from_child_completion_then_parent_resume(
    tmp_path,
):
    completion_ns = 1_788_560_000_000_000_000
    resume_ns = completion_ns + 18_000_000_000
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "timestamp_ns": completion_ns,
                "timestamp": "2026-09-05T02:29:28+00:00",
                "session_id": "child-1",
                "session_key": "agent:roadmap_bench:subagent:child-1",
                "role": "child",
                "event_type": "child_completion",
            },
            {
                "timestamp_ns": resume_ns,
                "timestamp": "2026-09-05T02:29:46+00:00",
                "session_id": "parent",
                "session_key": "agent:roadmap_bench:main",
                "role": "parent",
                "event_type": "resume",
            },
        ],
    )
    runtime_path = tmp_path / "runtime_observability.json"
    runtime_path.write_text(
        json.dumps(
            {
                "child_session_count": 1,
                "announce_attempt_count": 0,
                "announce_success_count": 0,
            }
        )
    )

    result = augment_roadmap_measurement(
        tmp_path,
        {
            "trajectory_path": str(trajectory),
            "runtime_observability_path": str(runtime_path),
            "runtime_observability": {
                "child_session_count": 1,
                "announce_attempt_count": 0,
                "announce_success_count": 0,
            },
            "trajectory_tokens": {
                "parent": {"turn_count": 1},
                "children": {},
                "total_logical_input": 1,
                "total_logical_output": 1,
            },
            "serving_tokens": {},
        },
    )

    runtime = result["runtime_observability"]
    assert runtime["announce_attempt_count"] == 1
    assert runtime["announce_success_count"] == 1
    assert runtime["announce_failure_count"] == 0
    assert runtime["announce_inferred_success_count"] == 1
    assert runtime["announce_measurement_source"] == (
        "durable_trajectory_child_completion_to_parent_resume"
    )
    assert runtime["announce_inferred_latency_seconds"] == [18.0]


def test_context_overflow_repairs_unknown_terminal_evidence():
    result = repair_roadmap_terminal_evidence(
        {
            "agent_execution_outcome": "context_overflow",
            "agent_lifecycle": {
                "workspaceQuiesced": True,
                "terminalEvidence": "unknown",
            },
        }
    )

    assert result["agent_lifecycle"]["terminalEvidence"] == "context_overflow"
    assert (
        result["agent_lifecycle"]["terminalEvidenceSource"]
        == "agent_execution_outcome"
    )


def test_terminal_evidence_does_not_override_stronger_existing_evidence():
    result = repair_roadmap_terminal_evidence(
        {
            "agent_execution_outcome": "context_overflow",
            "agent_lifecycle": {"terminalEvidence": "durable_session_terminal"},
        }
    )

    assert result["agent_lifecycle"]["terminalEvidence"] == (
        "durable_session_terminal"
    )


def test_nscc_sif_builder_bakes_and_verifies_runtime_compatibility():
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "agent_bench/deployments/pbs/build_oc_sif.sh"
    ).read_text()

    # The actual NSCC runtime path is a sealed patched SIF, not merely an
    # optional overlay. Keep the reliability patch and its fail-closed marker
    # validation wired into that builder.
    assert "runtime_compat_patch.py" in script
    assert "--session-patch-timeout-ms" in script
    assert ".agent-bench-runtime-compat.json" in script
    assert "sessions_patch_timeout_replacements" in script
    assert "openrouter_pricing_url_replacements" in script
    assert "litellm_pricing_url_replacements" in script
    assert "pricing_mode" in script
    assert "offline_data_urls" in script


def test_pbs_run_stamp_accepts_dummy_job_id():
    stamp = _run_stamp({"PBS_JOBID": "12345.pbs101"})

    assert stamp.endswith("_12345.pbs101")
