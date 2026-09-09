from __future__ import annotations

import json
from pathlib import Path

from agent_bench.benchmarks.conditions import gateway_helpers
from agent_bench.benchmarks.roadmap_bench.conditions.measurement_compat import (
    augment_roadmap_measurement,
    stderr_recovery_events,
)
from agent_bench.runtime import driver


def test_singularity_gateway_offline_env_is_installed_before_runtime(monkeypatch):
    monkeypatch.setenv("AB_DEPLOYMENT_CONTAINER_BACKEND", "singularity")
    monkeypatch.delenv(
        "SINGULARITYENV_OPENCLAW_DISABLE_PRICING_FETCH", raising=False
    )
    monkeypatch.delenv(
        "SINGULARITYENV_LITELLM_LOCAL_MODEL_COST_MAP", raising=False
    )

    gateway_helpers._configure_firewalled_singularity_env()

    assert (
        gateway_helpers.os.environ[
            "SINGULARITYENV_OPENCLAW_DISABLE_PRICING_FETCH"
        ]
        == "1"
    )
    assert (
        gateway_helpers.os.environ[
            "SINGULARITYENV_LITELLM_LOCAL_MODEL_COST_MAP"
        ]
        == "True"
    )


def test_non_singularity_does_not_install_offline_gateway_env(monkeypatch):
    monkeypatch.setenv("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker")
    monkeypatch.delenv(
        "SINGULARITYENV_OPENCLAW_DISABLE_PRICING_FETCH", raising=False
    )
    monkeypatch.delenv(
        "SINGULARITYENV_LITELLM_LOCAL_MODEL_COST_MAP", raising=False
    )

    gateway_helpers._configure_firewalled_singularity_env()

    assert "SINGULARITYENV_OPENCLAW_DISABLE_PRICING_FETCH" not in gateway_helpers.os.environ
    assert "SINGULARITYENV_LITELLM_LOCAL_MODEL_COST_MAP" not in gateway_helpers.os.environ


def test_roadmap_singularity_agent_requires_exact_gateway_rpc_and_rejects_fallback(
    monkeypatch,
):
    monkeypatch.setenv("AB_OPENCLAW_GATEWAY_RPC_READY_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("AB_OPENCLAW_GATEWAY_RPC_PROBE_TIMEOUT_MS", "4321")
    argv = [
        "openclaw",
        "agent",
        "--agent",
        "roadmap_bench",
        "--session-id",
        "run-1",
        "--message",
        "implement roadmap",
        "--timeout",
        "2400",
        "--json",
    ]
    wrapped, env = gateway_helpers.wrap_singularity_argv(
        argv,
        {
            "OPENCLAW_GATEWAY_URL": "ws://127.0.0.1:18800",
            "OPENCLAW_GATEWAY_TOKEN": "secret-token",
        },
        binds=[("/tmp/work", "/tmp/work")],
        sif="/tmp/openclaw.sif",
    )

    assert wrapped[:2] == ["singularity", "exec"]
    assert wrapped[-3:-1] == ["bash", "-lc"]
    guard = wrapped[-1]
    assert "openclaw gateway call status" in guard
    assert "--url ws://127.0.0.1:18800" in guard
    assert "--token secret-token" in guard
    assert "--timeout 4321 --json" in guard
    assert "SECONDS + 17" in guard
    assert "refusing embedded fallback" in guard
    assert "falling back to embedded" in guard
    assert "exit 86" in guard
    assert "exit 87" in guard
    assert "Roadmap gateway RPC ready; gateway execution selected" in guard
    assert "openclaw agent --agent roadmap_bench" in guard
    # Fallback detection reads a completed stderr file, so grep cannot race a
    # still-running process-substitution tee.
    assert '2>"$ab_stderr"' in guard
    assert 'cat "$ab_stderr" >&2' in guard
    assert "2> >(tee" not in guard
    assert env["OPENCLAW_DISABLE_PRICING_FETCH"] == "1"
    assert env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert env["SINGULARITYENV_OPENCLAW_DISABLE_PRICING_FETCH"] == "1"
    assert env["SINGULARITYENV_LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_other_singularity_agents_are_not_gateway_guard_wrapped():
    argv = [
        "openclaw", "agent", "--agent", "terminal_bench", "--json"
    ]
    wrapped, _env = gateway_helpers.wrap_singularity_argv(
        argv,
        {
            "OPENCLAW_GATEWAY_URL": "ws://127.0.0.1:18800",
            "OPENCLAW_GATEWAY_TOKEN": "secret-token",
        },
        binds=[],
        sif="/tmp/openclaw.sif",
    )
    assert wrapped[-len(argv):] == argv


def test_roadmap_runtime_policy_rejects_explicit_non_gateway_modes():
    for mode in ("embedded_fallback", "embedded"):
        error = driver._roadmap_benchmark_invalid_reason({
            "agent_runtime_mode": mode,
            "agent_execution_outcome": "runtime_error",
            "agent_lifecycle": {"workspaceQuiesced": True},
        })
        assert error is not None
        assert "requires OpenClaw gateway runtime" in error
        assert f"agent_runtime_mode={mode}" in error

    assert driver._roadmap_benchmark_invalid_reason({
        "agent_runtime_mode": "gateway",
        "agent_lifecycle": {"workspaceQuiesced": True},
    }) is None


def test_roadmap_runtime_policy_rejects_unknown_mode_independent_of_outcome():
    error = driver._roadmap_benchmark_invalid_reason({
        "agent_runtime_mode": "unknown",
        "agent_execution_outcome": "runtime_error",
        "agent_lifecycle": {"workspaceQuiesced": True},
    })
    assert error is not None
    assert "agent_runtime_mode=unknown" in error


def test_roadmap_runtime_policy_preserves_legacy_lifecycle_check_without_mode():
    error = driver._roadmap_lifecycle_verifier_error({
        "agent_lifecycle": {
            "workspaceQuiesced": False,
            "lifecycleError": "writers still active",
        }
    })
    assert error == "writers still active"


def test_runtime_failure_keeps_verifier_measurement_but_is_benchmark_invalid():
    verdict = driver._finalize_roadmap_verdict(
        {
            "status": "FAIL",
            "verifier_status": "FAIL",
            "reward": 0.4,
            "test_pass_rate": 0.4,
        },
        {
            "agent_execution_outcome": "runtime_error",
            "agent_completion_error": "incomplete_turn: Connection error",
            "exit_code": 1,
            "agent_lifecycle": {"workspaceQuiesced": True},
        },
        verifier_error=None,
        benchmark_invalid_reason="incomplete_turn: Connection error",
    )

    assert verdict["status"] == "ERROR"
    assert verdict["benchmark_valid"] is False
    assert verdict["verifier_eligible"] is True
    assert verdict["verifier_status"] == "FAIL"
    assert verdict["reward"] == 0.4
    assert verdict["test_pass_rate"] == 0.4


def test_roadmap_harness_trajectory_events_are_machine_readable(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.touch()

    driver._append_roadmap_trajectory_event(
        tmp_path,
        "verifier_start",
        timestamp_ns=100,
        candidate_patch_bytes=42,
    )
    driver._append_roadmap_trajectory_event(
        tmp_path,
        "verifier_result",
        timestamp_ns=200,
        status="FAIL",
        reward=0.4,
    )

    events = [json.loads(line) for line in trajectory.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "verifier_start", "verifier_result",
    ]
    assert events[0]["timestamp_ns"] == 100
    assert events[0]["candidate_patch_bytes"] == 42
    assert events[1]["status"] == "FAIL"
    assert events[1]["reward"] == 0.4


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_stderr_recovery_events_are_timestamped_and_typed(tmp_path):
    (tmp_path / "openclaw_stderr.log").write_text(
        "2026-09-04T13:30:00Z context overflow detected; attempting recovery\n"
        "2026-09-04T13:30:01Z compaction succeeded; retrying prompt\n"
        "2026-09-04T13:30:02Z truncating 513 tool results after context overflow\n"
        "2026-09-04T13:30:03Z maximum context length exceeded\n"
        "untimestamped compaction succeeded\n"
    )

    events = stderr_recovery_events(tmp_path)
    event_types = [event["event_type"] for event in events]

    assert event_types.count("compaction") == 1
    assert event_types.count("tool_result_truncation") == 1
    assert event_types.count("context_overflow") == 3
    assert all(event["timestamp_ns"] > 0 for event in events)
    truncation = next(
        event for event in events
        if event["event_type"] == "tool_result_truncation"
    )
    assert truncation["truncated_tool_results"] == 513
    assert all(event["source"] == "openclaw_stderr" for event in events)


def test_roadmap_measurement_augments_trajectory_and_uses_raw_token_fallback(
    tmp_path,
):
    trajectory_path = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory_path,
        [{
            "timestamp": "2026-09-04T13:29:59+00:00",
            "timestamp_ns": 1788528599000000000,
            "session_id": "parent",
            "session_key": "agent:roadmap_bench:main",
            "role": "parent",
            "event_type": "tool_call",
        }],
    )
    (tmp_path / "openclaw_stderr.log").write_text(
        "2026-09-04T13:30:01Z compaction completed; retrying prompt\n"
        "2026-09-04T13:30:02Z truncating 513 tool results after context overflow\n"
    )

    zero_tokens = {
        "parent": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "turn_count": 0,
            "resumed_turns": 7,
        },
        "children": {},
        "total_logical_input": 0,
        "total_logical_output": 0,
    }
    runtime_path = tmp_path / "runtime_observability.json"
    runtime_path.write_text(json.dumps({"child_session_count": 1}))
    accounting_path = tmp_path / "token_accounting.json"
    accounting_path.write_text(json.dumps({
        "trajectory_tokens": zero_tokens,
        "serving_tokens": {
            "processed_input": 47_443_942,
            "generated_output": 224_258,
            "request_count": 569,
        },
        "runtime_observability": {"child_session_count": 1},
        "sources": {},
    }))

    result = augment_roadmap_measurement(tmp_path, {
        "trajectory_path": str(trajectory_path),
        "token_accounting_path": str(accounting_path),
        "runtime_observability_path": str(runtime_path),
        "trajectory_tokens": zero_tokens,
        "serving_tokens": {
            "processed_input": 47_443_942,
            "generated_output": 224_258,
            "request_count": 569,
        },
        "runtime_observability": {"child_session_count": 1},
    })

    assert result["total_prompt_tokens"] == 47_443_942
    assert result["total_completion_tokens"] == 224_258
    assert result["trajectory_tokens"]["total_logical_input"] == 47_443_942
    assert result["trajectory_tokens"]["total_logical_output"] == 224_258
    assert (
        result["trajectory_tokens"]["accounting_source"]
        == "raw_response_fallback_unattributed"
    )
    assert result["trajectory_tokens"]["parent"]["resumed_turns"] == 7

    rows = _read_jsonl_for_test(trajectory_path)
    assert sum(row["event_type"] == "compaction" for row in rows) == 1
    assert sum(row["event_type"] == "context_overflow" for row in rows) == 1
    assert sum(row["event_type"] == "tool_result_truncation" for row in rows) == 1

    runtime = json.loads(runtime_path.read_text())
    assert runtime["stderr_compaction_count"] == 1
    assert runtime["stderr_context_overflow_count"] == 1
    assert runtime["tool_result_truncation_count"] == 1
    assert runtime["child_session_count"] == 1

    accounting = json.loads(accounting_path.read_text())
    assert (
        accounting["sources"]["trajectory_tokens"].startswith(
            "raw OpenAI-compatible response usage fallback"
        )
    )
    assert (
        accounting["sources"]["runtime_recovery_events"]
        == "timestamped openclaw_stderr.log"
    )


def _read_jsonl_for_test(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
