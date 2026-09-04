import json

from agent_bench.benchmarks.conditions.openclaw_measurement import (
    build_openclaw_measurement,
)
from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)
from agent_bench.core.manifest import RunManifest


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_manifest_round_trip_preserves_agent_phase_timing():
    manifest = RunManifest(
        run_id="run",
        condition_id="condition",
        instance_id="instance",
        seed=42,
        serving_config_id="serving",
        status="PASS",
        duration_seconds=100.0,
        total_instance_seconds=110.0,
        agent_execution_seconds=60.0,
        phase_timing={"verifier_seconds": 30.0},
    )
    restored = RunManifest.from_dict(manifest.to_dict())
    assert restored.duration_seconds == 100.0
    assert restored.total_instance_seconds == 110.0
    assert restored.agent_execution_seconds == 60.0
    assert restored.phase_timing == {"verifier_seconds": 30.0}


def test_roadmap_quiescence_timing_separates_parent_terminal(monkeypatch):
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    snapshot = {
        "parent_terminal": True,
        "parent_session_terminal": True,
        "pending_child_sessions": [],
    }
    clock = iter([1_000_000_000, 3_000_000_000])
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.time_ns",
        lambda: next(clock),
    )
    timed_snapshot, quiescence = condition._quiesce_with_timing(
        snapshot, logical_parent_terminal=True
    )
    assert quiescence["workspaceQuiesced"] is True
    assert timed_snapshot["parent_terminal_observed_ns"] == 1_000_000_000
    assert timed_snapshot["workspace_quiesced_ns"] == 3_000_000_000


def test_child_runtime_and_unmaterialized_spawn_are_distinct(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    child_a = "agent:roadmap_bench:subagent:a"
    child_b = "agent:roadmap_bench:subagent:b"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": "parent-id",
            "sessionFile": "parent.jsonl",
            "status": "done",
        },
        child_a: {
            "sessionId": "child-a-id",
            "sessionFile": "child-a.jsonl",
            "spawnedBy": parent_key,
            "status": "done",
            "createdAt": "2026-01-01T00:00:02Z",
            "endedAt": "2026-01-01T00:00:05Z",
        },
        child_b: {
            "sessionId": "child-b-id",
            "sessionFile": "child-b.jsonl",
            "spawnedBy": parent_key,
            "status": "done",
        },
    }))
    _write_jsonl(sessions / "parent.jsonl", [{
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:01Z",
            "stopReason": "toolUse",
            "content": [
                {"type": "toolCall", "id": "spawn-a", "name": "sessions_spawn", "arguments": {"task": "a"}},
                {"type": "toolCall", "id": "spawn-b", "name": "sessions_spawn", "arguments": {"task": "b"}},
                {"type": "toolCall", "id": "spawn-failed", "name": "sessions_spawn", "arguments": {"task": "never materialized"}},
            ],
        },
    }])
    _write_jsonl(sessions / "child-a.jsonl", [{
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:04Z",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "a done"}],
        },
    }])
    _write_jsonl(sessions / "child-b.jsonl", [
        {"type": "message", "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:06Z",
            "stopReason": "toolUse",
            "content": [],
        }},
        {"type": "message", "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:09Z",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "b done"}],
        }},
    ])

    result = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )
    obs = result["runtime_observability"]
    assert obs["spawn_attempt_count"] == 3
    assert obs["child_session_count"] == 2
    assert obs["unmaterialized_spawn_attempt_count"] == 1
    assert obs["spawn_runtime_policy"] == (
        "failed/unmaterialized spawn waits remain in agent_execution_seconds "
        "but never create child runtime"
    )
    assert obs["child_runtime_policy"]["failed_spawn_runtime"] is None
    assert obs["child_runtime_complete_count"] == 2
    assert obs["child_runtime_by_session"][child_a]["runtime_seconds"] == 3.0
    assert obs["child_runtime_by_session"][child_a]["start_timestamp_source"] == "createdAt"
    assert obs["child_runtime_by_session"][child_a]["terminal_timestamp_source"] == "endedAt"
    assert obs["child_runtime_by_session"][child_b]["runtime_seconds"] == 3.0
    assert obs["child_runtime_by_session"][child_b]["start_timestamp_quality"] == "transcript_inferred"
    assert obs["child_runtime_by_session"][child_b]["terminal_timestamp_quality"] == "transcript_inferred"
    assert obs["child_sum_runtime_seconds"] == 6.0
    assert obs["child_union_active_seconds"] == 6.0
    assert obs["child_overlap_wall_seconds"] == 0.0
    assert obs["child_concurrency_excess_seconds"] == 0.0
    assert obs["max_concurrent_children"] == 1

    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    completions = {
        event["session_key"]: event
        for event in events
        if event["event_type"] == "child_completion"
    }
    assert completions[child_a]["source_timestamp_field"] == "endedAt"
    assert completions[child_a]["timestamp_quality"] == "session_state"
    assert completions[child_a]["measurement_observed_timestamp_ns"] is not None
    assert "observed_timestamp_ns" not in completions[child_a]
    assert completions[child_b]["source_timestamp_field"] == "transcript_last_event"
    assert completions[child_b]["timestamp_quality"] == "transcript_inferred"


def test_three_child_overlap_distinguishes_wall_from_excess(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    index = {
        parent_key: {
            "sessionId": "parent-id",
            "status": "done",
        }
    }
    for name in ("a", "b", "c"):
        index[f"agent:roadmap_bench:subagent:{name}"] = {
            "sessionId": f"child-{name}-id",
            "spawnedBy": parent_key,
            "status": "done",
            "createdAt": "2026-01-01T00:00:00Z",
            "endedAt": "2026-01-01T00:00:10Z",
        }
    (sessions / "sessions.json").write_text(json.dumps(index))

    result = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )
    obs = result["runtime_observability"]

    assert obs["child_runtime_complete_count"] == 3
    assert obs["child_sum_runtime_seconds"] == 30.0
    assert obs["child_union_active_seconds"] == 10.0
    assert obs["child_overlap_wall_seconds"] == 10.0
    assert obs["child_concurrency_excess_seconds"] == 20.0
    assert obs["max_concurrent_children"] == 3
