import json

from agent_bench.benchmarks.conditions.openclaw_measurement import (
    append_trajectory_event,
    build_openclaw_measurement,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _usage_span(session_key, session_id, start_ns, input_tokens, output_tokens):
    return {
        "name": "openclaw.model.usage",
        "start_unix_nano": start_ns,
        "duration_ms": 10000,
        "attrs": {
            "openclaw.sessionKey": session_key,
            "openclaw.sessionId": session_id,
            "openclaw.tokens.input": str(input_tokens),
            "openclaw.tokens.output": str(output_tokens),
        },
    }


def test_openclaw_measurement_captures_trajectory_and_both_token_layers(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "sessions.json").write_text(json.dumps({
        "agent:roadmap_bench:main": {
            "sessionId": "parent-id",
            "sessionFile": "/root/.openclaw/agents/roadmap_bench/sessions/parent.jsonl",
            "status": "done",
        },
        "agent:roadmap_bench:subagent:child": {
            "sessionId": "child-id",
            "sessionFile": "child.jsonl",
            "spawnedBy": "agent:roadmap_bench:main",
            "status": "done",
        },
    }))
    _write_jsonl(sessions / "parent.jsonl", [
        {"type": "message", "message": {
            "role": "user", "timestamp": 0,
            "content": [{"type": "text", "text": "implement roadmap"}],
        }},
        {"type": "message", "message": {
            "role": "assistant", "timestamp": 1,
            "stopReason": "toolUse", "content": [{
                "type": "toolCall", "id": "spawn-1",
                "name": "sessions_spawn", "arguments": {"task": "child work"},
            }],
        }},
        {"type": "custom_message", "customType": "openclaw.sessions_yield", "timestamp": 2},
        {"type": "message", "message": {
            "role": "user", "timestamp": 3,
            "content": [{"type": "text", "text": "child completed"}],
        }},
        {"type": "message", "message": {
            "role": "assistant", "timestamp": 4, "stopReason": "stop",
            "content": [{"type": "text", "text": "DONE\nIntegrated."}],
        }},
    ])
    _write_jsonl(sessions / "child.jsonl", [{
        "type": "message", "message": {
            "role": "assistant", "timestamp": 3, "stopReason": "stop",
            "content": [{"type": "text", "text": "child result"}],
        },
    }])

    parent_key = "agent:roadmap_bench:main"
    child_key = "agent:roadmap_bench:subagent:child"
    _write_jsonl(tmp_path / "spans.jsonl", [
        _usage_span(parent_key, "parent-id", 1_000_000_000, 10, 2),
        # Diagnostics can emit a duplicate span for the same logical turn.
        _usage_span(parent_key, "parent-id", 1_500_000_000, 10, 2),
        _usage_span(parent_key, "parent-id", 10_000_000_000, 20, 3),
        _usage_span(child_key, "child-id", 2_000_000_000, 7, 4),
    ])
    _write_jsonl(tmp_path / "conversation" / "llm-raw-responses.jsonl", [
        {
            "ts": "2026-01-01T00:00:01Z", "httpStatus": 200,
            "model": "model", "response": {"usage": {
                "prompt_tokens": 100, "completion_tokens": 10,
            }}, "metrics": {"total_ms": 50},
        },
        {
            "ts": "2026-01-01T00:00:02Z", "httpStatus": 200,
            "model": "model", "response": {"usage": {
                "prompt_tokens": 200, "completion_tokens": 20,
            }}, "metrics": {"total_ms": 80},
        },
    ])
    (tmp_path / "openclaw_gateway.log").write_text(
        "1970-01-01T00:00:01.500Z [diagnostic] stuck session: "
        "sessionKey=agent:roadmap_bench:main state=processing age=150s\n"
        "2026-01-01T00:00:03Z [warn] Subagent announce completion direct "
        "announce agent call transient failure, retrying 2/4 in 5s: "
        "gateway timeout after 120000ms\n"
        "2026-01-01T00:00:04Z [ws] res ✓ agent 57ms "
        "runId=announce:v1:child\n"
        "2026-01-01T00:00:06Z [agent] context-overflow-diag compactionAttempts=1\n"
    )

    result = build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    logical = result["trajectory_tokens"]
    assert logical["parent"] == {
        "input": 30, "output": 5, "turn_count": 2, "resumed_turns": 1,
    }
    assert logical["children"][child_key] == {
        "input": 7, "output": 4, "turn_count": 1,
    }
    assert logical["total_logical_input"] == 37
    assert logical["total_logical_output"] == 9
    assert result["serving_tokens"] == {
        "processed_input": 300,
        "generated_output": 30,
        "request_count": 2,
    }
    observability = result["runtime_observability"]
    assert observability["announce_attempt_count"] == 2
    assert observability["announce_timeout_count"] == 1
    assert observability["announce_retry_count"] == 1
    assert observability["announce_success_count"] == 1
    assert observability["announce_latency_ms"] == [120000, 57]
    assert observability["stuck_session_count"] == 1
    assert observability["stuck_session_classification"] == {
        "active_operation": 1
    }

    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    event_types = {event["event_type"] for event in events}
    assert {
        "spawn", "yield", "resume", "assistant_stop", "done",
        "child_completion", "model_usage", "model_request",
        "announce", "announce_timeout", "stuck_session", "context_overflow",
        "compaction",
    } <= event_types
    assert all(
        {"timestamp", "session_id", "role", "event_type"} <= event.keys()
        for event in events
    )

    append_trajectory_event(tmp_path, "cleanup", status="completed")
    assert json.loads(
        (tmp_path / "trajectory.jsonl").read_text().splitlines()[-1]
    )["event_type"] == "cleanup"


def test_measurement_uses_durable_transcript_and_ignores_checkpoints(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "sessions.json").write_text(json.dumps({
        "agent:roadmap_bench:main": {
            "sessionId": "logical-parent",
            "sessionFile": (
                "/root/.openclaw/agents/roadmap_bench/sessions/physical.jsonl"
            ),
            "status": "done",
        },
    }))
    assistant = {
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": 1,
            "responseId": "response-1",
            "usage": {
                "input": 10,
                "cacheRead": 90,
                "output": 5,
            },
            "stopReason": "toolUse",
            "content": [{
                "type": "toolCall",
                "id": "call-1",
                "name": "exec",
                "arguments": {"command": "true"},
            }],
        },
    }
    # OpenClaw may replay a response in the durable file during recovery. A
    # stable responseId means it remains one logical model turn.
    _write_jsonl(sessions / "physical.jsonl", [assistant, assistant])
    _write_jsonl(
        sessions / "logical-parent.checkpoint.snapshot.jsonl", [assistant]
    )

    result = build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    assert result["trajectory_tokens"]["parent"] == {
        "input": 100,
        "output": 5,
        "cache_read": 90,
        "turn_count": 1,
        "resumed_turns": 0,
    }
    assert result["trajectory_tokens"]["children"] == {}
    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert sum(event["event_type"] == "tool_call" for event in events) == 1
    assert sum(event["event_type"] == "model_usage" for event in events) == 1


def test_stuck_observability_ignores_run_level_usage_span(tmp_path):
    span = _usage_span(
        "agent:roadmap_bench:main", "parent-id", 1_000_000_000, 10, 2
    )
    span["duration_ms"] = 600_000
    span["end_unix_nano"] = 601_000_000_000
    _write_jsonl(tmp_path / "spans.jsonl", [span])
    (tmp_path / "openclaw_gateway.log").write_text(
        "1970-01-01T00:05:00Z [diagnostic] stuck session: "
        "sessionKey=agent:roadmap_bench:main state=processing age=150s\n"
    )

    result = build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    assert result["runtime_observability"]["stuck_session_classification"] == {
        "unresolved": 1
    }


def test_announce_attempts_aggregate_across_children(tmp_path):
    child_a = "agent:roadmap_bench:subagent:a"
    child_b = "agent:roadmap_bench:subagent:b"
    run_a = f"announce:v1:{child_a}:run-a"
    run_b = f"announce:v1:{child_b}:run-b"
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-01-01T00:00:01Z [warn] Subagent announce completion "
        f"runId={run_a} transient failure, retrying 2/4: "
        "gateway timeout after 120000ms\n"
        "2026-01-01T00:00:02Z [ws] res ✓ agent 50ms "
        f"runId={run_a}\n"
        "2026-01-01T00:00:03Z [warn] Subagent announce completion "
        f"runId={run_b} transient failure, retrying 2/4: "
        "gateway timeout after 120000ms\n"
        "2026-01-01T00:00:04Z [ws] res ✓ agent 75ms "
        f"runId={run_b}\n"
    )

    result = build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")
    observability = result["runtime_observability"]

    assert observability["announce_attempt_count"] == 4
    assert observability["announce_timeout_count"] == 2
    assert observability["announce_retry_count"] == 2
    assert observability["announce_success_count"] == 2
    assert observability["announce_by_child"][child_a] == {
        "child_session_key": child_a,
        "announce_run_ids": [run_a],
        "attempt_count": 2,
        "success_count": 1,
        "timeout_count": 1,
        "retry_count": 1,
        "latency_ms": [120000, 50],
    }
    assert observability["announce_by_child"][child_b]["attempt_count"] == 2
