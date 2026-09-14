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


def test_post_compaction_loop_guard_is_machine_readable(tmp_path):
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-09-01T05:45:00Z [post-compaction-guard] "
        "compaction_loop_persisted: tool=read count=3\n"
    )

    result = build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    assert result["termination_reason"] == "compaction_loop_persisted"
    assert result["loop_detector"] == "compaction_loop_persisted"
    assert result["loop_tool"] == "read"
    assert result["loop_count"] == 3
    observability = result["runtime_observability"]
    assert observability["loop_detection_count"] == 1
    assert observability["loop_detections"] == [{
        "timestamp": "2026-09-01T05:45:00+00:00",
        "detector": "compaction_loop_persisted",
        "tool_name": "read",
        "count": 3,
        "level": None,
        "action": None,
        "terminal": True,
    }]

    trajectory = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    loop_events = [
        event for event in trajectory if event["event_type"] == "loop_detected"
    ]
    assert len(loop_events) == 1
    assert loop_events[0]["detector"] == "compaction_loop_persisted"
    assert loop_events[0]["tool_name"] == "read"
    assert loop_events[0]["count"] == 3


def test_gateway_compaction_event_requires_positive_count(tmp_path):
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-09-01T05:45:00Z [agent] run complete compactions=0\n"
        "2026-09-01T05:46:00Z [agent] context-overflow-diag "
        "compactionAttempts=0\n"
        "2026-09-01T05:47:00Z [agent] run complete compactions=2\n"
    )

    build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    trajectory = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    compactions = [
        event for event in trajectory if event["event_type"] == "compaction"
    ]
    assert len(compactions) == 1
    assert compactions[0]["count"] == 2


def test_native_tool_loop_warning_is_observable_but_not_terminal(tmp_path):
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-09-10T03:08:02.168+08:00 [diagnostic] tool loop: "
        "sessionId=parent-id sessionKey=agent:roadmap_bench:explicit:parent-id "
        "tool=exec level=warning action=warn detector=generic_repeat count=10 "
        'message="WARNING: repeated call"\n'
    )

    result = build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    assert "termination_reason" not in result
    assert result["runtime_observability"]["loop_detection_count"] == 1
    assert result["runtime_observability"]["loop_detections"] == [{
        "timestamp": "2026-09-09T19:08:02.168000+00:00",
        "detector": "generic_repeat",
        "tool_name": "exec",
        "count": 10,
        "level": "warning",
        "action": "warn",
        "terminal": False,
    }]

    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    loop = [event for event in events if event["event_type"] == "loop_detected"]
    assert len(loop) == 1
    assert loop[0]["session_id"] == "parent-id"
    assert loop[0]["session_key"] == "agent:roadmap_bench:explicit:parent-id"
    assert loop[0]["terminal"] is False


def test_normal_run_has_no_loop_termination_metadata(tmp_path):
    result = build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    assert "termination_reason" not in result
    assert "loop_detector" not in result
    assert "loop_tool" not in result
    assert "loop_count" not in result
    assert result["runtime_observability"]["loop_detection_count"] == 0
    assert result["runtime_observability"]["loop_detections"] == []


def test_measurement_records_length_as_terminal_assistant_stop(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "sessions.json").write_text(json.dumps({
        "agent:roadmap_bench:main": {
            "sessionId": "parent-id",
            "sessionFile": "parent.jsonl",
            "status": "done",
        },
    }))
    _write_jsonl(sessions / "parent.jsonl", [{
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": 10,
            "stopReason": "length",
            "content": [{"type": "text", "text": "partial final text"}],
        },
    }])

    build_openclaw_measurement(tmp_path, agent_id="roadmap_bench")

    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    terminal = [event for event in events if event["event_type"] == "assistant_stop"]
    assert len(terminal) == 1
    assert terminal[0]["stop_reason"] == "length"
    assert terminal[0]["text"] == "partial final text"
    assert not any(event["event_type"] == "done" for event in events)


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
    assert observability["announce_attribution"][
        "unattributed_timeout_count"
    ] == 1
    assert "does not infer identity" in observability[
        "announce_attribution"
    ]["limitation"]
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
    assert observability["announce_attribution"][
        "unattributed_timeout_count"
    ] == 0
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


def test_heartbeat_is_not_resume_or_terminal_but_tokens_still_count(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)

    (sessions / "sessions.json").write_text(json.dumps({
        "agent:roadmap_bench:main": {
            "sessionId": "parent-id",
            "sessionFile": "parent.jsonl",
            "status": "running",
        }
    }))

    _write_jsonl(sessions / "parent.jsonl", [
        {
            "type": "message",
            "message": {
                "role": "user",
                "timestamp": 1,
                "content": [{
                    "type": "text",
                    "text": "implement roadmap",
                }],
            },
        },
        {
            "type": "custom_message",
            "customType": "openclaw.sessions_yield",
            "timestamp": 2,
        },
        {
            "type": "message",
            "message": {
                "role": "user",
                "timestamp": 3,
                "content": [{
                    "type": "text",
                    "text": (
                        "Read HEARTBEAT.md if it exists. "
                        "If nothing needs attention, reply HEARTBEAT_OK"
                    ),
                }],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "timestamp": 4,
                "responseId": "heartbeat-response",
                "stopReason": "stop",
                "content": [{
                    "type": "text",
                    "text": "HEARTBEAT_OK",
                }],
                "usage": {
                    "input": 10,
                    "output": 1,
                },
            },
        },
    ])

    result = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]

    heartbeat_events = [
        e["event_type"]
        for e in events
        if e["event_type"].startswith("heartbeat_")
    ]

    assert heartbeat_events == [
        "heartbeat_request",
        "heartbeat_response",
    ]

    assert not any(
        e["event_type"] == "resume"
        and "HEARTBEAT" in str(e.get("text") or "")
        for e in events
    )

    assert not any(
        e["event_type"] == "assistant_stop"
        and e.get("text") == "HEARTBEAT_OK"
        for e in events
    )

    parent = result["trajectory_tokens"]["parent"]
    assert parent["resumed_turns"] == 0
    assert parent["turn_count"] == 1
    assert parent["input"] == 10
    assert parent["output"] == 1
