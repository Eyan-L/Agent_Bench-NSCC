import json

from agent_bench.benchmarks.conditions.openclaw_measurement import (
    build_openclaw_measurement,
)
from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)
from agent_bench.core.manifest import RunManifest
from agent_bench.runtime.driver import _preserve_partial_agent_result


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


def test_freeze_error_preserves_execution_facts_without_fake_verifier():
    manifest = RunManifest(
        run_id="run",
        condition_id="condition",
        instance_id="instance",
        seed=42,
        serving_config_id="serving",
        status="ERROR",
    )
    lifecycle = {
        "terminalEvidence": "explicit_done",
        "workspaceQuiesced": False,
    }
    _preserve_partial_agent_result(
        manifest,
        {
            "agent_completion_status": "normal_completion",
            "agent_runtime_mode": "gateway",
            "agent_lifecycle": lifecycle,
            "agent_execution_seconds": 12.5,
            "phase_timing": {"agent_execution_seconds": 12.5},
            "trajectory_path": "/run/trajectory.jsonl",
            "runtime_observability_path": "/run/runtime_observability.json",
            "total_prompt_tokens": 100,
            "total_completion_tokens": 20,
        },
        failure_phase="candidate_freeze",
        failure_message="freeze refused",
    )

    assert manifest.status == "ERROR"
    assert manifest.agent_execution_outcome == "normal_completion"
    assert manifest.agent_runtime_mode == "gateway"
    assert manifest.agent_lifecycle == lifecycle
    assert manifest.agent_execution_seconds == 12.5
    assert manifest.failure_phase == "candidate_freeze"
    assert manifest.verifier_eligible is False
    assert manifest.verifier_status == "SKIPPED"
    assert manifest.verifier_skipped is True
    assert manifest.reward is None
    assert manifest.test_pass_rate is None

    restored = RunManifest.from_dict(manifest.to_dict())
    assert restored.failure_phase == "candidate_freeze"
    assert restored.verifier_status == "SKIPPED"
    assert restored.verifier_skipped is True
    assert restored.agent_execution_outcome == "normal_completion"


def test_explicit_parent_and_runtime_waits_are_measured_separately(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:explicit:run"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": "parent-id",
            "sessionFile": "parent.jsonl",
            "status": "done",
        },
    }))
    _write_jsonl(sessions / "parent.jsonl", [
        {
            "type": "message",
            "timestamp": "1970-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "timestamp": "1970-01-01T00:00:02Z",
                "responseId": "initial",
                "stopReason": "toolUse",
                "usage": {"input": 10, "output": 2},
                "content": [],
            },
        },
        {
            "type": "custom_message",
            "customType": "openclaw.sessions_yield",
            "timestamp": "1970-01-01T00:00:03Z",
        },
        {
            "type": "message",
            "timestamp": "1970-01-01T00:00:09Z",
            "message": {
                "role": "user",
                "timestamp": "1970-01-01T00:00:09Z",
                "content": [{"type": "text", "text": "children settled"}],
            },
        },
        {
            "type": "message",
            "timestamp": "1970-01-01T00:00:10Z",
            "message": {
                "role": "assistant",
                "timestamp": "1970-01-01T00:00:10Z",
                "responseId": "final",
                "stopReason": "stop",
                "usage": {"input": 20, "output": 3},
                "content": [{"type": "text", "text": "DONE"}],
            },
        },
    ])
    _write_jsonl(tmp_path / "spans.jsonl", [
        {
            "name": "openclaw.model.usage",
            "start_unix_nano": 2_000_000_000,
            "end_unix_nano": 4_000_000_000,
            "duration_ms": 2000,
            "attrs": {"openclaw.span.source": "raw_response_log"},
        },
        {
            "name": "openclaw.tool.call",
            "start_unix_nano": 5_000_000_000,
            "end_unix_nano": 6_000_000_000,
            "duration_ms": 1000,
            "attrs": {"openclaw.tool.name": "exec"},
        },
    ])

    result = build_openclaw_measurement(
        tmp_path,
        agent_id="roadmap_bench",
        execution_start_ns=1_000_000_000,
        execution_end_ns=11_000_000_000,
    )

    assert result["trajectory_tokens"]["parent"] == {
        "input": 30,
        "output": 5,
        "cache_read": 0,
        "turn_count": 2,
        "resumed_turns": 0,
    }
    timing = result["runtime_observability"]["execution_time"]
    assert timing["agent_end_to_end_seconds"] == 10.0
    assert timing["observed_operation_union_seconds"] == 3.0
    assert timing["observed_model_union_seconds"] == 2.0
    assert timing["observed_tool_union_seconds"] == 1.0
    assert timing["parent_yield_wait_seconds"] == 6.0
    assert timing["unattributed_orchestration_seconds"] == 7.0


def test_parent_timeline_and_resumes_use_durable_v9_transcript(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": "parent-id",
            "sessionFile": "parent.jsonl",
            "status": "running",
            # OpenClaw updates this field when it resumes the same session.
            "startedAt": 8_000,
        },
    }))
    initial = {
        "type": "message",
        "id": "initial-request",
        "timestamp": 1,
        "message": {
            "role": "user",
            "timestamp": 1,
            "content": "implement roadmap",
        },
    }
    first_resume = {
        "type": "message",
        "id": "resume-one",
        "timestamp": 4,
        "message": {
            "role": "user",
            "timestamp": 4,
            "content": "children settled",
        },
    }
    _write_jsonl(sessions / "parent.jsonl", [
        initial,
        {
            "type": "message",
            "timestamp": 2,
            "message": {
                "role": "assistant",
                "timestamp": 2,
                "responseId": "first-turn",
                "stopReason": "toolUse",
                "usage": {"input": 10, "output": 2},
                "content": [],
            },
        },
        {
            "type": "custom_message",
            "customType": "openclaw.sessions_yield",
            "timestamp": 3,
        },
        first_resume,
        # A durable replay has the same event identity and is not a new resume.
        first_resume,
        {
            "type": "message",
            "id": "heartbeat",
            "timestamp": 5,
            "message": {
                "role": "user",
                "timestamp": 5,
                "content": "[OpenClaw heartbeat poll]",
                "provenance": {
                    "kind": "internal_system",
                    "sourceTool": "heartbeat",
                },
            },
        },
        {
            "type": "message",
            "id": "resume-two",
            "timestamp": 7,
            "message": {
                "role": "user",
                "timestamp": 7,
                # Identical text is a distinct durable event in v9.
                "content": "children settled",
            },
        },
        {
            "type": "message",
            "timestamp": 9,
            "message": {
                "role": "assistant",
                "timestamp": 9,
                "responseId": "final-turn",
                "stopReason": "stop",
                "usage": {"input": 20, "output": 3},
                "content": "DONE\nIntegrated.",
            },
        },
    ])

    result = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )

    assert result["trajectory_tokens"]["parent"]["resumed_turns"] == 2
    parent = result["runtime_observability"]["agent_timeline"]["agents"][0]
    assert parent["role"] == "parent"
    assert parent["start_timestamp_ns"] == 1_000_000_000
    assert parent["start_timestamp_source"] == "transcript_first_event"
    assert parent["end_timestamp_ns"] == 9_000_000_000
    assert parent["end_timestamp_source"] == "transcript_explicit_done"
    assert parent["duration_seconds"] == 8.0
    assert parent["runtime_censored"] is False


def test_markdown_done_is_observed_without_relaxing_strict_done(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": "parent-id",
            "sessionFile": "parent.jsonl",
            "status": "done",
            "createdAt": "2026-01-01T00:00:01Z",
            "endedAt": "2026-01-01T00:00:02Z",
        },
    }))
    _write_jsonl(sessions / "parent.jsonl", [{
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:02Z",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "**DONE**"}],
        },
    }])

    obs = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )["runtime_observability"]

    assert obs["strict_done_marker_count"] == 0
    assert obs["markdown_done_declaration_count"] == 1
    assert obs["done_marker_policy"]["markdown_done_declaration"] == (
        "measurement-only declaration"
    )


def test_roadmap_quiescence_timing_separates_parent_terminal(monkeypatch, tmp_path):
    class Harness:
        def quiesce_orphan_children(self, _env_info):
            return {
                "workspaceQuiesced": True,
                "quiescenceMethod": "stop-agent-and-restart-task-runtime",
                "openClawRuntimeStopped": True,
                "execBridgeStopped": True,
                "taskRuntimeRestarted": True,
            }

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"_harness": Harness(), "state_dir": str(tmp_path)}
    snapshot = {
        "parent_session_id": "parent",
        "parent_terminal": True,
        "parent_session_terminal": True,
        "parent_session_status": "done",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "pending_child_sessions": [],
    }
    monkeypatch.setattr(
        condition, "_read_yield_snapshot", lambda _sessions_dir: snapshot
    )
    clock = iter([1_000_000_000, 3_000_000_000])
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.time_ns",
        lambda: next(clock),
    )
    timed_snapshot, quiescence = condition._quiesce_with_timing(
        snapshot, logical_parent_terminal=True
    )
    assert quiescence["workspaceQuiesced"] is True
    assert quiescence["agentWriterQuiescenceAttempted"] is True
    assert quiescence["terminalFenceStable"] is True
    assert condition._env_info["roadmap_workspace_quiesced"] is True
    assert timed_snapshot["parent_terminal_observed_ns"] == 1_000_000_000
    assert timed_snapshot["workspace_quiesced_ns"] == 3_000_000_000


def test_roadmap_terminal_fence_rejects_late_parent_activity(monkeypatch, tmp_path):
    class Harness:
        def quiesce_orphan_children(self, _env_info):
            return {
                "workspaceQuiesced": True,
                "quiescenceMethod": "stop-agent-and-restart-task-runtime",
            }

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"_harness": Harness(), "state_dir": str(tmp_path)}
    snapshot = {
        "parent_session_id": "parent",
        "parent_terminal": True,
        "parent_session_terminal": True,
        "parent_session_status": "done",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "pending_child_sessions": [],
    }
    changed = dict(
        snapshot,
        message_count=2,
        transcript_size_bytes=20,
        latest_parent_event="resume",
    )
    monkeypatch.setattr(
        condition, "_read_yield_snapshot", lambda _sessions_dir: changed
    )

    timed_snapshot, quiescence = condition._quiesce_with_timing(
        snapshot, logical_parent_terminal=True
    )

    assert RoadmapBenchOpenClawCondition.terminal_stop_confirmation_seconds == 4.0
    assert quiescence["workspaceQuiesced"] is False
    assert quiescence["terminalFenceStable"] is False
    assert condition._env_info["roadmap_workspace_quiesced"] is False
    assert "activity changed" in quiescence["lifecycleError"]
    assert "workspace_quiesced_ns" not in timed_snapshot


def test_roadmap_terminal_fence_rejects_status_only_resume(monkeypatch, tmp_path):
    class Harness:
        def quiesce_orphan_children(self, _env_info):
            return {
                "workspaceQuiesced": True,
                "quiescenceMethod": "stop-agent-and-restart-task-runtime",
            }

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"_harness": Harness(), "state_dir": str(tmp_path)}
    snapshot = {
        "parent_session_id": "parent",
        "parent_terminal": True,
        "parent_session_terminal": True,
        "parent_session_status": "done",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "pending_child_sessions": [],
    }
    changed = dict(
        snapshot,
        parent_session_status="running",
        parent_session_terminal=False,
        parent_terminal=False,
    )
    monkeypatch.setattr(
        condition, "_read_yield_snapshot", lambda _sessions_dir: changed
    )

    timed_snapshot, quiescence = condition._quiesce_with_timing(
        snapshot, logical_parent_terminal=True
    )

    assert quiescence["workspaceQuiesced"] is False
    assert quiescence["terminalFenceStable"] is False
    assert condition._env_info["roadmap_workspace_quiesced"] is False
    assert "activity changed" in quiescence["lifecycleError"]
    assert "workspace_quiesced_ns" not in timed_snapshot


def test_roadmap_terminal_fence_allows_safe_running_to_done_convergence(
    monkeypatch, tmp_path
):
    class Harness:
        def quiesce_orphan_children(self, _env_info):
            return {
                "workspaceQuiesced": True,
                "quiescenceMethod": "stop-agent-and-restart-task-runtime",
                "openClawRuntimeStopped": True,
                "execBridgeStopped": True,
                "taskRuntimeRestarted": True,
            }

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"_harness": Harness(), "state_dir": str(tmp_path)}
    snapshot = {
        "parent_session_key": "agent:roadmap_bench:explicit:run",
        "parent_session_id": "parent",
        "parent_last_run_id": "parent-run",
        "parent_terminal": True,
        "parent_session_terminal": False,
        "parent_session_status": "running",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "latest_assistant_stop_reason": "stop",
        "done_text": "DONE",
        "pending_child_sessions": [],
    }
    settled = dict(
        snapshot,
        parent_session_status="done",
        parent_session_terminal=True,
    )
    monkeypatch.setattr(
        condition, "_read_yield_snapshot", lambda _sessions_dir: settled
    )

    timed_snapshot, quiescence = condition._quiesce_with_timing(
        snapshot, logical_parent_terminal=True
    )

    assert quiescence["workspaceQuiesced"] is True
    assert quiescence["terminalFenceStable"] is True
    assert quiescence["terminalFenceDecision"] == "safe_status_convergence"
    assert quiescence["terminalFenceBefore"]["parent_session_status"] == "running"
    assert quiescence["terminalFenceAfter"]["parent_session_status"] == "done"
    assert condition._env_info["roadmap_workspace_quiesced"] is True
    assert "workspace_quiesced_ns" in timed_snapshot


def test_roadmap_terminal_fence_rejects_writer_stop_failure(monkeypatch, tmp_path):
    class Harness:
        def quiesce_orphan_children(self, _env_info):
            return {
                "workspaceQuiesced": True,
                "openClawRuntimeStopped": False,
                "execBridgeStopped": True,
                "taskRuntimeRestarted": True,
            }

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"_harness": Harness(), "state_dir": str(tmp_path)}
    snapshot = {
        "parent_session_key": "agent:roadmap_bench:explicit:run",
        "parent_session_id": "parent",
        "parent_last_run_id": "parent-run",
        "parent_terminal": True,
        "parent_session_terminal": True,
        "parent_session_status": "done",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "latest_assistant_stop_reason": "stop",
        "done_text": "DONE",
        "pending_child_sessions": [],
    }
    monkeypatch.setattr(
        condition, "_read_yield_snapshot", lambda _sessions_dir: snapshot
    )

    timed_snapshot, quiescence = condition._quiesce_with_timing(
        snapshot, logical_parent_terminal=True
    )

    assert quiescence["workspaceQuiesced"] is False
    assert quiescence["terminalFenceStable"] is False
    assert quiescence["terminalFenceDecision"] == "writer_stop_failed"
    assert "workspace_quiesced_ns" not in timed_snapshot


def test_roadmap_terminal_fence_rejects_changed_execution_identity():
    before = {
        "parent_session_key": "agent:roadmap_bench:explicit:run",
        "parent_session_id": "parent",
        "parent_last_run_id": "run-a",
        "parent_session_status": "running",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "latest_assistant_stop_reason": "stop",
        "done_text": "DONE",
    }
    after = dict(
        before,
        parent_last_run_id="run-b",
        parent_session_status="done",
    )

    stable, decision = RoadmapBenchOpenClawCondition._terminal_fence_decision(
        before, after, logical_parent_terminal=True
    )

    assert stable is False
    assert decision == "parent_execution_identity_changed"


def test_roadmap_terminal_fence_allows_orphan_child_state_convergence():
    before = {
        "parent_session_key": "agent:roadmap_bench:explicit:run",
        "parent_session_id": "parent",
        "parent_last_run_id": "run",
        "parent_session_status": "running",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "latest_assistant_stop_reason": "stop",
        "done_text": "DONE",
        "pending_children": 1,
        "child_states": [{"session_id": "child", "status": "running"}],
    }
    after = dict(
        before,
        parent_session_status="done",
        pending_children=0,
        child_states=[{"session_id": "child", "status": "killed"}],
    )

    stable, decision = RoadmapBenchOpenClawCondition._terminal_fence_decision(
        before, after, logical_parent_terminal=True
    )

    assert stable is True
    assert decision == "safe_status_convergence"


def test_roadmap_terminal_fence_requires_verified_writer_stop(monkeypatch, tmp_path):
    class Harness:
        def quiesce_orphan_children(self, _env_info):
            return {
                "workspaceQuiesced": True,
                "openClawRuntimeStopped": True,
                "execBridgeStopped": True,
                # Missing taskRuntimeRestarted is not proof of a write fence.
            }

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"_harness": Harness(), "state_dir": str(tmp_path)}
    snapshot = {
        "parent_session_key": "agent:roadmap_bench:explicit:run",
        "parent_session_id": "parent",
        "parent_last_run_id": "run",
        "parent_terminal": True,
        "parent_session_terminal": True,
        "parent_session_status": "done",
        "message_count": 1,
        "transcript_size_bytes": 10,
        "yield_count": 0,
        "latest_parent_event": "assistant_stop",
        "latest_assistant_stop_reason": "stop",
        "done_text": "DONE",
        "pending_child_sessions": [],
    }
    monkeypatch.setattr(
        condition, "_read_yield_snapshot", lambda _sessions_dir: snapshot
    )

    timed_snapshot, quiescence = condition._quiesce_with_timing(
        snapshot, logical_parent_terminal=True
    )

    assert quiescence["workspaceQuiesced"] is False
    assert quiescence["terminalFenceDecision"] == "writer_stop_unverified"
    assert "workspace_quiesced_ns" not in timed_snapshot


def test_roadmap_candidate_freeze_requires_validated_quiescence(tmp_path):
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition.container = None
    env_info = {
        "roadmap_workspace_baseline": {"git_head": "deadbeef"},
        "workspace_host_path": str(tmp_path),
        "roadmap_workspace_quiesced": False,
    }

    try:
        condition._collect_artifacts(
            None, env_info, tmp_path, "/app"
        )
    except RuntimeError as exc:
        assert "validated agent-writer quiescence" in str(exc)
    else:
        raise AssertionError("candidate freeze must fail closed before quiescence")


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
    assert obs["child_runtime_by_session"][child_b]["start_timestamp_quality"] == "durable_transcript_creation"
    assert obs["child_runtime_by_session"][child_b]["terminal_timestamp_quality"] == "transcript_terminal_stop"
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
    assert completions[child_b]["source_timestamp_field"] == "transcript_final_assistant"
    assert completions[child_b]["timestamp_quality"] == "transcript_terminal_stop"


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


def test_nonterminal_children_are_lower_bounds_not_precise_concurrency(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    index = {
        parent_key: {
            "sessionId": "parent-id",
            "status": "done",
            "startedAt": "2026-01-01T00:00:00Z",
            "endedAt": "2026-01-01T00:00:09Z",
        }
    }
    for name, second in (("a", 2), ("b", 4), ("c", 6)):
        index[f"agent:roadmap_bench:subagent:{name}"] = {
            "sessionId": f"child-{name}-id",
            "spawnedBy": parent_key,
            "status": "unknown",
            "label": f"target-{name}",
            "createdAt": f"2026-01-01T00:00:0{second}Z",
        }
    (sessions / "sessions.json").write_text(json.dumps(index))
    quiesced_ns = 1_767_225_610_000_000_000
    (tmp_path / "openclaw_response.json").write_text(json.dumps({
        "result": {
            "meta": {
                "agentBenchLifecycle": {
                    "workspaceQuiescedNs": quiesced_ns,
                }
            }
        }
    }))

    result = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )
    obs = result["runtime_observability"]

    assert obs["child_session_count"] == 3
    assert obs["child_runtime_complete_count"] == 0
    assert obs["child_runtime_unknown_count"] == 3
    assert obs["child_runtime_observed_count"] == 0
    assert obs["child_sum_runtime_seconds"] == 0.0
    assert obs["child_sum_runtime_lower_bound_seconds"] == 0.0
    assert obs["child_union_active_seconds"] == 0.0
    assert obs["child_overlap_wall_seconds"] == 0.0
    assert obs["child_concurrency_excess_seconds"] == 0.0
    assert obs["max_concurrent_children"] == 0
    assert obs["max_concurrent_children_lower_bound"] == 0

    child = obs["child_runtime_by_session"][
        "agent:roadmap_bench:subagent:a"
    ]
    assert child["runtime_seconds"] is None
    assert child["runtime_lower_bound_seconds"] is None
    assert child["runtime_complete"] is False
    assert child["runtime_censored"] is True
    assert child["terminal_timestamp"] is None
    assert child["interval_end_timestamp_ns"] is None
    assert child["interval_end_timestamp_source"] is None

    timeline = obs["agent_timeline"]
    assert timeline["observation_end_timestamp_ns"] == quiesced_ns
    assert [agent["role"] for agent in timeline["agents"]] == ["parent"]
    assert sum(
        bool(agent["runtime_censored"]) for agent in timeline["agents"]
    ) == 0


def test_child_continuation_started_at_does_not_truncate_lifecycle(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    child_key = "agent:roadmap_bench:subagent:child"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": "parent-id",
            "status": "done",
        },
        child_key: {
            "sessionId": "child-id",
            "sessionFile": "child.jsonl",
            "spawnedBy": parent_key,
            "status": "done",
            "createdAt": "2026-01-01T00:00:02Z",
            # OpenClaw refreshes this when continuing the same child.
            "startedAt": "2026-01-01T00:00:08Z",
            "endedAt": "2026-01-01T00:00:10Z",
        },
    }))
    _write_jsonl(sessions / "child.jsonl", [{
        "type": "session",
        "timestamp": "2026-01-01T00:00:02.100Z",
    }, {
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:09Z",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "done"}],
        },
    }])

    obs = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )["runtime_observability"]
    child = obs["child_runtime_by_session"][child_key]

    assert child["start_timestamp_ns"] == 1_767_225_602_000_000_000
    assert child["start_timestamp_source"] == "createdAt"
    assert child["runtime_seconds"] == 8.0
    assert child["start_timestamp_conflict"] is True
    assert "startedAt" in child["start_timestamp_evidence"]


def test_child_counts_distinguish_online_and_restored_archive_projection(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {"sessionId": "parent-id", "status": "done"},
        "agent:roadmap_bench:subagent:online": {
            "sessionId": "online-id",
            "spawnedBy": parent_key,
            "status": "done",
            "createdAt": "2026-01-01T00:00:01Z",
            "endedAt": "2026-01-01T00:00:02Z",
            "agentBenchProjectionSource": "online_index",
        },
        "agent:roadmap_bench:subagent:archived": {
            "sessionId": "archived-id",
            "spawnedBy": parent_key,
            "status": "done",
            "createdAt": "2026-01-01T00:00:03Z",
            "endedAt": "2026-01-01T00:00:04Z",
            "agentBenchProjectionSource": "restored_archived_transcript",
        },
    }))

    obs = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )["runtime_observability"]

    assert obs["materialized_child_session_count"] == 2
    assert obs["online_index_child_count_at_archive"] == 1
    assert obs["restored_archived_child_count"] == 1
    assert obs["child_count_source_quality"] == "projection_sources_exact"


def test_unknown_child_has_lower_bound_but_no_precise_runtime(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    child_key = "agent:roadmap_bench:subagent:child"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": "parent-id",
            "status": "done",
        },
        child_key: {
            "sessionId": "child-id",
            "sessionFile": "child.jsonl",
            "spawnedBy": parent_key,
            "status": "unknown",
            "createdAt": "2026-01-01T00:00:02Z",
            "updatedAt": "2026-01-01T00:00:03Z",
        },
    }))
    _write_jsonl(sessions / "child.jsonl", [{
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:05Z",
            "stopReason": "toolUse",
            "content": [],
        },
    }])
    (tmp_path / "openclaw_response.json").write_text(json.dumps({
        "result": {"meta": {"agentBenchLifecycle": {
            "workspaceQuiescedNs": 1_767_225_610_000_000_000,
        }}}
    }))

    obs = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )["runtime_observability"]
    child = obs["child_runtime_by_session"][child_key]

    assert child["runtime_seconds"] is None
    assert child["runtime_complete"] is False
    assert child["runtime_censored"] is True
    assert child["runtime_lower_bound_seconds"] == 3.0
    assert child["last_observed_timestamp_source"] == "transcript_last_event"
    assert child["interval_end_timestamp_ns"] is None
    assert obs["child_runtime_unknown_count"] == 1
    assert obs["child_sum_runtime_seconds"] == 0.0
    assert obs["child_sum_runtime_lower_bound_seconds"] == 3.0
    assert obs["child_derived_metrics_quality"] == "partial_excludes_unknown"


def test_matched_gateway_run_failure_is_child_terminal_evidence(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    child_key = "agent:roadmap_bench:subagent:child"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {"sessionId": "parent-id", "status": "done"},
        child_key: {
            "sessionId": "child-id",
            "sessionFile": "child.jsonl",
            "spawnedBy": parent_key,
            "status": "unknown",
            "createdAt": "2026-01-01T00:00:02Z",
        },
    }))
    _write_jsonl(sessions / "child.jsonl", [{
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:04Z",
            "stopReason": "error",
            "content": [],
        },
    }])
    _write_jsonl(tmp_path / "conversation" / "cache-trace.jsonl", [{
        "runId": "child-run",
        "sessionId": "child-id",
        "sessionKey": child_key,
        "stage": "session:loaded",
    }])
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-01-01T00:00:05Z [sqlite/transaction] lock wait failed\n"
        "2026-01-01T00:00:06Z [agent/embedded] embedded run agent end: "
        "runId=child-run isError=true model=model provider=provider "
        "rawError=database is locked\n"
    )

    obs = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )["runtime_observability"]
    child = obs["child_runtime_by_session"][child_key]

    assert child["observed_status"] == "unknown"
    assert child["status"] == "failed"
    assert child["terminal_timestamp_ns"] == 1_767_225_606_000_000_000
    assert child["terminal_timestamp_source"] == "gateway_embedded_run_error"
    assert child["runtime_seconds"] == 4.0
    assert child["terminal_evidence"]["run_id"] == "child-run"


def test_child_activity_after_gateway_error_keeps_terminal_unknown(tmp_path):
    sessions = tmp_path / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:main"
    child_key = "agent:roadmap_bench:subagent:child"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {"sessionId": "parent-id", "status": "done"},
        child_key: {
            "sessionId": "child-id",
            "sessionFile": "child.jsonl",
            "spawnedBy": parent_key,
            "status": "unknown",
            "createdAt": "2026-01-01T00:00:02Z",
        },
    }))
    _write_jsonl(sessions / "child.jsonl", [{
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:04Z",
            "stopReason": "error",
            "content": [],
        },
    }, {
        "type": "message",
        "message": {
            "role": "assistant",
            "timestamp": "2026-01-01T00:00:07Z",
            "stopReason": "toolUse",
            "content": [{"type": "toolCall", "name": "read", "id": "call"}],
        },
    }])
    _write_jsonl(tmp_path / "conversation" / "cache-trace.jsonl", [{
        "runId": "failed-run",
        "sessionId": "child-id",
        "sessionKey": child_key,
    }])
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-01-01T00:00:05Z [agent/embedded] embedded run agent end: "
        "runId=failed-run isError=true rawError=database is locked\n"
    )

    obs = build_openclaw_measurement(
        tmp_path, agent_id="roadmap_bench"
    )["runtime_observability"]
    child = obs["child_runtime_by_session"][child_key]

    assert child["observed_status"] == "unknown"
    assert child["status"] == "unknown"
    assert child["terminal_timestamp_ns"] is None
    assert child["runtime_seconds"] is None
    assert child["runtime_lower_bound_seconds"] == 5.0
