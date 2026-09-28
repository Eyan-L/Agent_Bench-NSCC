"""Protocol, completion and concurrent native-turn measurement checks."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from agent_bench.harnesses.codex.condition import CodexAgentCondition, archive_state
from agent_bench.harnesses.codex.responses_adapter import from_chat, to_chat
from agent_bench.harnesses.codex.telemetry import api_response, summarize
from agent_bench.harnesses.codex.completion import parent_completion
from agent_bench.harnesses.codex.messages import assistant_phase, continuation_preamble
from scripts import compare_roadmap_runs
from scripts.roadmap_codex_telemetry import install_comparison
from agent_bench.benchmarks.roadmap_bench.conditions.runtime_policy import install_roadmap_gateway_only_policy
from agent_bench.runtime import driver


def test_namespace_and_freeform_roundtrip_retains_call_identity():
    tools = [{"type": "namespace", "name": "multi_agent_v1", "tools": [
        {"type": "function", "name": "spawn_agent", "parameters": {"type": "object", "properties": {}}},
    ]}, {"type": "custom", "name": "apply_patch", "description": "Apply a patch"}]
    request = {"model": "qwen", "tools": tools, "input": [
        {"type": "function_call", "namespace": "multi_agent_v1", "name": "spawn_agent", "call_id": "one", "arguments": '{"message":"work"}'},
        {"type": "custom_tool_call", "name": "apply_patch", "call_id": "two", "input": "*** Begin Patch\n*** End Patch"},
        {"type": "function_call_output", "call_id": "one", "output": '{"agent_id":"child"}'},
        {"type": "custom_tool_call_output", "call_id": "two", "output": "Success"},
    ]}
    chat, identities = to_chat(request, {})
    assert [c["id"] for c in chat["messages"][0]["tool_calls"]] == ["one", "two"]
    assert [m["tool_call_id"] for m in chat["messages"][1:]] == ["one", "two"]
    restored = from_chat({"choices": [{"finish_reason": "tool_calls", "message": {
        "tool_calls": chat["messages"][0]["tool_calls"], "content": None}}]}, identities, "qwen")
    assert restored["output"][0]["namespace"] == "multi_agent_v1"
    assert restored["output"][1]["type"] == "custom_tool_call"
    assert restored["output"][1]["input"] == request["input"][1]["input"]


def test_adapter_preserves_truncation_and_rejects_unsupported_inputs():
    response = from_chat({"choices": [{"finish_reason": "length", "message": {"content": "unfinished"}}]}, {}, "qwen")
    assert response["status"] == "incomplete"
    assert response["incomplete_details"]["reason"] == "max_output_tokens"
    with pytest.raises(ValueError, match="Unsupported"):
        to_chat({"model": "qwen", "input": [{"type": "message", "role": "user", "content": [{"type": "input_image"}]}]}, {})


def test_qwen_receives_one_initial_system_block_with_developer_priority():
    chat, _ = to_chat({"model": "qwen", "instructions": "base", "input": [
        {"role": "user", "content": "task"},
        {"role": "developer", "content": "environment instructions"},
        {"role": "assistant", "content": "reply"},
        {"role": "developer", "content": "later constraints"},
    ]}, {})
    assert chat["messages"][0] == {"role": "system", "content": "base\n\nenvironment instructions\n\nlater constraints"}
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]


def test_roadmap_accepts_native_codex_and_still_rejects_openclaw_embedded(monkeypatch):
    monkeypatch.setattr(driver, "_roadmap_benchmark_invalid_reason", lambda result: None)
    install_roadmap_gateway_only_policy()
    check = driver._roadmap_benchmark_invalid_reason
    assert check({"harness": "codex", "agent_runtime_mode": "codex_cli"}) is None
    assert check({"harness": "openclaw", "agent_runtime_mode": "gateway"}) is None
    assert check({"harness": "openclaw", "agent_runtime_mode": "embedded"})
    assert check({"harness": "codex", "agent_runtime_mode": "embedded"})


def test_cleanup_accepts_driver_positional_task_and_keyword_adapter(monkeypatch):
    from agent_bench.benchmarks.terminal_bench.codex.app import TerminalBenchCodexHarness
    from agent_bench.benchmarks.terminal_bench.app import TerminalBenchOpenClawHarness
    import agent_bench.benchmarks.terminal_bench.codex.app as app

    harness = TerminalBenchCodexHarness.__new__(TerminalBenchCodexHarness)
    harness.container = SimpleNamespace(backend_id="singularity")
    stopped, released, delegated = [], [], []
    monkeypatch.setattr(harness, "_best_effort_stop", lambda handle: stopped.append(handle.id))
    monkeypatch.setattr(app, "release_gateway_port", lambda port: released.append(port))
    monkeypatch.setattr(TerminalBenchOpenClawHarness, "cleanup", lambda self, container_id, **kwargs: delegated.append((container_id, kwargs)))
    harness.cleanup("task", adapter_container_id="adapter", adapter_port=12345, state_dir="state")
    assert stopped == ["adapter"]
    assert released == [12345]
    assert delegated == [("task", {"state_dir": "state"})]


def test_response_passthrough_retains_incomplete_status_and_usage():
    response = {"status": "incomplete", "usage": {"input_tokens": 12, "output_tokens": 4}}
    row = {"response_sse": "event: response.incomplete\ndata: " + json.dumps({"type": "response.incomplete", "response": response}) + "\n\n"}
    assert api_response(row) == response


def rollout(root, thread, parent, events):
    directory = root / "sessions/2026/09/28"
    directory.mkdir(parents=True, exist_ok=True)
    base = datetime.datetime(2026, 9, 28, tzinfo=datetime.timezone.utc)
    rows = [{"timestamp": base.isoformat(), "type": "session_meta", "payload": {
        "id": thread, "parent_thread_id": parent, "source": "exec" if not parent else "subagent", "cli_version": "0.157.1"}}]
    for seconds, kind, payload in events:
        rows.append({"timestamp": (base + datetime.timedelta(seconds=seconds)).isoformat(), "type": kind, "payload": payload})
    (directory / f"rollout-{thread}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_native_active_turns_exclude_idle_and_wait_is_not_parent_work(tmp_path):
    state = tmp_path / "codex_state"
    rollout(state, "parent", None, [
        (0, "event_msg", {"type": "task_started"}),
        (4, "response_item", {"type": "function_call", "name": "exec_command", "call_id": "work", "arguments": "{}"}),
        (5, "response_item", {"type": "function_call_output", "call_id": "work", "output": "ok"}),
        (5, "response_item", {"type": "function_call", "name": "wait_agent", "namespace": "multi_agent_v1", "call_id": "wait", "arguments": "{}"}),
        (12, "response_item", {"type": "function_call_output", "call_id": "wait", "output": "ok"}),
        (13, "event_msg", {"type": "task_complete"}),
    ])
    rollout(state, "a", "parent", [(t, "event_msg", {"type": event}) for t, event in
            [(2, "task_started"), (6, "task_complete"), (10, "task_started"), (12, "task_complete")]])
    rollout(state, "b", "parent", [(4, "event_msg", {"type": "task_started"}), (11, "event_msg", {"type": "task_complete"})])
    _, obs = summarize(tmp_path)
    assert obs["max_concurrent_children"] == 2
    assert obs["child_sum_runtime_seconds"] == 13
    assert obs["child_union_active_seconds"] == 10
    assert obs["child_overlap_wall_seconds"] == 3
    assert obs["parent_overlapping_tool_calls"] == 1
    assert obs["parent_overlapping_coordination_calls"] == 1
    record = SimpleNamespace(run_dir=tmp_path, manifest={})
    comparison = SimpleNamespace(_runtime_observability=compare_roadmap_runs._runtime_observability,
                                 _child_intervals=compare_roadmap_runs._child_intervals,
                                 record_to_dict=lambda record: {})
    install_comparison(comparison)
    turns = comparison._child_intervals(record)
    assert len(turns) == 3
    assert sum(t["runtime_seconds"] for t in turns) == 13
    row = comparison.record_to_dict(record)
    assert row["parent_activity_during_child_execution"] is True
    assert row["parent_overlapping_tool_calls"] == 1


def test_successful_spawn_without_rollout_is_reported_missing(tmp_path):
    rollout(tmp_path / "codex_state", "parent", None, [
        (1, "response_item", {"type": "function_call", "name": "spawn_agent", "namespace": "multi_agent_v1", "call_id": "spawn", "arguments": "{}"}),
        (2, "response_item", {"type": "function_call_output", "call_id": "spawn", "output": '{"agent_id":"missing"}'}),
    ])
    _, obs = summarize(tmp_path)
    assert obs["missing_child_session_ids"] == ["missing"]
    assert obs["child_runtime_unknown_count"] == 1
    assert obs["max_concurrent_children"] is None


def test_state_archive_excludes_cyclic_cache_and_auth(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".tmp").mkdir()
    (source / ".tmp/loop").symlink_to(source, target_is_directory=True)
    (source / "auth.json").write_text("secret")
    (source / "config.toml").write_text("secret")
    (source / "thread_history_1.sqlite").write_bytes(b"ledger")
    rollout(source, "parent", None, [])
    archive = tmp_path / "archive"
    archive_state(source, archive)
    assert not (archive / ".tmp").exists()
    assert not (archive / "auth.json").exists()
    assert not (archive / "config.toml").exists()
    assert (archive / "thread_history_1.sqlite").read_bytes() == b"ledger"
    assert list(archive.rglob("rollout-*.jsonl"))


@pytest.mark.parametrize("truncated, expected", [(False, "normal_completion"), (True, "blocked_incomplete")])
def test_rc_zero_needs_native_completion_and_untruncated_final(tmp_path, truncated, expected):
    source, state, output = tmp_path / "source", tmp_path / "state", tmp_path / "run"
    state.mkdir()
    rollout(source, "parent", None, [
        (0, "event_msg", {"type": "task_started"}),
        (1, "response_item", {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "Implementation complete"}]}),
        (2, "event_msg", {"type": "task_complete"}),
    ])
    (state / "codex_api.jsonl").write_text(json.dumps({"start_ns": 1, "end_ns": 2, "thread_id": "parent",
        "turn_id": "rollout_turn_0", "response": {"status": "incomplete" if truncated else "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Implementation complete"}]}]}}) + "\n")

    class Container:
        backend_id = "singularity"

        def popen_exec(self, _handle, argv, **_kwargs):
            assert argv[0] == "sh" and "prompt.txt" in argv[2]
            assert (source / "prompt.txt").read_text() == "task"
            return subprocess.Popen([sys.executable, "-c", 'print(\'{"type":"turn.started"}\\n{"type":"turn.completed"}\')'],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    class Harness:
        def quiesce_agent_writers(self, _env):
            return {"workspaceQuiesced": True, "workspaceQuiescedNs": time.time_ns()}

    condition = CodexAgentCondition(None, SimpleNamespace(max_time_seconds=10), Container())
    result = condition.run({"instruction": "task"}, output, 42, "/app", "task", {
        "agent_env": {}, "state_dir": str(state), "codex_state_dir": str(source), "_harness": Harness()})
    assert result["agent_execution_outcome"] == expected
    assert result["phase_timing"]["agent_execution_end_ns"] == result["agent_lifecycle"]["workspaceQuiescedNs"]


def final_message(text, turn="one", identity="final"):
    return {"type": "message", "id": identity, "role": "assistant", "phase": "final_answer",
            "content": [{"type": "output_text", "text": text}],
            "internal_chat_message_metadata_passthrough": {"turn_id": turn}}


def complete_turn(text="DONE", turn="one", start=0):
    return [(start, "event_msg", {"type": "task_started", "turn_id": turn}),
            (start + 1, "response_item", final_message(text, turn)),
            (start + 2, "event_msg", {"type": "task_complete", "turn_id": turn})]


def completion_case(root, events, *, final="DONE", status="completed", turn="one", stdout_turns=1, api=True):
    rollout(root / "codex_state", "parent", None, events)
    if api:
        (root / "codex_api.jsonl").write_text(json.dumps({"start_ns": 1, "end_ns": 2, "thread_id": "parent",
            "turn_id": turn, "response": {"status": status, "output": [final_message(final, turn)]}}) + "\n")
    stdout = [{"type": "thread.started", "thread_id": "parent"}]
    for _ in range(stdout_turns):
        stdout.extend([{"type": "turn.started"}, {"type": "turn.completed"}])
    return stdout


def evidence_for(root, stdout):
    from agent_bench.harnesses.codex.telemetry import jsonl
    sessions, obs = summarize(root)
    return sessions, obs, parent_completion(sessions, stdout, list(jsonl(root / "codex_api.jsonl")))


@pytest.mark.parametrize("second", [
    [(3, "event_msg", {"type": "task_started", "turn_id": "two"})],
    complete_turn("", "two", 3), complete_turn("   \n", "two", 3),
    [(3, "event_msg", {"type": "task_started", "turn_id": "two"}),
     (4, "event_msg", {"type": "task_complete", "turn_id": "two"})],
])
def test_old_done_and_terminal_are_not_reused_across_turns(tmp_path, second):
    stdout = completion_case(tmp_path, complete_turn() + second, stdout_turns=2)
    sessions, _, evidence = evidence_for(tmp_path, stdout)
    assert not evidence["complete"]
    assert not evidence["done_marker"]
    assert not (sessions[0]["final_assistant"] or "").strip()


def test_previous_turn_end_cannot_close_the_new_turn(tmp_path):
    stdout = completion_case(tmp_path, complete_turn() + [
        (3, "event_msg", {"type": "task_started", "turn_id": "two"}),
        (4, "event_msg", {"type": "task_complete", "turn_id": "one"})], stdout_turns=2)
    sessions, _, evidence = evidence_for(tmp_path, stdout)
    assert sessions[0]["latest_turn"]["is_open"] is True
    assert sessions[0]["terminal_event"] is None
    assert not evidence["complete"]


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_final_message_replaces_done_even_in_same_turn(tmp_path, blank):
    events = complete_turn()[:2] + [(1.5, "response_item", final_message(blank)), complete_turn()[-1]]
    stdout = completion_case(tmp_path, events, final=blank)
    _, _, evidence = evidence_for(tmp_path, stdout)
    assert not evidence["complete"] and not evidence["done_marker"]


@pytest.mark.parametrize("mismatch", ["old_api", "old_cli", "failed_api", "mismatched_item", "pending_api"])
def test_final_turn_needs_its_own_consistent_protocol_evidence(tmp_path, mismatch):
    events = complete_turn("old answer") + complete_turn("new answer", "two", 3)
    stdout = completion_case(tmp_path, events, final="new answer", turn="two", stdout_turns=2)
    path = tmp_path / "codex_api.jsonl"
    row = json.loads(path.read_text())
    if mismatch == "old_api":
        row["turn_id"] = "one"
    elif mismatch == "old_cli":
        stdout = stdout[:-1]
    elif mismatch == "failed_api":
        row["response"]["status"] = "failed"
    elif mismatch == "mismatched_item":
        row["response"]["output"][0]["id"] = "older-item"
    else:
        path.write_text(json.dumps(row) + "\n" + json.dumps({"event": "request_start", "thread_id": "parent",
            "turn_id": "two", "start_ns": 3}) + "\n")
    if mismatch != "pending_api":
        path.write_text(json.dumps(row) + "\n")
    assert not evidence_for(tmp_path, stdout)[2]["complete"]


@pytest.mark.parametrize("status,expected", [("completed", True), ("incomplete", False)])
def test_only_latest_turn_truncation_matters(tmp_path, status, expected):
    stdout = completion_case(tmp_path, complete_turn() + complete_turn("unfinished", "two", 3),
                             final="unfinished", status=status, turn="two", stdout_turns=2)
    evidence = evidence_for(tmp_path, stdout)[2]
    assert evidence["complete"] is expected
    assert evidence["done_marker"] is False


def test_pending_tool_call_is_not_hidden_by_done(tmp_path):
    events = complete_turn()[:2] + [(1.5, "response_item", {"type": "function_call", "name": "exec_command",
        "call_id": "unfinished-call", "arguments": "{}"}), complete_turn()[-1]]
    stdout = completion_case(tmp_path, events)
    sessions, _, evidence = evidence_for(tmp_path, stdout)
    assert sessions[0]["pending_call_ids"] == ["unfinished-call"]
    assert evidence["terminal_evidence"] == "pending_parent_tool_calls"
    assert not evidence["complete"]


@pytest.mark.parametrize("final,marker", [("DONE", True), ("Implemented.\nDONE", True), ("done: validated", True),
    ("Done", True), ("DONEish", False), ("The task is DONE", False), ("Implementation complete", False)])
def test_done_marker_matches_roadmap_durable_policy(tmp_path, final, marker):
    stdout = completion_case(tmp_path, complete_turn(final), final=final, status="incomplete")
    evidence = evidence_for(tmp_path, stdout)[2]
    assert evidence["done_marker"] is marker
    assert evidence["complete"] is marker


@pytest.mark.parametrize("final", [
    "Now let me check the remaining key files I need:",
    "Now I have a comprehensive understanding of the codebase. Let me check the existing tests and the `fyne.Locale` type:",
    "I'll implement the remaining targets next.",
    "Next, we need to run the tests.",
    "Implementation complete.\n\nI still need to integrate the child's changes.",
])
def test_explicit_working_preamble_is_not_final_completion(tmp_path, final):
    assert continuation_preamble(final)
    assert assistant_phase({"content": final}) == "commentary"
    stdout = completion_case(tmp_path, complete_turn(final), final=final)
    evidence = evidence_for(tmp_path, stdout)[2]
    assert not evidence["complete"]
    assert evidence["terminal_evidence"] == "final_parent_message_continues_work"


@pytest.mark.parametrize("final", [
    "Implementation complete. Tests pass.",
    "I checked the tests and implemented the changes.",
    "Changes are integrated. Let me know if you need anything else.",
    "DONE: validated", "DONE", "The task is complete.",
])
def test_completed_closeout_remains_valid_without_mandatory_done(tmp_path, final):
    assert not continuation_preamble(final)
    assert assistant_phase({"content": final}) == "final_answer"
    stdout = completion_case(tmp_path, complete_turn(final), final=final)
    assert evidence_for(tmp_path, stdout)[2]["complete"]


def test_chat_upstream_explicit_phase_is_preserved():
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": "progress", "phase": "commentary"}}]}
    assert from_chat(reply, {}, "qwen")["output"][0]["phase"] == "commentary"


def test_preamble_recovery_shares_original_deadline(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    invocation = []

    class Container:
        backend_id = "singularity"

        def popen_exec(self, _handle, argv, **_kwargs):
            invocation.append(argv)
            pause = 0.15 if len(invocation) == 1 else 2
            return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({pause})"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    class Harness:
        def quiesce_agent_writers(self, _env):
            return {"workspaceQuiesced": True, "workspaceQuiescedNs": time.time_ns()}

    condition = CodexAgentCondition(None, SimpleNamespace(max_time_seconds=0.4), Container())
    monkeypatch.setattr(condition, "_live_completion", lambda *_args: {"continuation_eligible": True, "parent_thread_id": "same-parent"})
    result = condition.run({"instruction": "original task"}, tmp_path / "run", 42, "/app", "task", {
        "agent_env": {}, "state_dir": str(state), "codex_state_dir": str(state), "_harness": Harness()})
    assert result["agent_execution_outcome"] == "timeout"
    assert len(invocation) == 2 and "resume" in invocation[1] and "same-parent" in invocation[1]
    assert result["agent_lifecycle"]["continuationCount"] == 1
    assert result["phase_timing"]["parent_execution_seconds"] < 0.65
    assert (state / "prompt.txt").read_text() == "original task"
    assert "original task" in (state / "continuation.txt").read_text()


def collab_call(name, identity, args, result, start):
    return [(start, "response_item", {"type": "function_call", "namespace": "multi_agent_v1", "name": name,
            "call_id": identity, "arguments": json.dumps(args)}),
            (start + .1, "response_item", {"type": "function_call_output", "call_id": identity, "output": json.dumps(result)})]


@pytest.mark.parametrize("child_case,complete,timing", [
    ("completed", True, True), ("running", False, False), ("interrupted", False, True),
    ("missing_timestamps", True, False), ("missing_terminal_timestamp", True, False), ("missing_rollout_completed", True, False),
    ("missing_rollout", False, False), ("queued_after_completed", False, False),
    ("resumed_running", False, False), ("pending_api", False, False)])
def test_child_execution_and_timing_are_independent(tmp_path, child_case, complete, timing):
    events = [(0, "event_msg", {"type": "task_started", "turn_id": "one"})]
    events += collab_call("spawn_agent", "spawn", {}, {"agent_id": "child"}, .5)
    if child_case in {"missing_timestamps", "missing_rollout_completed"}:
        events += collab_call("wait_agent", "wait", {"targets": ["child"]}, {"status": {"child": {"completed": "result"}}}, 7)
    if child_case in {"queued_after_completed", "resumed_running"}:
        events += collab_call("send_input", "send", {"target": "child", "message": "more work"}, {"submission_id": "new"}, 6)
    events += [(9, "response_item", final_message("DONE")), (10, "event_msg", {"type": "task_complete", "turn_id": "one"})]
    stdout = completion_case(tmp_path, events)
    if child_case not in {"missing_rollout", "missing_rollout_completed"}:
        child_events = [(1, "event_msg", {"type": "task_started", "turn_id": "child-one"})]
        if child_case != "running":
            child_events += [(5, "event_msg", {"type": "turn_aborted" if child_case == "interrupted" else "task_complete", "turn_id": "child-one"})]
        if child_case == "resumed_running":
            child_events += [(6.5, "event_msg", {"type": "task_started", "turn_id": "child-two"})]
        rollout(tmp_path / "codex_state", "child", "parent", child_events)
        if child_case in {"missing_timestamps", "missing_terminal_timestamp"}:
            path = next((tmp_path / "codex_state").rglob("rollout-child.jsonl"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if child_case == "missing_timestamps" or row.get("payload", {}).get("type") == "task_complete":
                    row.pop("timestamp", None)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if child_case == "pending_api":
        with (tmp_path / "codex_api.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": "request_start", "thread_id": "child", "start_ns": 3}) + "\n")
    _, obs, evidence = evidence_for(tmp_path, stdout)
    assert evidence["complete"] is True
    assert (not obs["unfinished_child_session_ids"]) is complete
    assert obs["child_runtime_by_session"]["child"]["runtime_complete"] is timing
    assert obs["child_runtime_unknown_count"] == int(not timing)
    if not timing:
        assert obs["max_concurrent_children"] is None
        assert obs["child_sum_runtime_seconds"] is None
        assert obs["parent_activity_during_child_execution"] is None
        assert obs["parent_overlapping_tool_calls"] is None
        assert obs["parent_overlap_metrics_quality"] == "censored"


@pytest.mark.parametrize("child_case,expected", [("old_completion", False), ("old_wait", False),
    ("new_turn", True), ("steered_turn", True), ("closed_running", False), ("closed_completed", True)])
def test_native_followup_and_close_lifecycle_do_not_reuse_old_status(tmp_path, child_case, expected):
    events = [(0, "event_msg", {"type": "task_started", "turn_id": "one"})]
    events += collab_call("spawn_agent", "spawn", {}, {"agent_id": "child"}, .5)
    child_events = [(1, "event_msg", {"type": "task_started", "turn_id": "old-child-turn"})]
    if child_case in {"closed_running", "closed_completed"}:
        previous = "running" if child_case == "closed_running" else {"completed": "result"}
        events += collab_call("close_agent", "close", {"target": "child"}, {"previous_status": previous}, 6)
        child_events += [(5, "event_msg", {"type": "turn_aborted" if child_case == "closed_running" else "task_complete", "turn_id": "old-child-turn"})]
    else:
        events += collab_call("send_input", "send", {"target": "child", "message": "more work"}, {"submission_id": "new-child-turn"}, 3)
        child_events += [(5, "event_msg", {"type": "task_complete", "turn_id": "old-child-turn"})]
        if child_case == "old_wait":
            events += [(2.5, "response_item", {"type": "function_call", "name": "wait_agent", "namespace": "multi_agent_v1",
                "arguments": '{"targets":["child"]}', "call_id": "wait"}),
                (6, "response_item", {"type": "function_call_output", "call_id": "wait", "output": '{"status":{"child":{"completed":"old result"}}}'})]
            events.sort(key=lambda row: row[0])
        elif child_case == "new_turn":
            child_events += [(5.5, "event_msg", {"type": "task_started", "turn_id": "new-child-turn"}),
                             (7, "event_msg", {"type": "task_complete", "turn_id": "new-child-turn"})]
        elif child_case == "steered_turn":
            child_events.insert(1, (3.05, "response_item", {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "more work"}]}))
    events += [(9, "response_item", final_message("DONE")), (10, "event_msg", {"type": "task_complete", "turn_id": "one"})]
    stdout = completion_case(tmp_path, events)
    rollout(tmp_path / "codex_state", "child", "parent", child_events)
    _, obs, evidence = evidence_for(tmp_path, stdout)
    assert evidence["complete"]
    assert (not obs["unfinished_child_session_ids"]) is expected


def test_later_user_input_invalidates_previous_final_message(tmp_path):
    events = complete_turn()[:2] + [(1.5, "response_item", {"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "continue work"}]}), complete_turn()[-1]]
    stdout = completion_case(tmp_path, events)
    assert not evidence_for(tmp_path, stdout)[2]["complete"]


def test_unclosed_previous_parent_turn_is_not_skipped(tmp_path):
    events = [(0, "event_msg", {"type": "task_started", "turn_id": "one"})] + complete_turn("DONE", "two", 3)
    stdout = completion_case(tmp_path, events, turn="two", stdout_turns=2)
    assert evidence_for(tmp_path, stdout)[2]["terminal_evidence"] == "unclosed_parent_turn"
