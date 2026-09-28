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
    (state / "codex_api.jsonl").write_text(json.dumps({"start_ns": 1, "end_ns": 2, "thread_id": "parent", "response": {
        "status": "incomplete" if truncated else "completed"}}) + "\n")

    class Container:
        backend_id = "singularity"

        def popen_exec(self, _handle, argv, **_kwargs):
            assert argv[0] == "sh" and "prompt.txt" in argv[2]
            assert (source / "prompt.txt").read_text() == "task"
            return subprocess.Popen([sys.executable, "-c", 'print(\'{"type":"turn.completed"}\')'],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    class Harness:
        def quiesce_agent_writers(self, _env):
            return {"workspaceQuiesced": True, "workspaceQuiescedNs": time.time_ns()}

    condition = CodexAgentCondition(None, SimpleNamespace(max_time_seconds=10), Container())
    result = condition.run({"instruction": "task"}, output, 42, "/app", "task", {
        "agent_env": {}, "state_dir": str(state), "codex_state_dir": str(source), "_harness": Harness()})
    assert result["agent_execution_outcome"] == expected
    assert result["phase_timing"]["agent_execution_end_ns"] == result["agent_lifecycle"]["workspaceQuiescedNs"]
