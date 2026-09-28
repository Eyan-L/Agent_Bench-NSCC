"""Exercise HTTP/SSE wire sequences, including the unmodified CLI parser."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
import urllib.request

import pytest

from agent_bench.harnesses.codex.responses_adapter import AdapterServer, response_events, to_chat
from agent_bench.harnesses.codex.runtime import write_config
from agent_bench.harnesses.codex.telemetry import api_response, jsonl, summarize
from agent_bench.harnesses.codex.completion import parent_completion
from agent_bench.harnesses.codex.condition import CodexAgentCondition
from agent_bench.harnesses.codex.runtime import PAYLOAD_PATH, STATE_PATH


@contextmanager
def adapter_with_reply(state, message, finish="stop", *, sequence=None, config=None):
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            reply, reason = sequence[min(len(requests) - 1, len(sequence) - 1)] if sequence else (message, finish)
            body = json.dumps({"choices": [{"finish_reason": reason, "message": reply}],
                               "usage": {"prompt_tokens": 12, "completion_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    adapter = AdapterServer(("127.0.0.1", 0), {
        "base_url": f"http://127.0.0.1:{upstream.server_port}/v1", "token": "test-only",
        "upstream_api": "chat_completions", "max_tokens": 64, **(config or {})}, state)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, adapter)]
    for thread in threads:
        thread.start()
    try:
        yield adapter, requests
    finally:
        for server in (adapter, upstream):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


def request_stream(adapter, **extra):
    request = urllib.request.Request(f"http://127.0.0.1:{adapter.server_port}/v1/responses",
        data=json.dumps({"model": "qwen", "input": "task", "stream": True, **extra}).encode(),
        headers={"Authorization": "Bearer test-only", "Content-Type": "application/json", "thread-id": "parent"})
    with urllib.request.urlopen(request, timeout=5) as reply:
        assert reply.headers["Content-Type"] == "text/event-stream"
        wire = reply.read().decode()
    packets = wire.strip().split("\n\n")
    events = []
    for packet in packets:
        header, data = packet.split("\n", 1)
        event = json.loads(data.removeprefix("data: "))
        assert header == "event: " + event["type"]
        events.append(event)
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    return events


@pytest.mark.parametrize("finish,terminal,item_status", [
    ("stop", "response.completed", "completed"), ("length", "response.incomplete", "incomplete")])
def test_http_text_sequence_and_terminal_status(tmp_path, finish, terminal, item_status):
    with adapter_with_reply(tmp_path, {"content": "visible answer"}, finish) as (adapter, _):
        events = request_stream(adapter, client_metadata={"turn_id": "final-turn"})
    assert [e["type"] for e in events] == [
        "response.created", "response.in_progress", "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.done", "response.content_part.done",
        "response.output_item.done", terminal]
    assert events[0]["response"]["incomplete_details"] is None
    assert events[2]["item"]["status"] == "in_progress" and events[2]["item"]["content"] == []
    assert events[-2]["item"]["status"] == item_status
    assert events[4]["delta"] == events[5]["text"] == "visible answer"
    assert events[-1]["response"]["status"] == item_status
    rows = [r for r in jsonl(tmp_path / "codex_api.jsonl") if "end_ns" in r]
    assert rows[0]["turn_id"] == "final-turn"
    assert api_response(rows[0])["usage"]["output_tokens"] == 4


@pytest.mark.parametrize("kind,name,namespace,payload", [
    ("function", "spawn_agent", "multi_agent_v1", {"message": "work"}),
    ("custom", "apply_patch", None, {"input": "*** Begin Patch\n*** End Patch"})])
def test_http_forced_tool_sequence_restores_identity(tmp_path, kind, name, namespace, payload):
    tool = {"type": kind, "name": name}
    if kind == "function":
        tool["parameters"] = {"type": "object", "properties": {}}
    tools = [{"type": "namespace", "name": namespace, "tools": [tool]}] if namespace else [tool]
    choice = {"type": kind, "name": name, **({"namespace": namespace} if namespace else {})}
    flattened = (namespace + "__" if namespace else "") + name
    reply = {"content": None, "tool_calls": [{"id": "call-one", "type": "function",
             "function": {"name": flattened, "arguments": json.dumps(payload)}}]}
    with adapter_with_reply(tmp_path, reply, "tool_calls") as (adapter, requests):
        events = request_stream(adapter, tools=tools, tool_choice=choice, parallel_tool_calls=False)
    assert requests[0]["tool_choice"] == {"type": "function", "function": {"name": flattened}}
    assert requests[0]["parallel_tool_calls"] is False
    field = "arguments" if kind == "function" else "input"
    prefix = "response.function_call_arguments" if kind == "function" else "response.custom_tool_call_input"
    assert [e["type"] for e in events] == ["response.created", "response.in_progress",
        "response.output_item.added", prefix + ".delta", prefix + ".done", "response.output_item.done", "response.completed"]
    item = events[-2]["item"]
    assert item["call_id"] == "call-one" and item["name"] == name
    assert item.get("namespace") == namespace
    assert events[2]["item"][field] == ""
    assert events[3]["delta"] == events[4][field] == item[field]


@pytest.mark.parametrize("choice", ["none", "required", "auto"])
def test_http_explicit_string_choice_is_preserved(tmp_path, choice):
    with adapter_with_reply(tmp_path, {"content": "DONE"}) as (adapter, requests):
        request_stream(adapter, tools=[{"type": "function", "name": "exec", "parameters": {"type": "object", "properties": {}}}],
                       tool_choice=choice)
    assert requests[0]["tool_choice"] == choice


@pytest.mark.parametrize("choice", [{"type": "web_search"}, {"type": "function", "name": "missing"}, "unexpected", "required"])
def test_unsupported_or_unbound_choice_is_rejected(choice):
    with pytest.raises(ValueError, match="tool_choice"):
        to_chat({"model": "qwen", "input": "task", "tool_choice": choice}, {})


def test_failed_terminal_event_is_not_completed():
    events = list(response_events({"id": "failed", "status": "failed", "output": [], "error": {"message": "failure"}}))
    assert events[-1]["type"] == "response.failed"


def test_http_working_preamble_retains_commentary_and_no_think(tmp_path):
    preamble = "Now I have a comprehensive understanding of the codebase. Let me check the existing tests and the `fyne.Locale` type:"
    with adapter_with_reply(tmp_path, {"content": preamble}, config={"chat_template_kwargs": {"enable_thinking": False}}) as (adapter, requests):
        events = request_stream(adapter)
    assert events[-1]["type"] == "response.completed"
    assert events[-2]["item"]["phase"] == "commentary"
    assert events[-2]["item"]["content"][0]["text"] == preamble
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def run_public_cli(tmp_path, adapter, prompt):
    workspace, home = tmp_path / "work", tmp_path / "codex_state"
    workspace.mkdir()
    config = write_config(home, model="qwen", base_url=f"http://127.0.0.1:{adapter.server_port}/v1",
                         context_window=262144, settings={"multi_agent": False})
    config.write_text(config.read_text().replace("stream_max_retries = 2", "stream_max_retries = 0"))
    result = subprocess.run(["singularity", "exec", "--bind", f"{tmp_path}:/opt/protocol-probe",
        os.environ["AB_CODEX_PROTOCOL_PROBE_SIF"], "/opt/codex/bin/codex", "exec", "--json", "--color", "never",
        "--strict-config", "--ignore-rules", "--skip-git-repo-check", "--cd", "/opt/protocol-probe/work", "-"],
        input=prompt, text=True, capture_output=True, timeout=180,
        env=dict(os.environ, SINGULARITYENV_CODEX_HOME="/opt/protocol-probe/codex_state", SINGULARITYENV_AB_CODEX_API_KEY="test-only"))
    (tmp_path / "codex_stdout.jsonl").write_text(result.stdout)
    (tmp_path / "codex_stderr.log").write_text(result.stderr)
    return result


@pytest.mark.skipif(not os.environ.get("AB_CODEX_PROTOCOL_PROBE_SIF"), reason="requires explicitly selected NSCC SIF")
def test_public_cli_resumes_working_preamble_in_same_native_session(tmp_path):
    sequence = [
        ({"content": "Let me check the repository before implementing the task:"}, "stop"),
        ({"content": None, "tool_calls": [{"id": "write-marker", "type": "function", "function": {
            "name": "exec_command", "arguments": json.dumps({"cmd": "echo implementation > continuation-marker"})}}]}, "tool_calls"),
        ({"content": "DONE"}, "stop"),
    ]
    workspace, home, run = tmp_path / "work", tmp_path / "codex_state", tmp_path / "run"
    workspace.mkdir()

    class Container:
        backend_id = "singularity"

        def popen_exec(self, _handle, argv, **kwargs):
            mapped = [a.replace(PAYLOAD_PATH + "/bin/codex", "/opt/codex/bin/codex")
                       .replace(STATE_PATH, "/opt/protocol-probe/codex_state") for a in argv]
            env = dict(os.environ, **{"SINGULARITYENV_" + k: v for k, v in kwargs["env"].items()})
            return subprocess.Popen(["singularity", "exec", "--bind", f"{tmp_path}:/opt/protocol-probe",
                os.environ["AB_CODEX_PROTOCOL_PROBE_SIF"], *mapped], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    class Harness:
        def quiesce_agent_writers(self, _env):
            return {"workspaceQuiesced": True, "workspaceQuiescedNs": time.time_ns()}

    with adapter_with_reply(tmp_path, None, sequence=sequence,
                            config={"chat_template_kwargs": {"enable_thinking": False}}) as (adapter, requests):
        write_config(home, model="qwen", base_url=f"http://127.0.0.1:{adapter.server_port}/v1",
                     context_window=262144, settings={"multi_agent": False})
        condition = CodexAgentCondition(None, SimpleNamespace(max_time_seconds=180), Container())
        result = condition.run({"instruction": "Implement the task and declare DONE."}, run, 42,
            "/opt/protocol-probe/work", "probe", {
                "state_dir": str(tmp_path), "codex_state_dir": str(home), "_harness": Harness(),
                "agent_env": {"CODEX_HOME": "/opt/protocol-probe/codex_state", "AB_CODEX_API_KEY": "test-only"}})
    stdout = (run / "codex_stdout.jsonl").read_text()
    assert result["agent_execution_outcome"] == "normal_completion", result
    assert result["agent_lifecycle"]["continuationCount"] == 1
    assert result["agent_lifecycle"]["strictDone"] is True
    assert len(requests) == 3, stdout
    assert (tmp_path / "work/continuation-marker").read_text().strip() == "implementation"
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in requests)
    sessions, _ = summarize(run)
    evidence = parent_completion(sessions, jsonl(run / "codex_stdout.jsonl"), jsonl(run / "codex_api.jsonl"))
    assert evidence["complete"] and evidence["done_marker"], evidence
    assert len(sessions) == 1 and len(sessions[0]["turns"]) == 2
    assert sessions[0]["turns"][0]["last_assistant_phase"] == "commentary"
    assert [r["tool_name"] for r in jsonl(run / "steps.jsonl")] == ["codex.exec", "codex.exec.resume"]
    assert stdout.count('"type":"turn.completed"') == 2


@pytest.mark.skipif(not os.environ.get("AB_CODEX_PROTOCOL_PROBE_SIF"), reason="requires explicitly selected NSCC SIF")
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_public_cli_consumes_actual_adapter_sequence(tmp_path, finish):
    with adapter_with_reply(tmp_path, {"content": "DONE" if finish == "stop" else "unfinished"}, finish) as (adapter, _):
        result = run_public_cli(tmp_path, adapter, "Respond with the test answer.")
    sessions, _ = summarize(tmp_path)
    evidence = parent_completion(sessions, list(jsonl(tmp_path / "codex_stdout.jsonl")), list(jsonl(tmp_path / "codex_api.jsonl")))
    if finish == "length":
        assert result.returncode != 0, result.stdout
        assert not evidence["complete"]
        assert '"type":"turn.completed"' not in result.stdout
        assert "Incomplete response returned" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert evidence["complete"] and evidence["done_marker"], evidence
