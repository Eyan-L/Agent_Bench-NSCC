"""Smoke checks for shared agent OTel and Terminal-Bench adapter wiring."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from agent_bench.benchmarks.conditions.agent_otel import (
    otel_endpoint_for_container,
    otel_env,
    snapshot_otel_spans,
    write_spans_and_otlp,
)
from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition
from agent_bench.benchmarks.swebench.conditions.openhands_agent import (
    OpenHandsAgentCondition,
)
from agent_bench.benchmarks.terminal_bench import app as tb_app
from agent_bench.benchmarks.terminal_bench import loader as tb_loader
from agent_bench.benchmarks.terminal_bench.conditions.hermes_agent import (
    TerminalBenchHermesCondition,
)
from agent_bench.benchmarks.terminal_bench.conditions.openhands_agent import (
    TerminalBenchOpenHandsCondition,
)
from agent_bench.benchmarks.terminal_bench.conditions.react import (
    TerminalBenchReactCondition,
)
from agent_bench.core.profile_io import load_profile
from agent_bench.deployments.containers.docker import DockerContainer
from agent_bench.harnesses.openhands import driver as openhands_driver
from agent_bench.harnesses.openhands.driver import (
    _BridgeToolExecutor,
    EventSink,
    _LLMTelemetryProxy,
    _patch_laminar_observe_compat,
    _action_to_bridge_call,
    _file_editor_action_payload,
    _instrument_llm_for_telemetry,
    _missing_required_bridge_tools,
    _shell_result_to_observation,
    _wrap_tool_executors,
)


ROOT = Path(__file__).resolve().parents[1]


def test_docker_container_traced_subprocess_writes_span(tmp_path: Path) -> None:
    spans_path = tmp_path / "spans.jsonl"
    container = DockerContainer()
    container.set_trace_path(spans_path)

    result = container._run_traced(
        "agent_bench.container.exec",
        [
            sys.executable,
            "-c",
            "import sys; print('ok'); print('err', file=sys.stderr)",
        ],
        attrs={"agent_bench.container.name": "demo"},
    )

    assert result.returncode == 0
    rows = [json.loads(line) for line in spans_path.read_text().splitlines()]
    assert len(rows) == 1
    span = rows[0]
    attrs = span["attrs"]
    assert span["name"] == "agent_bench.container.exec"
    assert attrs["agent_bench.container.name"] == "demo"
    assert attrs["agent_bench.returncode"] == 0
    assert "ok" in attrs["agent_bench.stdout_preview"]
    assert "err" in attrs["agent_bench.stderr_preview"]
    assert sys.executable in attrs["agent_bench.command_json"]


def test_shared_otel_writes_flat_and_raw_spans(tmp_path):
    spans_path = tmp_path / "spans.jsonl"
    write_spans_and_otlp(
        spans_path,
        [
            {
                "name": "agent.model.usage",
                "canonical_name": "agent.model.usage",
                "start_unix_nano": 1,
                "end_unix_nano": 2,
                "attrs": {"agent.tokens.total": 3},
                "resource": {"oc.run_id": "run-1", "service.name": "agent-bench-test"},
                "synthesized": True,
            }
        ],
        scope_name="agent-bench-test",
        resource_defaults={"agent.run_id": "run-1"},
    )

    flat = [json.loads(line) for line in spans_path.read_text().splitlines()]
    raw = json.loads((tmp_path / "spans_raw.jsonl").read_text())

    assert flat[0]["canonical_name"] == "agent.model.usage"
    attrs = raw["resourceSpans"][0]["resource"]["attributes"]
    assert {item["key"] for item in attrs} >= {"ab.run_id", "oc.run_id", "agent.run_id", "service.name"}


def test_otel_env_aligns_agent_resource_with_openclaw_run_id():
    env = otel_env(
        service_name="agent-bench-hermes",
        run_id="run-1",
        endpoint="http://collector:4318",
        extra_attrs={"hm.run_id": "run-1"},
    )

    assert env["AB_TELEMETRY_ENDPOINT"] == "http://collector:4318"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4318"
    assert "ab.run_id=run-1" in env["OTEL_RESOURCE_ATTRIBUTES"]
    assert "oc.run_id=run-1" in env["OTEL_RESOURCE_ATTRIBUTES"]
    assert "agent.run_id=run-1" in env["OTEL_RESOURCE_ATTRIBUTES"]
    assert "hm.run_id=run-1" in env["OTEL_RESOURCE_ATTRIBUTES"]


def test_otel_endpoint_for_container_ignores_loopback_host_endpoint(monkeypatch):
    monkeypatch.setenv("AB_TELEMETRY_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.delenv("AB_TELEMETRY_CONTAINER_ENDPOINT", raising=False)

    assert otel_endpoint_for_container("vllm-host") == "http://vllm-host:4318"

    monkeypatch.setenv("AB_TELEMETRY_CONTAINER_ENDPOINT", "http://collector:4318")
    assert otel_endpoint_for_container("vllm-host") == "http://collector:4318"


def test_snapshot_otel_spans_filters_by_ab_run_id_and_canonicalizes(tmp_path):
    trace_file = tmp_path / "otel-traces.jsonl"
    trace_file.write_text(
        json.dumps({
            "resourceSpans": [
                {
                    "resource": {"attributes": [
                        {"key": "oc.run_id", "value": {"stringValue": "run-1"}},
                        {"key": "service.name", "value": {"stringValue": "agent-bench-openclaw"}},
                    ]},
                    "scopeSpans": [{"spans": [{
                        "name": "openclaw.tool.call",
                        "startTimeUnixNano": "1000000000",
                        "endTimeUnixNano": "1100000000",
                        "attributes": [
                            {"key": "openclaw.tool.name", "value": {"stringValue": "bash"}},
                            {"key": "openclaw.tool.callId", "value": {"stringValue": "call-1"}},
                            {"key": "openclaw.tool.isError", "value": {"boolValue": False}},
                        ],
                    }]}],
                },
                {
                    "resource": {"attributes": [
                        {"key": "oc.run_id", "value": {"stringValue": "run-2"}},
                    ]},
                    "scopeSpans": [{"spans": [{
                        "name": "agent.model.usage",
                        "startTimeUnixNano": "1",
                        "endTimeUnixNano": "2",
                    }]}],
                },
            ]
        }) + "\n"
    )
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    spans_path = snapshot_otel_spans(str(trace_file), 0, run_dir, expected_run_id="run-1")

    spans = [json.loads(line) for line in spans_path.read_text().splitlines()]
    raw = json.loads((run_dir / "spans_raw.jsonl").read_text())
    assert len(raw["resourceSpans"]) == 1
    assert [span["canonical_name"] for span in spans] == ["agent.tool.call"]
    assert spans[0]["attrs"]["agent.tool.name"] == "bash"
    assert spans[0]["attrs"]["agent.tool.call_id"] == "call-1"


def test_openclaw_raw_response_log_synthesizes_per_request_model_usage(tmp_path):
    run_dir = tmp_path / "run-1"
    raw_dir = run_dir / "conversation"
    raw_dir.mkdir(parents=True)
    spans_path = run_dir / "spans.jsonl"
    spans_path.write_text("")
    rows = [
        {
            "ts": "2026-06-01T00:00:00.000Z",
            "sessionId": "sess-1",
            "runId": "run-1",
            "model": "model-a",
            "provider": "provider-a",
            "api": "openai-completions",
            "httpStatus": 200,
            "response": {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            },
            "metrics": {"total_ms": 1200, "ttft_ms": 100, "tbt_avg_ms": 20},
        },
        {
            "ts": "2026-06-01T00:00:02.000Z",
            "sessionId": "sess-1",
            "runId": "run-1",
            "model": "model-a",
            "provider": "provider-a",
            "api": "openai-completions",
            "httpStatus": 200,
            "response": {
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                }
            },
            "metrics": {"total_ms": 900, "ttft_ms": 80, "tbt_avg_ms": 15},
        },
    ]
    (raw_dir / "llm-raw-responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    count = OpenClawAgentCondition._maybe_synthesize_model_usage_spans_from_raw_log(
        spans_path, run_dir, "run-1"
    )

    spans = [json.loads(line) for line in spans_path.read_text().splitlines()]
    assert count == 2
    assert [span["attrs"]["openclaw.llm.request_index"] for span in spans] == [1, 2]
    assert [span["duration_ms"] for span in spans] == [1200.0, 900.0]
    assert [span["attrs"]["openclaw.tokens.input"] for span in spans] == [10, 20]
    assert [span["attrs"]["openclaw.tokens.output"] for span in spans] == [5, 7]
    assert all(span["attrs"]["openclaw.channel"] == "synth.raw_response" for span in spans)



def test_openhands_driver_proxy_emits_per_call_model_usage(tmp_path):
    class FakeLLM:
        def chat(self, *_args, **_kwargs):
            return {
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 8,
                    "total_tokens": 21,
                    "prompt_tokens_details": {"cached_tokens": 5},
                }
            }

    events_path = tmp_path / "events.jsonl"
    sink = EventSink(events_path)
    proxy = _LLMTelemetryProxy(
        FakeLLM(),
        sink,
        {"provider": "openai", "model": "model-a"},
    )

    proxy.chat("hello")
    sink.close()

    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["driver_event"] == "model_usage"
    assert row["type"] == "model.usage"
    assert row["channel"] == "openhands.llm"
    assert row["request_index"] == 1
    assert row["provider"] == "openai"
    assert row["model"] == "model-a"
    assert row["api"] == "chat"
    assert row["input_tokens"] == 13
    assert row["output_tokens"] == 8
    assert row["cache_read_tokens"] == 5
    assert row["total_tokens"] == 21


def test_openhands_driver_patches_laminar_observe_rollout_entrypoint(monkeypatch, tmp_path):
    calls = []
    lmnr_mod = ModuleType("lmnr")

    def legacy_observe(*, name=None):
        calls.append({"name": name})
        return "wrapped"

    lmnr_mod.observe = legacy_observe

    def fake_import_module(name):
        if name == "lmnr":
            return lmnr_mod
        raise AssertionError(name)

    monkeypatch.setattr(openhands_driver.importlib, "import_module", fake_import_module)

    events_path = tmp_path / "events.jsonl"
    sink = EventSink(events_path)

    assert _patch_laminar_observe_compat(sink) is True
    assert lmnr_mod.observe(
        name="conversation.run",
        rollout_entrypoint="ignored",
    ) == "wrapped"
    sink.close()

    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert rows[0]["driver_event"] == "laminar_observe_compat"
    assert rows[0]["patched"] is True
    assert rows[0]["target"] == "lmnr.observe"
    assert rows[0]["dropped_kwargs"] == ["rollout_entrypoint"]
    assert calls == [{"name": "conversation.run"}]


def test_openhands_driver_main_returns_nonzero_on_run_exception(monkeypatch, tmp_path):
    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

    class FakeConversation:
        def __init__(self, **_kwargs):
            self.state = SimpleNamespace(events=[], agent_state="failed")
            self.conversation_stats = None
            self.closed = False

        def send_message(self, _task):
            raise TypeError("boom")

        def run(self):
            raise AssertionError("run should not be called")

        def close(self):
            self.closed = True

    class FakeWorkspace:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(
        openhands_driver,
        "_import_sdk",
        lambda: {
            "LLM": FakeLLM,
            "Conversation": FakeConversation,
            "LocalWorkspace": FakeWorkspace,
            "get_default_agent": lambda **_kwargs: SimpleNamespace(tools=[]),
            "get_default_tools": lambda **_kwargs: [],
        },
    )
    monkeypatch.setattr(
        openhands_driver,
        "_patch_laminar_observe_compat",
        lambda sink: False,
    )
    monkeypatch.setattr(openhands_driver, "_BRIDGE_BIN", "")
    monkeypatch.setattr(openhands_driver, "_BRIDGE_SOCKET", "")
    monkeypatch.setattr(openhands_driver, "_BRIDGE_TOKEN", "")

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '''
[llm]
model = "model-a"
base_url = "http://example.invalid/v1"
api_key = "EMPTY"
temperature = 0

[agent]
name = "CodeActAgent"

[run]
workspace = "/workspace"
max_iterations = 1

[telemetry.otel]
enabled = false
'''
    )
    events_path = tmp_path / "events.jsonl"

    rc = openhands_driver.main([
        "--config", str(config_path),
        "--task", "fix it",
        "--events", str(events_path),
        "--workspace", str(tmp_path),
    ])

    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert rc == 7
    assert any(
        row.get("driver_event") == "error"
        and row.get("kind") == "run"
        and row.get("message") == "boom"
        for row in rows
    )
    summary = rows[-1]
    assert summary["driver_event"] == "summary"
    assert summary["run_error"] == "TypeError: boom"


def test_openhands_condition_marks_driver_run_error_fatal():
    events = [{"driver_event": "error", "kind": "run", "message": "boom"}]

    event = OpenHandsAgentCondition._fatal_driver_error_event(events)

    assert event == events[0]
    assert (
        OpenHandsAgentCondition._error_type_for_driver_result(0, event)
        == "openhands_driver_run"
    )


def test_openhands_driver_instruments_llm_without_replacing_instance(tmp_path):
    class FakeLLM:
        def model_copy(self):
            return self

        def completion(self, *_args, **_kwargs):
            return {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    events_path = tmp_path / "events.jsonl"
    sink = EventSink(events_path)
    llm = FakeLLM()

    instrumented = _instrument_llm_for_telemetry(
        llm,
        sink,
        {"provider": "openai", "model": "model-a"},
    )
    assert instrumented is llm

    llm.model_copy()
    llm.completion("hello")
    sink.close()

    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert rows[0]["driver_event"] == "llm_telemetry_wrap"
    assert rows[0]["methods_wrapped"] == ["completion"]
    usage_rows = [row for row in rows if row["driver_event"] == "model_usage"]
    assert len(usage_rows) == 1
    assert usage_rows[0]["api"] == "completion"
    assert usage_rows[0]["input_tokens"] == 2
    assert usage_rows[0]["output_tokens"] == 3


def test_openhands_bridge_wraps_resolved_tools_map(monkeypatch):
    monkeypatch.setattr(openhands_driver, "_BRIDGE_BIN", "/usr/local/bin/oh-exec-bridge")
    monkeypatch.setattr(openhands_driver, "_BRIDGE_SOCKET", "/tmp/exec-bridge.sock")
    monkeypatch.setattr(openhands_driver, "_BRIDGE_TOKEN", "token")

    class FakeExecutor:
        def __call__(self, *_args, **_kwargs):
            return {"content": [{"type": "text", "text": "local"}], "is_error": False}

    class FakeTool:
        def __init__(self, name, executor):
            self.name = name
            self.executor = executor

        def set_executor(self, executor):
            return FakeTool(self.name, executor)

    task_tracker_executor = FakeExecutor()
    tools = {
        "terminal": FakeTool("terminal", FakeExecutor()),
        "file_editor": FakeTool("file_editor", FakeExecutor()),
        "task_tracker": FakeTool("task_tracker", task_tracker_executor),
    }

    wrapped = _wrap_tool_executors(tools, "/app")

    assert wrapped == 2
    assert isinstance(tools["terminal"].executor, _BridgeToolExecutor)
    assert isinstance(tools["file_editor"].executor, _BridgeToolExecutor)
    assert tools["task_tracker"].executor is task_tracker_executor
    assert _missing_required_bridge_tools(tools) == []


def test_openhands_bridge_detects_missing_required_tool(monkeypatch):
    monkeypatch.setattr(openhands_driver, "_BRIDGE_BIN", "/usr/local/bin/oh-exec-bridge")
    monkeypatch.setattr(openhands_driver, "_BRIDGE_SOCKET", "/tmp/exec-bridge.sock")
    monkeypatch.setattr(openhands_driver, "_BRIDGE_TOKEN", "token")

    class FakeTool:
        def __init__(self, name, executor=None):
            self.name = name
            self.executor = executor

    tools = {
        "terminal": FakeTool("terminal", object()),
        "file_editor": FakeTool("file_editor", None),
    }

    _wrap_tool_executors(tools, "/app")

    assert _missing_required_bridge_tools(tools) == ["file_editor"]


def test_openhands_file_editor_bridge_payload_covers_writes():
    class FileAction:
        command = "str_replace"
        path = "/app/input.txt"
        file_text = None
        view_range = None
        old_str = "old"
        new_str = "new"
        insert_line = None

    payload = _file_editor_action_payload(FileAction())

    assert payload["command"] == "str_replace"
    assert payload["path"] == "/app/input.txt"
    assert payload["old_str"] == "old"
    assert payload["new_str"] == "new"


def test_openhands_file_editor_bridge_uses_shell_fs_ops(monkeypatch, tmp_path):
    files = {"/app/input.txt": b"hello old\n"}
    dirs = {"/app"}
    calls = []

    def fake_bridge_exec_argv(argv, *, workdir, stdin_bytes=None, **_kwargs):
        calls.append(argv)
        tag = argv[3]
        path = argv[4]
        stdout = b""
        stderr = b""
        exit_code = 0
        if tag == "openhands-file-stat":
            if path in dirs:
                stdout = b"dir"
            elif path in files:
                stdout = b"file"
            else:
                stdout = b"missing"
        elif tag == "openhands-file-read":
            stdout = files[path]
        elif tag == "openhands-file-write":
            files[path] = stdin_bytes or b""
        else:
            exit_code = 127
            stderr = f"unknown tag {tag}".encode()
        return {
            "exit_code": exit_code,
            "stdout_bytes": stdout,
            "stderr_bytes": stderr,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }

    class FileAction:
        command = "str_replace"
        path = "/app/input.txt"
        file_text = None
        view_range = None
        old_str = "old"
        new_str = "new"
        insert_line = None

    monkeypatch.setattr(openhands_driver, "bridge_exec_argv", fake_bridge_exec_argv)
    monkeypatch.setattr(
        openhands_driver,
        "_FILE_EDITOR_HISTORY_PATH",
        tmp_path / "file_editor_history.json",
    )

    obs = openhands_driver._run_file_editor_action(FileAction(), "/app")

    assert obs["is_error"] is False
    assert files["/app/input.txt"] == b"hello new\n"
    assert obs["old_content"] == "hello old\n"
    assert obs["new_content"] == "hello new\n"
    assert calls
    assert all(call[:2] == ["sh", "-c"] for call in calls)
    assert "python3" not in json.dumps(calls)


def test_openhands_bridge_translates_swebench_workspace_paths(monkeypatch):
    monkeypatch.setattr(openhands_driver, "_BRIDGE_AGENT_WORKSPACE", "/workspace")
    monkeypatch.setattr(openhands_driver, "_BRIDGE_TASK_WORKSPACE", "/testbed")

    class TerminalAction:
        command = "python /workspace/tools/check.py"
        timeout = None

    terminal_call = _action_to_bridge_call("terminal", TerminalAction(), "/workspace")
    assert terminal_call.command == "python /testbed/tools/check.py"
    assert terminal_call.workdir == "/testbed"

    class FileAction:
        command = "view"
        path = "/workspace/input.txt"
        file_text = None
        view_range = None
        old_str = None
        new_str = None
        insert_line = None

    payload = _file_editor_action_payload(FileAction())
    assert payload["path"] == "/testbed/input.txt"

    file_payload = {
        "ab_bridge_file_editor": True,
        "observation": {
            "content": [{"type": "text", "text": "contents", "cache_prompt": False}],
            "is_error": False,
            "command": "view",
            "path": "/testbed/input.txt",
            "prev_exist": True,
            "old_content": None,
            "new_content": None,
        },
    }
    file_obs = _shell_result_to_observation(
        "file_editor",
        FileAction(),
        {"exit_code": 0, "stdout": json.dumps(file_payload), "stderr": ""},
        workdir="/testbed",
    )
    assert file_obs["path"] == "/workspace/input.txt"


def test_openhands_bridge_observations_match_sdk_shapes():
    class TerminalAction:
        command = "echo ok"

    terminal_obs = _shell_result_to_observation(
        "terminal",
        TerminalAction(),
        {"exit_code": 0, "stdout": "ok\n", "stderr": ""},
        workdir="/app",
    )

    assert terminal_obs["content"][0]["type"] == "text"
    assert terminal_obs["exit_code"] == 0
    assert terminal_obs["metadata"]["working_dir"] == "/app"

    class FileAction:
        command = "view"
        path = "/app/input.txt"

    file_payload = {
        "ab_bridge_file_editor": True,
        "observation": {
            "content": [{"type": "text", "text": "contents", "cache_prompt": False}],
            "is_error": False,
            "command": "view",
            "path": "/app/input.txt",
            "prev_exist": True,
            "old_content": None,
            "new_content": None,
        },
    }
    file_obs = _shell_result_to_observation(
        "file_editor",
        FileAction(),
        {"exit_code": 0, "stdout": json.dumps(file_payload), "stderr": ""},
        workdir="/app",
    )

    assert file_obs["command"] == "view"
    assert file_obs["path"] == "/app/input.txt"
    assert file_obs["content"][0]["text"] == "contents"


def test_openhands_synthesizes_agent_spans_from_events():
    condition = OpenHandsAgentCondition(
        model_config=SimpleNamespace(name="model"),
        agent_config=SimpleNamespace(),
    )
    events = [
        {
            "kind": "ActionEvent",
            "tool_call_id": "call-1",
            "tool_name": "execute_bash",
            "timestamp": "2026-06-02T00:00:00+00:00",
        },
        {
            "kind": "ObservationEvent",
            "tool_call_id": "call-1",
            "timestamp": "2026-06-02T00:00:01+00:00",
        },
        {
            "driver_event": "summary",
            "metrics": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
    ]

    spans = condition._build_spans(events, "run-1", 1, 2)

    assert [span["canonical_name"] for span in spans] == [
        "agent.model.usage",
        "agent.tool.call",
    ]
    assert spans[0]["attrs"]["agent.channel"] == "synth.agent_call"
    assert spans[0]["attrs"]["agent.tokens.total"] == 12
    assert spans[1]["attrs"]["agent.tool.name"] == "execute_bash"
    assert spans[0]["resource"]["oc.run_id"] == "run-1"


def test_openhands_prefers_model_usage_events_over_summary():
    condition = OpenHandsAgentCondition(
        model_config=SimpleNamespace(name="fallback-model"),
        agent_config=SimpleNamespace(),
    )
    events = [
        {
            "type": "model.usage",
            "start_ns": 1_000_000_000,
            "end_ns": 1_500_000_000,
            "duration_ms": 500.0,
            "provider": "openai",
            "model": "model-a",
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
            "ttft_ms": 50,
            "tbt_avg_ms": 10,
        },
        {
            "driver_event": "summary",
            "metrics": {"prompt_tokens": 50, "completion_tokens": 70, "total_tokens": 120},
        },
    ]

    spans = condition._build_spans(events, "run-1", 1, 2)

    assert [span["canonical_name"] for span in spans] == ["agent.model.usage"]
    assert spans[0]["attrs"]["agent.channel"] == "openhands.events"
    assert spans[0]["attrs"]["agent.llm.request_index"] == 1
    assert spans[0]["attrs"]["agent.tokens.total"] == 7
    assert spans[0]["attrs"]["agent.ttft_ms"] == 50


def test_openhands_runtime_and_condition_use_collector_otel():
    driver = (ROOT / "agent_bench/harnesses/openhands/driver.py").read_text()
    condition = (
        ROOT / "agent_bench/benchmarks/swebench/conditions/openhands_agent.py"
    ).read_text()
    swe_app = (ROOT / "agent_bench/benchmarks/swebench/openhands/app.py").read_text()
    tb_app = (ROOT / "agent_bench/benchmarks/terminal_bench/openhands/app.py").read_text()

    template = (
        ROOT / "agent_bench/harnesses/openhands/state/config.toml.template"
    ).read_text()
    profile = (ROOT / "profiles/harnesses/openhands_swebench.yaml").read_text()

    assert "class _OtelEmitter" in driver
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in driver
    assert "telemetry_cfg" in driver
    assert "AB_RUN_ID" in driver
    assert "agent.model.usage" in driver
    assert "agent.tool.call" in driver
    assert "_ensure_agent_ready" in driver
    assert "_wrap_tool_executors(tools_map" in driver
    assert "_wrap_tool_executors(list(agent.tools)" not in driver
    assert "missing_required_tools" in driver
    assert "OPENHANDS_EXEC_BRIDGE_AGENT_WORKSPACE" in swe_app
    assert "OPENHANDS_EXEC_BRIDGE_TASK_WORKSPACE" in swe_app
    assert "OPENHANDS_EXEC_BRIDGE_AGENT_WORKSPACE" in tb_app
    assert "OPENHANDS_EXEC_BRIDGE_TASK_WORKSPACE" in tb_app
    assert "snapshot_otel_spans(" in condition
    assert "write_spans_and_otlp(" not in condition
    assert "[telemetry.otel]" in template
    assert 'endpoint = "${OTEL_ENDPOINT}"' in template
    assert 'run_id = "${AB_RUN_ID}"' in template
    assert "telemetry:" in profile
    assert "service_name: agent-bench-openhands" in profile


def test_terminal_bench_harness_profiles_and_driver_wiring():
    driver = (ROOT / "agent_bench/runtime/driver.py").read_text()
    assert "TerminalBenchOpenHandsCondition" in driver
    assert "TerminalBenchHermesCondition" in driver
    assert "TerminalBenchOpenHandsHarness" in driver
    assert "TerminalBenchHermesHarness" in driver

    oh = load_profile(ROOT / "profiles/harnesses/openhands_terminal_bench.yaml")
    hm = load_profile(ROOT / "profiles/harnesses/hermes_terminal_bench.yaml")
    assert oh["harness"] == "openhands"
    assert hm["harness"] == "hermes"
    assert oh["conditions"][0]["condition_type"] == "openhands_agent"
    assert hm["conditions"][0]["condition_type"] == "hermes_agent"


def test_terminal_bench_tests_are_scoring_only():
    app = (ROOT / "agent_bench/benchmarks/terminal_bench/app.py").read_text()
    base = (
        ROOT / "agent_bench/benchmarks/terminal_bench/agent_harness_base.py"
    ).read_text()
    openclaw_profile = (
        ROOT / "profiles/harnesses/openclaw_terminal_bench.yaml"
    ).read_text()

    assert 'f"{logs_dir}:/logs/verifier:rw"' in app
    assert ':/tests:ro' not in app
    assert 'self.container.copy_into(self._handle(task_container_name)' not in app
    assert '["docker", "cp", str(tests_host), f"{container_name}:/tests"]' in app
    assert "_copy_tests_into_singularity_instance(instance_name, tests_host)" in app

    assert 'Mount(host=logs_dir, container="/logs/verifier", mode="rw")' in base
    assert 'Mount(host=tests' not in base
    assert 'self.container.copy_into(handle, tests_host, "/tests")' in base
    assert "_clear_reward_file(host_logs_dir)" in base

    assert "- verify" not in openclaw_profile


def test_terminal_bench_workspace_follows_task_image_workdir(tmp_path):
    row = tb_loader.load_task("prove-plus-comm")
    assert row["workspace"] == "/workspace"
    assert row["workspace_source"] == "dockerfile"

    assert tb_app._resolve_task_workspace({"workspace": "/app"}, "/workspace") == "/workspace"
    assert (
        tb_app._resolve_task_workspace(
            {"workspace": "/app", "workspace_source": "task.toml"},
            "/workspace",
        )
        == "/app"
    )
    assert (
        tb_app._resolve_task_workspace(
            {"workspace": "/from-dockerfile", "workspace_source": "dockerfile"},
            "/from-image",
        )
        == "/from-image"
    )

    tb_app._seed_workspace_helpers(tmp_path, row, "/workspace")
    bootstrap = (tmp_path / "BOOTSTRAP.md").read_text()
    assert "`/workspace`" in bootstrap
    assert "`/app`" not in bootstrap

    app_source = (ROOT / "agent_bench/benchmarks/terminal_bench/app.py").read_text()
    base_source = (
        ROOT / "agent_bench/benchmarks/terminal_bench/agent_harness_base.py"
    ).read_text()
    assert 'f"{app_dir}:{task_workspace}:rw"' in app_source
    assert '"-w", task_workspace' in app_source
    assert "workdir=task_workspace" in app_source
    assert "Mount(host=app_dir, container=task_workspace, mode=\"rw\")" in base_source
    assert "workdir=task_workspace" in base_source


def test_terminal_bench_react_uses_dynamic_task_workspace():
    cond = TerminalBenchReactCondition(SimpleNamespace(name="model"))
    prompt = cond._compose_prompt("Create the answer.", [], "/workspace")
    assert "The task workspace is `/workspace`" in prompt
    assert "cat > /workspace/foo.py" in prompt
    assert "cat > /app/foo.py" not in prompt

    source = (
        ROOT / "agent_bench/benchmarks/terminal_bench/conditions/react.py"
    ).read_text()
    assert '"docker", "exec", "-i", "-w", self._task_workspace' in source
    assert '"singularity", "exec", "--pwd", self._task_workspace' in source


def test_bridge_harnesses_preclean_deterministic_run_resources():
    sources = [
        ROOT / "agent_bench/benchmarks/terminal_bench/agent_harness_base.py",
        ROOT / "agent_bench/benchmarks/swebench/hermes/app.py",
        ROOT / "agent_bench/benchmarks/swebench/openhands/app.py",
    ]
    for source in sources:
        text = source.read_text()
        assert "def _preflight_cleanup(" in text
        assert text.index("self._preflight_cleanup(") < text.index(
            "self.container.create_network("
        )


def test_interrupt_cleanup_wiring_is_baseexception_safe():
    commands = (ROOT / "agent_bench/core/commands.py").read_text()
    driver = (ROOT / "agent_bench/runtime/driver.py").read_text()
    assert "_install_signal_handlers()" in commands
    assert "signal.SIGINT" in driver
    assert "def _protect_cleanup_from_interrupts(" in driver
    assert "with _protect_cleanup_from_interrupts():" in driver
    assert "except BaseException:" in driver

    sources = [
        ROOT / "agent_bench/benchmarks/terminal_bench/agent_harness_base.py",
        ROOT / "agent_bench/benchmarks/swebench/hermes/app.py",
        ROOT / "agent_bench/benchmarks/swebench/openhands/app.py",
        ROOT / "agent_bench/benchmarks/swebench/openclaw/app.py",
        ROOT / "agent_bench/benchmarks/terminal_bench/app.py",
    ]
    for source in sources:
        text = source.read_text()
        assert "except BaseException:" in text
        assert "_best_effort_cleanup(" in text


def test_terminal_bench_conditions_prompt_and_marker_patch():
    data = {"instruction": "Create the expected answer file."}
    oh = TerminalBenchOpenHandsCondition(SimpleNamespace(name="model"), SimpleNamespace())
    hm = TerminalBenchHermesCondition(SimpleNamespace(name="model"), SimpleNamespace())

    for message in (oh._build_message(data), hm._build_message(data)):
        assert "/tests/test.sh" not in message
        assert "/logs/verifier" not in message
        assert "Declare DONE only when the task is complete" in message
    assert oh._collect_artifacts({}, "cid", "/app").startswith("# terminal_bench: no patch")
    assert hm._collect_artifacts({}, "cid", "/app").startswith("# terminal_bench: no patch")
