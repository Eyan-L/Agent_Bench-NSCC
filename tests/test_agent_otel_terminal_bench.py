"""Smoke checks for shared agent OTel and Terminal-Bench adapter wiring."""

import json
from pathlib import Path
from types import SimpleNamespace

from agent_bench.benchmarks.conditions.agent_otel import otel_env, write_spans_and_otlp
from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition
from agent_bench.benchmarks.swebench.conditions.openhands_agent import (
    OpenHandsAgentCondition,
    TerminalBenchOpenHandsCondition,
)
from agent_bench.benchmarks.swebench.conditions.hermes_agent import TerminalBenchHermesCondition
from agent_bench.core.profile_io import load_profile
from agent_bench.harnesses.openhands.driver import EventSink, _LLMTelemetryProxy


ROOT = Path(__file__).resolve().parents[1]


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
    assert {item["key"] for item in attrs} >= {"oc.run_id", "agent.run_id", "service.name"}


def test_otel_env_aligns_agent_resource_with_openclaw_run_id():
    env = otel_env(
        service_name="agent-bench-hermes",
        run_id="run-1",
        endpoint="http://collector:4318",
        extra_attrs={"hm.run_id": "run-1"},
    )

    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4318"
    assert "oc.run_id=run-1" in env["OTEL_RESOURCE_ATTRIBUTES"]
    assert "agent.run_id=run-1" in env["OTEL_RESOURCE_ATTRIBUTES"]
    assert "hm.run_id=run-1" in env["OTEL_RESOURCE_ATTRIBUTES"]


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


def test_terminal_bench_conditions_prompt_and_marker_patch():
    data = {"instruction": "Create the expected answer file."}
    oh = TerminalBenchOpenHandsCondition(SimpleNamespace(name="model"), SimpleNamespace())
    hm = TerminalBenchHermesCondition(SimpleNamespace(name="model"), SimpleNamespace())

    assert "/tests/test.sh" in oh._build_message(data)
    assert "/logs/verifier/reward.txt" in hm._build_message(data)
    assert oh._collect_artifacts({}, "cid", "/app").startswith("# terminal_bench: no patch")
    assert hm._collect_artifacts({}, "cid", "/app").startswith("# terminal_bench: no patch")
