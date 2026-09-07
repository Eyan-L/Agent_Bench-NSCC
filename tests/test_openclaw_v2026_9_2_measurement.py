from __future__ import annotations

import json
from pathlib import Path

from agent_bench.benchmarks.conditions import openclaw_measurement as shared_measurement
from agent_bench.benchmarks.roadmap_bench.conditions.measurement_compat import (
    augment_roadmap_measurement,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _zero_tokens() -> dict:
    return {
        "parent": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "turn_count": 0,
            "resumed_turns": 0,
        },
        "children": {},
        "total_logical_input": 0,
        "total_logical_output": 0,
    }


def _model_call_span(
    *,
    span_id: str,
    start_ns: int,
    duration_ms: float,
    prompt_tokens: int,
    output_tokens: int,
) -> dict:
    return {
        "name": "openclaw.model.call",
        "trace_id": "trace-1",
        "span_id": span_id,
        "start_unix_nano": start_ns,
        "end_unix_nano": start_ns + int(duration_ms * 1e6),
        "duration_ms": duration_ms,
        "attrs": {
            "openclaw.model_call.observation_unit": "request",
            "openclaw.provider": "openai",
            "openclaw.model": "Qwen/Qwen3.6-35B-A3B",
            "openclaw.model_call.usage.prompt_tokens": prompt_tokens,
            "openclaw.model_call.usage.output_tokens": output_tokens,
            "openclaw.model_call.time_to_first_byte_ms": 42.5,
            "openclaw.model_call.request_bytes": 1234,
            "openclaw.model_call.response_bytes": 5678,
            "openclaw.sessionId": "session-1",
            "openclaw.sessionKey": "agent:roadmap_bench:main",
        },
    }


def test_v2026_9_2_native_model_call_spans_replace_missing_raw_serving_log(
    tmp_path: Path,
):
    _write_jsonl(
        tmp_path / "spans.jsonl",
        [
            _model_call_span(
                span_id="span-1",
                start_ns=1_000_000_000,
                duration_ms=100.0,
                prompt_tokens=120,
                output_tokens=30,
            ),
            _model_call_span(
                span_id="span-2",
                start_ns=2_000_000_000,
                duration_ms=250.0,
                prompt_tokens=240,
                output_tokens=60,
            ),
        ],
    )

    result = augment_roadmap_measurement(
        tmp_path,
        {
            "trajectory_tokens": _zero_tokens(),
            "serving_tokens": {
                "processed_input": 0,
                "generated_output": 0,
                "request_count": 0,
            },
            "runtime_observability": {},
        },
    )

    assert result["serving_measurement_source"] == "native_model_call_otel"
    assert result["serving_tokens"] == {
        "processed_input": 360,
        "generated_output": 90,
        "request_count": 2,
    }
    assert result["total_prompt_tokens"] == 360
    assert result["total_completion_tokens"] == 90
    assert (
        result["trajectory_tokens"]["accounting_source"]
        == "native_model_call_otel_fallback_unattributed"
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
        if line
    ]
    requests = [row for row in rows if row.get("event_type") == "model_request"]
    assert len(requests) == 2
    assert all(row.get("source") == "native_model_call_otel" for row in requests)
    assert requests[0]["time_to_first_byte_ms"] == 42.5
    assert requests[0]["request_payload_bytes"] == 1234
    assert requests[0]["response_stream_bytes"] == 5678

    accounting = json.loads((tmp_path / "token_accounting.json").read_text())
    assert accounting["serving_tokens"]["request_count"] == 2
    assert "native per-request model-call OTel spans" in accounting["sources"]["serving_tokens"]


def test_v2026_4_22_raw_serving_totals_remain_authoritative_when_spans_exist(
    tmp_path: Path,
):
    _write_jsonl(
        tmp_path / "spans.jsonl",
        [
            _model_call_span(
                span_id="span-native",
                start_ns=3_000_000_000,
                duration_ms=100.0,
                prompt_tokens=999,
                output_tokens=888,
            )
        ],
    )

    result = augment_roadmap_measurement(
        tmp_path,
        {
            "trajectory_tokens": _zero_tokens(),
            "serving_tokens": {
                "processed_input": 500,
                "generated_output": 100,
                "request_count": 3,
            },
            "runtime_observability": {},
        },
    )

    assert result["serving_measurement_source"] == "raw_response_log"
    assert result["serving_tokens"] == {
        "processed_input": 500,
        "generated_output": 100,
        "request_count": 3,
    }
    assert result["total_prompt_tokens"] == 500
    assert result["total_completion_tokens"] == 100
    assert (
        result["trajectory_tokens"]["accounting_source"]
        == "raw_response_fallback_unattributed"
    )
    assert not (tmp_path / "trajectory.jsonl").exists()


def test_v2026_9_2_model_and_tool_spans_count_as_active_runtime_work(
    tmp_path: Path,
):
    request = _model_call_span(
        span_id="request-span",
        start_ns=10_000_000_000,
        duration_ms=125.0,
        prompt_tokens=100,
        output_tokens=25,
    )
    turn = {
        **_model_call_span(
            span_id="turn-span",
            start_ns=11_000_000_000,
            duration_ms=400.0,
            prompt_tokens=0,
            output_tokens=0,
        ),
        "attrs": {
            "openclaw.model_call.observation_unit": "turn",
            "openclaw.sessionKey": "agent:roadmap_bench:main",
        },
    }
    tool = {
        "name": "openclaw.tool.execution",
        "start_unix_nano": 12_000_000_000,
        "end_unix_nano": 12_300_000_000,
        "duration_ms": 300.0,
        "attrs": {
            "openclaw.toolName": "exec",
            "gen_ai.tool.name": "exec",
            "openclaw.sessionKey": "agent:roadmap_bench:main",
        },
    }
    _write_jsonl(tmp_path / "spans.jsonl", [request, turn, tool])

    intervals = shared_measurement._active_intervals(tmp_path)

    assert len(intervals) == 2
    assert intervals[0]["activity"] == "model_request"
    assert intervals[0]["start"] == 10_000_000_000
    assert intervals[1]["activity"] == "tool:exec"
    assert intervals[1]["start"] == 12_000_000_000
