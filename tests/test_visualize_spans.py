from __future__ import annotations

from pathlib import Path

from agent_bench.telemetry.otel.visualize_spans import render_html


def _span(name: str, start_ms: int, end_ms: int) -> dict:
    return {
        "line": 1,
        "name": name,
        "channel": "test",
        "start": start_ms * 1_000_000,
        "end": end_ms * 1_000_000,
        "duration_ms": end_ms - start_ms,
        "tool": "exec" if "tool" in name else "",
        "returncode": None,
        "run_id": "run-1",
        "attrs": {},
        "resource": {},
    }


def test_render_html_excludes_agent_tool_time_when_runtime_tool_spans_exist() -> None:
    spans = [
        _span("agent.tool.call", 0, 100),
        _span("openclaw.tool.call", 0, 100),
        _span("openclaw.model.usage", 100, 250),
    ]

    html = render_html(spans, Path("spans.jsonl"))

    assert "Time (E2E, Deduped)" in html
    assert "Tool calls (agent trace)</span><span>100.0 ms excluded" in html
    assert '"name": "agent.tool.call", "e2e_time_bucket": "Tool calls (agent trace)", "e2e_time_included": false' in html
    assert '"name": "openclaw.tool.call", "e2e_time_bucket": "Tool calls (runtime)", "e2e_time_included": true' in html
