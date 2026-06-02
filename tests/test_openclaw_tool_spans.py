from __future__ import annotations

import json
from pathlib import Path

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def test_synthesizes_tool_call_content(tmp_path: Path) -> None:
    session_path = (
        tmp_path
        / "conversation"
        / "sessions"
        / "session-1.jsonl"
    )
    _write_jsonl(
        session_path,
        [
            {
                "type": "session",
                "id": "session-1",
                "timestamp": "2026-06-02T00:00:00.000Z",
            },
            {
                "type": "message",
                "timestamp": "2026-06-02T00:00:01.000Z",
                "message": {
                    "role": "assistant",
                    "sessionKey": "agent:swebench:main",
                    "timestamp": 1780358401000,
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_1",
                            "name": "edit",
                            "arguments": {
                                "path": "requests/models.py",
                                "edits": [
                                    {
                                        "oldText": "old",
                                        "newText": "new",
                                    }
                                ],
                            },
                        }
                    ],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-06-02T00:00:01.250Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "edit",
                    "timestamp": 1780358401250,
                    "content": [
                        {
                            "type": "text",
                            "text": "Successfully replaced 1 block(s).",
                        }
                    ],
                    "details": {
                        "durationMs": 250,
                        "diff": "-old\n+new",
                    },
                    "isError": False,
                },
            },
        ],
    )

    spans_path = tmp_path / "spans.jsonl"
    spans_path.write_text(
        json.dumps({"name": "agent_bench.container.start", "attrs": {}}) + "\n"
    )

    count = OpenClawAgentCondition._maybe_synthesize_tool_call_spans(
        spans_path, tmp_path, "run-1",
    )

    assert count == 1
    rows = [json.loads(line) for line in spans_path.read_text().splitlines()]
    assert rows[0]["name"] == "agent_bench.container.start"
    span = rows[1]
    attrs = span["attrs"]
    assert span["name"] == "openclaw.tool.call"
    assert attrs["openclaw.tool.name"] == "edit"
    assert attrs["openclaw.tool.callId"] == "call_1"
    assert attrs["openclaw.tool.isError"] is False
    assert "requests/models.py" in attrs["openclaw.tool.arguments_json"]
    assert "Successfully replaced" in attrs["openclaw.tool.result_preview"]
    assert "-old\\n+new" in attrs["openclaw.tool.details_json"]
    assert attrs["openclaw.tool.arguments_truncated"] is False
    assert attrs["openclaw.tool.result_truncated"] is False
    assert attrs["openclaw.sessionId"] == "session-1"
    assert attrs["openclaw.sessionKey"] == "agent:swebench:main"


def test_detects_synthetic_span_sources(tmp_path: Path) -> None:
    assert not OpenClawAgentCondition._has_synthetic_span_sources(tmp_path)

    (tmp_path / "conversation" / "sessions").mkdir(parents=True)

    assert OpenClawAgentCondition._has_synthetic_span_sources(tmp_path)
