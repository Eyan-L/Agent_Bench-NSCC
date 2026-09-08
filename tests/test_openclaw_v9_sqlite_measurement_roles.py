from __future__ import annotations

import json
from pathlib import Path

from agent_bench.benchmarks.conditions import openclaw_measurement


def _write_explicit_parent(run_dir: Path) -> str:
    parent_key = f"agent:roadmap_bench:explicit:{run_dir.name}"
    sessions = run_dir / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "sessions.json").write_text(
        json.dumps(
            {
                parent_key: {
                    "sessionId": run_dir.name,
                    "sessionFile": f"{run_dir.name}.jsonl",
                    "status": "done",
                }
            }
        )
    )
    event = {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "DONE"}],
            "stopReason": "stop",
            "responseId": "response-1",
            "timestamp": 1_700_000_000_000,
            "usage": {"input": 5, "cacheRead": 3, "output": 2},
        },
    }
    (sessions / f"{run_dir.name}.jsonl").write_text(json.dumps(event) + "\n")
    return parent_key


def test_explicit_sqlite_parent_model_usage_event_is_parent(tmp_path: Path):
    run_dir = tmp_path / "explicit-run"
    parent_key = _write_explicit_parent(run_dir)

    logical, usage_events = openclaw_measurement._logical_tokens(
        run_dir, "roadmap_bench"
    )

    assert logical["parent"]["input"] == 8
    assert logical["parent"]["output"] == 2
    assert parent_key not in logical["children"]
    parent_usage = [
        event
        for event in usage_events
        if event.get("session_key") == parent_key
        and event.get("event_type") == "model_usage"
    ]
    assert parent_usage
    assert all(event["role"] == "parent" for event in parent_usage)


def test_v9_runtime_metadata_does_not_claim_legacy_announce_limit(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("AB_AGENT_RUNTIME_VERSION", "2026.9.2")
    result = openclaw_measurement._runtime_observability(
        tmp_path, [], "roadmap_bench"
    )

    assert result["runtime_version"] == "v2026.9.2"
    limitation = result["announce_attribution"]["limitation"]
    assert "v2026.4.22" not in limitation
    assert "does not infer identity from timing" in limitation
