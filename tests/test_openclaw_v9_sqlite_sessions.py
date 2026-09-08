from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from agent_bench.benchmarks.conditions import openclaw_measurement
from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)
from agent_bench.benchmarks.roadmap_bench.conditions.sqlite_session_compat import (
    _export_sqlite_store,
    _runtime_version,
)


def _event(message: dict) -> str:
    return json.dumps({"type": "message", "message": message})


def _build_sqlite_store(
    tmp_path: Path,
    *,
    run_id: str = "current-run",
    parent_status: str = "running",
    parent_events: list[str] | None = None,
    with_child: bool = False,
) -> tuple[Path, Path, str]:
    state_dir = tmp_path / run_id
    agent_dir = state_dir / "agents" / "roadmap_bench" / "agent"
    sessions_dir = state_dir / "agents" / "roadmap_bench" / "sessions"
    agent_dir.mkdir(parents=True)
    sessions_dir.mkdir(parents=True)

    db_path = agent_dir / "openclaw-agent.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE session_nodes (
          session_key TEXT PRIMARY KEY,
          current_session_id TEXT NOT NULL,
          entry_json TEXT NOT NULL,
          updated_at INTEGER NOT NULL,
          status TEXT,
          parent_session_key TEXT,
          spawned_by TEXT
        );
        CREATE TABLE session_windows (
          session_id TEXT PRIMARY KEY,
          session_key TEXT NOT NULL,
          status TEXT,
          started_at INTEGER,
          ended_at INTEGER,
          created_at INTEGER NOT NULL
        );
        CREATE TABLE transcript_events (
          session_id TEXT NOT NULL,
          seq INTEGER NOT NULL,
          event_json TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          PRIMARY KEY (session_id, seq)
        );
        """
    )

    parent_key = f"agent:roadmap_bench:explicit:{run_id}"
    parent_session_id = "parent-sqlite-session"
    connection.execute(
        """
        INSERT INTO session_nodes
          (session_key, current_session_id, entry_json, updated_at, status,
           parent_session_key, spawned_by)
        VALUES (?, ?, '{}', 1000, ?, NULL, NULL)
        """,
        (parent_key, parent_session_id, parent_status),
    )
    connection.execute(
        """
        INSERT INTO session_windows
          (session_id, session_key, status, started_at, ended_at, created_at)
        VALUES (?, ?, ?, 1000, ?, 1000)
        """,
        (
            parent_session_id,
            parent_key,
            parent_status,
            2000 if parent_status == "done" else None,
        ),
    )

    for seq, raw in enumerate(parent_events or []):
        connection.execute(
            "INSERT INTO transcript_events VALUES (?, ?, ?, ?)",
            (parent_session_id, seq, raw, 1000 + seq),
        )

    if with_child:
        child_key = "agent:roadmap_bench:subagent:child-1"
        connection.execute(
            """
            INSERT INTO session_nodes
              (session_key, current_session_id, entry_json, updated_at, status,
               parent_session_key, spawned_by)
            VALUES (?, 'child-session', '{}', 1100, 'running', ?, ?)
            """,
            (child_key, parent_key, parent_key),
        )
        connection.execute(
            """
            INSERT INTO session_windows
              (session_id, session_key, status, started_at, ended_at, created_at)
            VALUES ('child-session', ?, 'running', 1100, NULL, 1100)
            """,
            (child_key,),
        )

    connection.commit()
    connection.close()
    return sessions_dir, db_path, parent_key


def test_sqlite_watcher_keeps_working_parent_nonterminal(tmp_path: Path):
    events = [
        _event(
            {
                "role": "user",
                "content": [{"type": "text", "text": "implement the roadmap"}],
                "timestamp": 1000,
            }
        ),
        _event(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Let me check the current state and continue.",
                    },
                    {
                        "type": "toolCall",
                        "id": "call-1",
                        "name": "exec",
                        "arguments": {"command": "git status"},
                    },
                ],
                "stopReason": "stop",
                "timestamp": 1100,
            }
        ),
    ]
    sessions_dir, _db_path, parent_key = _build_sqlite_store(
        tmp_path, parent_events=events
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["session_store"] == "sqlite"
    assert snapshot["parent_session_key"] == parent_key
    assert snapshot["parent_session_id"] == "parent-sqlite-session"
    assert snapshot["parent_session_active"] is True
    assert snapshot["assistant_stop_candidate"] is True
    assert snapshot["parent_terminal"] is False
    assert snapshot["done_text"] == ""


def test_sqlite_watcher_accepts_explicit_done_and_counts_children(tmp_path: Path):
    events = [
        _event(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "DONE\nAll targets validated."}],
                "stopReason": "stop",
                "timestamp": 2000,
            }
        )
    ]
    sessions_dir, _db_path, _parent_key = _build_sqlite_store(
        tmp_path,
        parent_status="done",
        parent_events=events,
        with_child=True,
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["parent_session_terminal"] is True
    assert snapshot["parent_terminal"] is True
    assert snapshot["done_text"].startswith("DONE")
    assert snapshot["total_children"] == 1
    assert snapshot["pending_children"] == 1


def test_sqlite_projection_preserves_explicit_parent_for_measurement(tmp_path: Path):
    events = [
        _event(
            {
                "role": "user",
                "content": [{"type": "text", "text": "roadmap"}],
                "timestamp": 1000,
            }
        ),
        _event(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "DONE"}],
                "stopReason": "stop",
                "timestamp": 2000,
                "responseId": "response-1",
                "usage": {"input": 5, "output": 2, "cacheRead": 3},
            }
        ),
    ]
    _sessions_dir, db_path, parent_key = _build_sqlite_store(
        tmp_path, parent_status="done", parent_events=events
    )
    run_dir = tmp_path / "measurement" / "current-run"
    projected = run_dir / "conversation" / "sessions"

    assert _export_sqlite_store(db_path, projected) is True
    index = json.loads((projected / "sessions.json").read_text())
    assert parent_key in index
    assert index[parent_key]["sessionId"] == "parent-sqlite-session"

    logical, _usage_events = openclaw_measurement._logical_tokens(
        run_dir, "roadmap_bench"
    )
    assert logical["parent"]["input"] == 8
    assert logical["parent"]["output"] == 2
    assert parent_key not in logical["children"]

    trajectory = openclaw_measurement._session_events(run_dir, "roadmap_bench")
    parent_events = [
        event for event in trajectory if event.get("session_key") == parent_key
    ]
    assert parent_events
    assert all(event["role"] == "parent" for event in parent_events)


def test_runtime_observability_version_uses_active_runtime(monkeypatch):
    monkeypatch.setenv("AB_AGENT_RUNTIME_VERSION", "2026.9.2")
    assert _runtime_version() == "v2026.9.2"
