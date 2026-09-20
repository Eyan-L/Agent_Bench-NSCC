from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess

from agent_bench.benchmarks.conditions import openclaw_measurement
from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)
from agent_bench.benchmarks.roadmap_bench.conditions.sqlite_session_compat import (
    _archive_sqlite_database,
    _export_sqlite_store,
    _restore_archived_child_index,
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
    parent_last_run_id: str | None = None,
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
    parent_entry = json.dumps(
        {"lastRunId": parent_last_run_id} if parent_last_run_id else {}
    )
    connection.execute(
        """
        INSERT INTO session_nodes
          (session_key, current_session_id, entry_json, updated_at, status,
           parent_session_key, spawned_by)
        VALUES (?, ?, ?, 1000, ?, NULL, NULL)
        """,
        (parent_key, parent_session_id, parent_entry, parent_status),
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


def test_sqlite_watcher_keeps_requester_settle_no_reply_pending(tmp_path: Path):
    settle_run = "announce:requester-settle:roadmap_bench:parent:retry-1"
    events = [
        _event(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "NO_REPLY"}],
                "stopReason": "stop",
                "timestamp": 2000,
                "__openclaw": {"runId": settle_run},
            }
        )
    ]
    sessions_dir, _db_path, _parent_key = _build_sqlite_store(
        tmp_path,
        parent_status="done",
        parent_events=events,
        parent_last_run_id=settle_run,
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["session_store"] == "sqlite"
    assert snapshot["latest_assistant_text"] == "NO_REPLY"
    assert snapshot["parent_last_run_id"] == settle_run
    assert snapshot["requester_settle_pending"] is True
    assert snapshot["parent_terminal"] is False
    assert snapshot["post_yield_parent_terminal"] is False
    assert snapshot["post_yield_terminal_reason"] == ""
    assert RoadmapBenchOpenClawCondition._terminal_stop_signature(snapshot) is None


def test_sqlite_watcher_preserves_done_across_requester_settle_no_reply(
    tmp_path: Path,
):
    settle_run = "announce:requester-settle:roadmap_bench:parent:retry-1"
    events = [
        _event({
            "role": "assistant",
            "content": [{"type": "text", "text": "DONE\nValidated."}],
            "stopReason": "stop",
            "timestamp": 2000,
            "__openclaw": {"runId": "parent-run"},
        }),
        _event({
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "[Subagent Context] Every subagent spawned from this "
                    "session has now settled."
                ),
            }],
            "timestamp": 2500,
        }),
        _event({
            "role": "assistant",
            "content": [{"type": "text", "text": "NO_REPLY"}],
            "stopReason": "stop",
            "timestamp": 3000,
            "__openclaw": {"runId": settle_run},
        }),
    ]
    sessions_dir, _db_path, _parent_key = _build_sqlite_store(
        tmp_path,
        parent_status="done",
        parent_events=events,
        parent_last_run_id=settle_run,
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["session_store"] == "sqlite"
    assert snapshot["done_text"].startswith("DONE")
    assert snapshot["requester_settle_pending"] is False
    assert snapshot["parent_terminal"] is True
    assert snapshot["post_yield_terminal_reason"] == "explicit_done"
    assert snapshot["latest_parent_event"] == "post_terminal_settle_no_reply"


def test_sqlite_watcher_preserves_done_across_post_terminal_settle_error(
    tmp_path: Path,
):
    settle_run = "announce:requester-settle:roadmap_bench:parent:retry-1"
    events = [
        _event({
            "role": "assistant",
            "content": [{"type": "text", "text": "DONE\nValidated."}],
            "stopReason": "stop",
            "timestamp": 2000,
            "__openclaw": {"runId": "parent-run"},
        }),
        _event({
            "role": "assistant",
            "content": [],
            "stopReason": "error",
            "errorMessage": "requester settle transport error",
            "timestamp": 3000,
            "__openclaw": {"runId": settle_run},
        }),
    ]
    sessions_dir, _db_path, _parent_key = _build_sqlite_store(
        tmp_path,
        parent_status="done",
        parent_events=events,
        parent_last_run_id=settle_run,
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["done_text"].startswith("DONE")
    assert snapshot["parent_terminal"] is True
    assert snapshot["latest_parent_event"] == "post_terminal_settle_error"
    assert "transport error" in snapshot["post_terminal_settle_error"]
    assert snapshot["post_yield_terminal_reason"] == "explicit_done"
    assert RoadmapBenchOpenClawCondition._terminal_stop_signature(snapshot) is None


def test_sqlite_watcher_classifies_done_empty_stop_as_incomplete(
    tmp_path: Path,
):
    events = [
        _event(
            {
                "role": "assistant",
                "content": [],
                "stopReason": "stop",
                "timestamp": 2000,
            }
        )
    ]
    sessions_dir, _db_path, _parent_key = _build_sqlite_store(
        tmp_path,
        parent_status="done",
        parent_events=events,
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["session_store"] == "sqlite"
    assert snapshot["latest_assistant_text"] == ""
    assert snapshot["latest_assistant_has_payload"] is False
    assert snapshot["latest_assistant_has_tool_call"] is False
    assert RoadmapBenchOpenClawCondition._is_durable_empty_assistant_stop_terminal(
        snapshot
    )


def test_sqlite_watcher_keeps_ordinary_no_reply_terminal(tmp_path: Path):
    events = [
        _event(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "NO_REPLY"}],
                "stopReason": "stop",
                "timestamp": 2000,
                "__openclaw": {"runId": "ordinary-parent-run"},
            }
        )
    ]
    sessions_dir, _db_path, _parent_key = _build_sqlite_store(
        tmp_path,
        parent_status="done",
        parent_events=events,
        parent_last_run_id="ordinary-parent-run",
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["requester_settle_pending"] is False
    assert snapshot["parent_terminal"] is True
    assert RoadmapBenchOpenClawCondition._terminal_stop_signature(snapshot)


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


def test_archived_child_is_restored_from_durable_cache_trace(tmp_path: Path):
    run_dir = tmp_path / "archived-child-run"
    conversation = run_dir / "conversation"
    sessions = conversation / "sessions"
    sessions.mkdir(parents=True)
    parent_key = f"agent:roadmap_bench:explicit:{run_dir.name}"
    child_key = "agent:roadmap_bench:subagent:child-1"
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": run_dir.name,
            "sessionFile": f"{run_dir.name}.jsonl",
            "status": "done",
        }
    }))
    parent_events = [
        {
            "type": "message",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "content": [{
                    "type": "toolCall",
                    "id": "spawn-1",
                    "name": "sessions_spawn",
                    "arguments": {"task": "child work"},
                }],
            },
        },
        {
            "type": "message",
            "timestamp": "2026-01-01T00:00:06Z",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "DONE"}],
            },
        },
    ]
    (sessions / f"{run_dir.name}.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in parent_events)
    )
    child_events = [
        {
            "type": "message",
            "timestamp": "2026-01-01T00:00:02Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "child work"}],
            },
        },
        {
            "type": "message",
            "timestamp": "2026-01-01T00:00:05Z",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "responseId": "child-response",
                "usage": {"input": 11, "cacheRead": 3, "output": 5},
                "content": [{"type": "text", "text": "child complete"}],
            },
        },
    ]
    (sessions / "child-session.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in child_events)
    )
    (conversation / "cache-trace.jsonl").write_text(json.dumps({
        "sessionId": "child-session",
        "sessionKey": child_key,
        "stage": "session:loaded",
    }) + "\n")

    assert _restore_archived_child_index(sessions, "roadmap_bench") == 1
    assert _restore_archived_child_index(sessions, "roadmap_bench") == 0

    index = json.loads((sessions / "sessions.json").read_text())
    assert index[child_key]["spawnedBy"] == parent_key
    assert index[child_key]["sessionId"] == "child-session"
    assert index[child_key]["status"] == "done"

    result = openclaw_measurement.build_openclaw_measurement(
        run_dir, agent_id="roadmap_bench"
    )
    runtime = result["runtime_observability"]
    assert runtime["child_session_count"] == 1
    assert runtime["child_runtime_complete_count"] == 1
    assert runtime["unmaterialized_spawn_attempt_count"] == 0
    assert runtime["max_concurrent_children"] == 1
    assert result["trajectory_tokens"]["children"][child_key] == {
        "input": 14,
        "output": 5,
        "cache_read": 3,
        "turn_count": 1,
    }
    trajectory = [
        json.loads(line)
        for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event_type"] == "child_completion"
        and event["session_key"] == child_key
        for event in trajectory
    )


def test_runtime_observability_version_uses_active_runtime(monkeypatch):
    monkeypatch.setenv("AB_AGENT_RUNTIME_VERSION", "2026.9.2")
    assert _runtime_version() == "v2026.9.2"


def test_v9_watcher_prefers_runtime_projection_when_host_sqlite_is_unreadable(
    tmp_path: Path,
):
    sessions_dir = tmp_path / "run" / "agents" / "roadmap_bench" / "sessions"
    sessions_dir.mkdir(parents=True)
    parent_key = "agent:roadmap_bench:explicit:parent"
    (sessions_dir / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": "parent",
            "sessionFile": "parent.jsonl",
            "status": "done",
        }
    }))
    (sessions_dir / "parent.jsonl").write_text(_event({
        "role": "assistant",
        "content": [{"type": "text", "text": "DONE"}],
        "stopReason": "stop",
    }) + "\n")
    db_path = sessions_dir.parent / "agent" / "openclaw-agent.sqlite"
    db_path.parent.mkdir()
    db_path.write_bytes(b"not readable by host sqlite")

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot is not None
    assert snapshot["parent_session_key"] == parent_key
    assert snapshot["parent_terminal"] is True


def test_v9_database_archive_falls_back_to_file_copy(tmp_path: Path, monkeypatch):
    source = tmp_path / "openclaw-agent.sqlite"
    destination = tmp_path / "archive" / "openclaw-agent.sqlite"
    source.write_bytes(b"strict-sqlite-placeholder")

    def unsupported(_path):
        raise sqlite3.OperationalError("near STRICT: syntax error")

    module = __import__(
        "agent_bench.benchmarks.roadmap_bench.conditions.sqlite_session_compat",
        fromlist=["_readonly_connection"],
    )
    monkeypatch.setattr(module, "_readonly_connection", unsupported)

    _archive_sqlite_database(source, destination)

    assert destination.read_bytes() == source.read_bytes()


def test_nonzero_openclaw_exit_preserves_structured_error_payload():
    payload = {
        "status": "error",
        "result": {
            "meta": {
                "livenessState": "abandoned",
                "replayInvalid": True,
                "error": {"kind": "incomplete_turn", "message": "Connection error"},
            }
        },
    }

    class ErrorContainer:
        backend_id = "singularity"

        def exec(self, *_args, **_kwargs):
            return subprocess.CompletedProcess([], 1, json.dumps(payload), "transport failed")

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition.container = ErrorContainer()
    condition._env_info = {}
    condition._watch_durable_invocation_terminal = lambda: False

    result, error = condition._invoke_agent(
        "oc-runtime", "/workspace", "run", "task", 60
    )

    assert result == payload
    assert condition._last_exit_code == 1
    assert condition._last_completion_status == condition.RUNTIME_ERROR
    assert "incomplete_turn" in error
    assert condition._last_stdout == json.dumps(payload)


def _write_explicit_parent(run_dir: Path) -> str:
    parent_key = f"agent:roadmap_bench:explicit:{run_dir.name}"
    sessions = run_dir / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "sessions.json").write_text(json.dumps({
        parent_key: {
            "sessionId": run_dir.name,
            "sessionFile": f"{run_dir.name}.jsonl",
            "status": "done",
        }
    }))
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
