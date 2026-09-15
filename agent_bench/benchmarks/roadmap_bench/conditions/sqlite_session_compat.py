"""OpenClaw v2026.9.2 SQLite session-store compatibility for RoadmapBench.

The v9 runtime persists session metadata and transcript events in the per-agent
``openclaw-agent.sqlite`` database.  RoadmapBench historically watched/exported
the legacy ``sessions.json`` + JSONL store.  This adapter keeps the lifecycle
semantics unchanged while sourcing the same facts from SQLite, and exports a
legacy-compatible read-only snapshot for the existing measurement pipeline.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import Dict, Optional, Tuple
from urllib.parse import quote

logger = logging.getLogger(__name__)

_OPENCLAW_V9 = "2026.9.2"
_TERMINAL_CHILD_STATES = frozenset(
    {"done", "completed", "failed", "error", "timeout", "timed_out", "cancelled", "canceled", "killed"}
)


def _agent_db_from_sessions_dir(sessions_dir: Path) -> Path:
    return sessions_dir.parent / "agent" / "openclaw-agent.sqlite"


def _agent_db_from_state_dir(state_dir: Path, agent_id: str) -> Path:
    return state_dir / "agents" / agent_id / "agent" / "openclaw-agent.sqlite"


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=0.5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=500")
    return connection


def _safe_transcript_name(session_id: str) -> str:
    if session_id and Path(session_id).name == session_id and session_id not in {".", ".."}:
        return f"{session_id}.jsonl"
    # Session ids are UUID-like in OpenClaw, so this path is only a defensive
    # fallback for malformed/corrupt rows.
    import hashlib

    digest = hashlib.sha256(session_id.encode("utf-8", errors="replace")).hexdigest()
    return f"session-{digest[:24]}.jsonl"


def _normalize_entry(row: sqlite3.Row) -> dict:
    try:
        raw = json.loads(str(row["entry_json"] or "{}"))
    except (json.JSONDecodeError, TypeError):
        raw = {}
    entry = dict(raw) if isinstance(raw, dict) else {}

    session_id = str(row["current_session_id"] or "").strip()
    status = str(
        row["node_status"]
        or row["window_status"]
        or entry.get("status")
        or "unknown"
    ).strip().lower()

    entry.update(
        {
            "sessionId": session_id,
            "status": status,
            "updatedAt": row["node_updated_at"],
            "startedAt": row["started_at"],
            "endedAt": row["ended_at"],
            "createdAt": row["created_at"],
            "sessionFile": _safe_transcript_name(session_id),
        }
    )
    spawned_by = row["spawned_by"]
    parent_key = row["parent_session_key"]
    if spawned_by:
        entry["spawnedBy"] = str(spawned_by)
    else:
        entry.pop("spawnedBy", None)
    if parent_key:
        entry["parentSessionKey"] = str(parent_key)
    else:
        entry.pop("parentSessionKey", None)
    return entry


def _read_sqlite_store(
    db_path: Path,
) -> Optional[tuple[dict[str, dict], dict[str, list[str]], dict[str, int]]]:
    """Read one consistent SQLite snapshot.

    Returns an index compatible with the legacy sessions.json shape, exact
    persisted transcript event JSON strings keyed by current session id, and
    byte counts used only as a stable activity/fence signature.
    """
    if not db_path.is_file():
        return None

    connection: Optional[sqlite3.Connection] = None
    try:
        connection = _readonly_connection(db_path)
        connection.execute("BEGIN")
        required = {"session_nodes", "session_windows", "transcript_events"}
        # ``sqlite_schema`` is only an alias on newer SQLite releases.  NSCC's
        # Python runtime can be linked against an older library, while
        # ``sqlite_master`` is supported by both old and new versions.
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        present = {str(row[0]) for row in rows}
        if not required.issubset(present):
            return None

        node_rows = connection.execute(
            """
            SELECT
                n.session_key,
                n.current_session_id,
                n.entry_json,
                n.updated_at AS node_updated_at,
                n.status AS node_status,
                n.parent_session_key,
                n.spawned_by,
                w.status AS window_status,
                w.started_at,
                w.ended_at,
                w.created_at
            FROM session_nodes AS n
            LEFT JOIN session_windows AS w
              ON w.session_id = n.current_session_id
            """
        ).fetchall()

        index: dict[str, dict] = {}
        session_ids: list[str] = []
        for row in node_rows:
            session_key = str(row["session_key"] or "")
            if not session_key:
                continue
            entry = _normalize_entry(row)
            index[session_key] = entry
            session_id = str(entry.get("sessionId") or "")
            if session_id:
                session_ids.append(session_id)

        events: dict[str, list[str]] = {}
        sizes: dict[str, int] = {}
        for session_id in dict.fromkeys(session_ids):
            event_rows = connection.execute(
                """
                SELECT event_json
                FROM transcript_events
                WHERE session_id = ?
                ORDER BY seq
                """,
                (session_id,),
            ).fetchall()
            raw_events = [str(row["event_json"]) for row in event_rows]
            events[session_id] = raw_events
            sizes[session_id] = sum(
                len(item.encode("utf-8", errors="replace")) + 1
                for item in raw_events
            )
        return index, events, sizes
    except sqlite3.Error as exc:
        logger.debug("Roadmap SQLite session snapshot unavailable: %s", exc)
        return None
    finally:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass


def _expected_run_id(sessions_dir: Path) -> str:
    try:
        return sessions_dir.parents[2].name.strip()
    except (IndexError, AttributeError):
        return ""


def _select_sqlite_parent(
    cls,
    sessions_dir: Path,
    index: dict[str, dict],
) -> Optional[Tuple[str, dict]]:
    """Select the current root deterministically without recency heuristics."""
    prefix = f"agent:{cls.agent_id}:"
    expected_run_id = _expected_run_id(sessions_dir)

    def is_root(value: object) -> bool:
        return (
            isinstance(value, dict)
            and not value.get("spawnedBy")
            and not value.get("parentSessionKey")
            and bool(value.get("sessionId"))
        )

    if expected_run_id:
        key = f"{prefix}explicit:{expected_run_id}"
        value = index.get(key)
        if is_root(value):
            return key, value

    roots = [
        (key, value)
        for key, value in index.items()
        if isinstance(key, str) and key.startswith(prefix) and is_root(value)
    ]
    if len(roots) != 1:
        return None

    key, value = roots[0]
    if key == f"{prefix}main" or key.startswith(f"{prefix}explicit:"):
        return key, value
    return None


def _event_messages(raw_events: list[str]) -> list[dict]:
    messages: list[dict] = []
    for raw in raw_events:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") in {
            "message",
            "custom",
            "custom_message",
        }:
            messages.append(event)
    return messages


def _sqlite_yield_snapshot(cls, sessions_dir: Path) -> Optional[Dict]:
    store = _read_sqlite_store(_agent_db_from_sessions_dir(sessions_dir))
    if store is None:
        return None
    index, transcripts, transcript_sizes = store

    selected = _select_sqlite_parent(cls, sessions_dir, index)
    if selected is None:
        return None
    parent_key, parent = selected
    parent_session_id = str(parent.get("sessionId") or "")
    if not parent_session_id:
        return None

    parent_session_status = str(parent.get("status") or "unknown").strip().lower()
    parent_session_active = parent_session_status in cls._ACTIVE_PARENT_STATES
    parent_session_terminal = parent_session_status in cls._TERMINAL_PARENT_STATES

    child_entries: list[dict] = []
    for key, value in index.items():
        if not isinstance(value, dict):
            continue
        lineage = str(value.get("spawnedBy") or value.get("parentSessionKey") or "")
        if lineage != parent_key:
            continue
        status = str(value.get("status") or "unknown").strip().lower()
        child_entries.append(
            {
                "session_key": str(key),
                "session_id": str(value.get("sessionId") or ""),
                "status": status,
            }
        )

    pending_child_sessions = [
        child
        for child in child_entries
        if child["status"] in cls._ACTIVE_CHILD_STATES
    ]
    completed_children = sum(
        child["status"] in _TERMINAL_CHILD_STATES for child in child_entries
    )
    child_timeout = any(
        child["status"] in {"timeout", "timed_out"} for child in child_entries
    )

    messages = _event_messages(transcripts.get(parent_session_id, []))
    yield_indexes = [
        i
        for i, event in enumerate(messages)
        if event.get("type") == "custom_message"
        and event.get("customType") == "openclaw.sessions_yield"
    ]
    latest_yield = yield_indexes[-1] if yield_indexes else -1

    done_text = ""
    post_yield_parent_activity = False
    post_yield_parent_terminal = False
    post_yield_prompt_error = ""
    latest_event_kind = ""
    heartbeat_turn = False

    for event in messages[latest_yield + 1 :]:
        if (
            event.get("type") == "custom"
            and event.get("customType") == "openclaw:prompt-error"
        ):
            data = event.get("data")
            raw_error = data.get("error") if isinstance(data, dict) else data
            post_yield_prompt_error = str(raw_error or "OpenClaw prompt error")
            post_yield_parent_terminal = False
            done_text = ""
            latest_event_kind = "prompt_error"
            continue

        if event.get("type") != "message":
            continue

        message = event.get("message") or {}
        role = message.get("role")
        parts = [
            str(item.get("text") or "")
            for item in message.get("content") or []
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part).strip()

        if role in {"user", "assistant", "toolResult"}:
            post_yield_prompt_error = ""

        if role == "user":
            if cls._is_heartbeat_request(text):
                heartbeat_turn = True
                latest_event_kind = "heartbeat_request"
                continue
            heartbeat_turn = False
            post_yield_parent_terminal = False
            done_text = ""
            latest_event_kind = "resume"
            continue

        if role == "toolResult":
            if heartbeat_turn:
                latest_event_kind = "heartbeat_tool_result"
                continue
            post_yield_parent_activity = True
            post_yield_parent_terminal = False
            done_text = ""
            latest_event_kind = "tool_result"
            continue

        if role != "assistant":
            continue

        stop_reason = str(message.get("stopReason") or "").strip().lower()
        if heartbeat_turn:
            latest_event_kind = "heartbeat_response"
            if stop_reason == "stop":
                heartbeat_turn = False
            continue

        post_yield_parent_activity = True
        post_yield_parent_terminal = stop_reason == "stop"
        latest_event_kind = (
            "assistant_stop" if post_yield_parent_terminal else "assistant_active"
        )
        if (
            post_yield_parent_terminal
            and text
            and re.search(r"(?im)^\s*DONE(?:\s*$|\b)", text)
        ):
            done_text = text
        else:
            done_text = ""

    assistant_stop_candidate = post_yield_parent_terminal
    logical_parent_terminal = bool(
        done_text or (assistant_stop_candidate and parent_session_terminal)
    )
    return {
        "session_store": "sqlite",
        "parent_session_key": str(parent_key),
        "parent_session_id": parent_session_id,
        "parent_session_status": parent_session_status,
        "parent_session_active": parent_session_active,
        "parent_session_terminal": parent_session_terminal,
        "total_children": len(child_entries),
        "pending_children": len(pending_child_sessions),
        "pending_child_sessions": pending_child_sessions,
        "completed_children": completed_children,
        "child_states": child_entries,
        "child_timeout": child_timeout,
        "yield_count": len(yield_indexes),
        "message_count": len(messages),
        "transcript_size_bytes": transcript_sizes.get(parent_session_id, 0),
        "post_yield_parent_activity": post_yield_parent_activity,
        "post_yield_parent_terminal": (
            assistant_stop_candidate and parent_session_terminal
        ),
        "post_yield_terminal_reason": (
            "explicit_done"
            if done_text
            else (
                "assistant_stop"
                if assistant_stop_candidate and parent_session_terminal
                else ""
            )
        ),
        "post_yield_prompt_error": post_yield_prompt_error,
        "done_text": done_text,
        "parent_active": bool(
            (
                parent_session_active
                or (
                    post_yield_parent_activity
                    and not logical_parent_terminal
                )
            )
            and not post_yield_prompt_error
        ),
        "parent_waiting": bool(yield_indexes and not post_yield_parent_activity),
        "parent_resumed": bool(yield_indexes and post_yield_parent_activity),
        "parent_terminal": logical_parent_terminal,
        "assistant_stop_candidate": assistant_stop_candidate,
        "latest_parent_event": latest_event_kind,
    }


def _export_sqlite_store(db_path: Path, sessions_dir: Path) -> bool:
    store = _read_sqlite_store(db_path)
    if store is None:
        return False
    index, transcripts, _sizes = store
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sessions.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for entry in index.values():
        if not isinstance(entry, dict):
            continue
        session_id = str(entry.get("sessionId") or "")
        name = str(entry.get("sessionFile") or "")
        if not session_id or not name or Path(name).name != name:
            continue
        path = sessions_dir / name
        rows = transcripts.get(session_id, [])
        text = "".join(f"{row}\n" for row in rows)
        path.write_text(text, encoding="utf-8")
    return True


def _archive_sqlite_database(db_path: Path, destination: Path) -> None:
    """Create a consistent post-run SQLite backup including committed WAL rows."""
    if not db_path.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source: Optional[sqlite3.Connection] = None
    target: Optional[sqlite3.Connection] = None
    try:
        source = _readonly_connection(db_path)
        target = sqlite3.connect(destination)
        source.backup(target)
    except sqlite3.Error:
        # OpenClaw v9 uses STRICT tables, which NSCC's Python-linked SQLite
        # 3.26 cannot parse. The gateway has stopped before archival, so a
        # direct copy is a consistent fallback without requiring host SQLite.
        if target is not None:
            target.close()
            target = None
        shutil.copy2(db_path, destination)
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()


def _measurement_parent_key(run_dir: Path, agent_id: str) -> str:
    sessions_dir = run_dir / "conversation" / "sessions"
    try:
        index = json.loads((sessions_dir / "sessions.json").read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(index, dict):
        return ""

    explicit = f"agent:{agent_id}:explicit:{run_dir.name}"
    value = index.get(explicit)
    if isinstance(value, dict) and not value.get("spawnedBy") and not value.get("parentSessionKey"):
        return explicit

    roots = [
        str(key)
        for key, entry in index.items()
        if isinstance(key, str)
        and key.startswith(f"agent:{agent_id}:")
        and isinstance(entry, dict)
        and not entry.get("spawnedBy")
        and not entry.get("parentSessionKey")
    ]
    main = f"agent:{agent_id}:main"
    if main in roots:
        return main
    return roots[0] if len(roots) == 1 else ""


def _count_parent_resumes(run_dir: Path, parent_key: str) -> int:
    return openclaw_measurement._parent_resume_count(
        run_dir,
        "roadmap_bench",
        parent_session_key=parent_key,
    )


def _runtime_version() -> str:
    raw = str(os.environ.get("AB_AGENT_RUNTIME_VERSION") or "").strip()
    if not raw:
        return f"v{_OPENCLAW_V9}"
    return raw if raw.startswith("v") else f"v{raw}"


def install_roadmap_sqlite_session_compat() -> None:
    """Install SQLite lifecycle, archival, and measurement compatibility once."""
    from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition
    from agent_bench.benchmarks.conditions import openclaw_measurement
    from .openclaw_agent import RoadmapBenchOpenClawCondition

    # Lifecycle: prefer the v9 SQLite store, retain the existing JSON fallback.
    current_read = RoadmapBenchOpenClawCondition._read_yield_snapshot
    underlying_read = getattr(current_read, "__func__", current_read)
    if not getattr(underlying_read, "_roadmap_sqlite_session_compat", False):
        def read_snapshot(cls, sessions_dir: Path):
            # The runtime-side Node exporter can read v9 STRICT databases even
            # when host Python cannot. Prefer that durable projection when it
            # has been published, retaining direct SQLite for tests/new hosts.
            snapshot = underlying_read(cls, sessions_dir)
            if snapshot is not None:
                return snapshot
            return _sqlite_yield_snapshot(cls, sessions_dir)

        setattr(read_snapshot, "_roadmap_sqlite_session_compat", True)
        RoadmapBenchOpenClawCondition._read_yield_snapshot = classmethod(read_snapshot)

    # Archive a stable legacy-compatible projection after the gateway stops so
    # existing post-run measurement code can consume v9 without schema forks.
    current_archive = OpenClawAgentCondition._archive_session_artifacts
    if not getattr(current_archive, "_roadmap_sqlite_session_compat", False):
        def archive(self, container_id, run_dir: Path, session_id: str):
            current_archive(self, container_id, run_dir, session_id)
            if getattr(self, "agent_id", "") != "roadmap_bench":
                return
            env_info = getattr(self, "_env_info", {}) or {}
            state_dir = env_info.get("state_dir") or env_info.get("host_state_dir")
            if not state_dir:
                return
            db_path = _agent_db_from_state_dir(Path(state_dir), self.agent_id)
            if not db_path.is_file():
                return
            out_dir = run_dir / "conversation"
            try:
                sessions_out = out_dir / "sessions"
                if not (sessions_out / "sessions.json").is_file():
                    if _export_sqlite_store(db_path, sessions_out):
                        logger.info("  archived OpenClaw SQLite session projection")
                _archive_sqlite_database(db_path, out_dir / "openclaw-agent.sqlite")
            except Exception as exc:  # noqa: BLE001 - measurement remains best-effort
                logger.warning("OpenClaw SQLite session archive failed: %s", exc)

        setattr(archive, "_roadmap_sqlite_session_compat", True)
        OpenClawAgentCondition._archive_session_artifacts = archive

    # Preserve exact explicit session keys in the exported projection while
    # teaching the legacy measurement aggregation which root is the parent.
    current_session_events = openclaw_measurement._session_events
    if not getattr(current_session_events, "_roadmap_sqlite_session_compat", False):
        def session_events(run_dir: Path, agent_id: str):
            events = current_session_events(run_dir, agent_id)
            if agent_id != "roadmap_bench":
                return events
            parent_key = _measurement_parent_key(run_dir, agent_id)
            if parent_key:
                for event in events:
                    if event.get("session_key") == parent_key:
                        event["role"] = "parent"
            return events

        setattr(session_events, "_roadmap_sqlite_session_compat", True)
        openclaw_measurement._session_events = session_events

    current_logical = openclaw_measurement._logical_tokens
    if not getattr(current_logical, "_roadmap_sqlite_session_compat", False):
        def logical_tokens(run_dir: Path, agent_id: str):
            logical, events = current_logical(run_dir, agent_id)
            if agent_id != "roadmap_bench":
                return logical, events
            parent_key = _measurement_parent_key(run_dir, agent_id)
            legacy_main = f"agent:{agent_id}:main"
            if parent_key and parent_key != legacy_main:
                children = logical.get("children") or {}
                explicit_parent = children.pop(parent_key, None)
                if isinstance(explicit_parent, dict):
                    parent = dict(explicit_parent)
                    parent["resumed_turns"] = _count_parent_resumes(
                        run_dir, parent_key
                    )
                    logical["parent"] = parent
            return logical, events

        setattr(logical_tokens, "_roadmap_sqlite_session_compat", True)
        openclaw_measurement._logical_tokens = logical_tokens

    current_runtime = openclaw_measurement._runtime_observability
    if not getattr(current_runtime, "_roadmap_sqlite_session_compat", False):
        def runtime_observability(
            run_dir: Path,
            events: list[dict],
            agent_id: str,
            *,
            execution_start_ns: int = 0,
            execution_end_ns: int = 0,
        ):
            result = current_runtime(
                run_dir,
                events,
                agent_id,
                execution_start_ns=execution_start_ns,
                execution_end_ns=execution_end_ns,
            )
            if agent_id != "roadmap_bench":
                return result
            version = _runtime_version()
            result["runtime_version"] = version
            policy = result.get("stuck_diagnostic_policy")
            if isinstance(policy, dict):
                policy["runtime_version"] = version
            return result

        setattr(runtime_observability, "_roadmap_sqlite_session_compat", True)
        openclaw_measurement._runtime_observability = runtime_observability
