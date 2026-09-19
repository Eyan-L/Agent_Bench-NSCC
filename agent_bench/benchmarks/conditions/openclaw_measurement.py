"""Build machine-readable OpenClaw trajectory and token artifacts."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Iterable, Optional


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _timestamp_ns(value: Any) -> int:
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1e15:
            return int(number)
        if number > 1e12:
            return int(number * 1e6)
        return int(number * 1e9)
    if isinstance(value, str) and value:
        try:
            return int(
                datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                * 1e9
            )
        except ValueError:
            return 0
    return 0


def _iso_from_ns(value: int) -> Optional[str]:
    if not value:
        return None
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc).isoformat()


def _message_text(message: dict) -> str:
    content = message.get("content") or []
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _is_heartbeat_request(text: str, message: Optional[dict] = None) -> bool:
    """Recognize OpenClaw heartbeat requester turns."""
    normalized = " ".join(str(text or "").lower().split())
    provenance = (message or {}).get("provenance") or {}
    return (
        ("read heartbeat.md" in normalized and "heartbeat_ok" in normalized)
        or normalized == "[openclaw heartbeat poll]"
        or (
            isinstance(provenance, dict)
            and str(provenance.get("sourceTool") or "").lower() == "heartbeat"
        )
    )


def _session_role(session_key: str, entry: dict, agent_id: str) -> str:
    """Classify indexed sessions without assuming the parent uses ``:main``."""
    if entry.get("spawnedBy"):
        return "child"
    if session_key == f"agent:{agent_id}:main" or entry:
        return "parent"
    return "child"


def _session_file_name(entry: dict) -> str:
    raw = str(entry.get("sessionFile") or "").strip()
    if raw:
        return PurePosixPath(raw).name
    session_id = str(entry.get("sessionId") or "").strip()
    return f"{session_id}.jsonl" if session_id else ""


def _session_index(sessions_dir: Path) -> dict:
    try:
        value = json.loads((sessions_dir / "sessions.json").read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _session_sources(sessions_dir: Path) -> list[tuple[Path, str, dict]]:
    """Resolve durable session transcripts without reading replay checkpoints."""
    index = _session_index(sessions_dir)
    sources: list[tuple[Path, str, dict]] = []
    seen: set[Path] = set()
    for session_key, entry in index.items():
        if not isinstance(entry, dict):
            continue
        name = _session_file_name(entry)
        if not name:
            continue
        path = sessions_dir / name
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        sources.append((path, str(session_key), entry))
    if sources:
        return sources

    # Older archives may lack sessions.json. Checkpoints and generated
    # trajectories replay durable messages and must never count as sessions.
    for path in sorted(sessions_dir.glob("*.jsonl")):
        if ".checkpoint." in path.name or ".trajectory." in path.name:
            continue
        sources.append((path, "", {}))
    return sources


def _entry_time(entry: dict, fields: tuple[str, ...]) -> tuple[int, str]:
    for field in fields:
        timestamp_ns = _timestamp_ns(entry.get(field))
        if timestamp_ns:
            return timestamp_ns, field
    return 0, ""


def _transcript_time_bounds(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    first = 0
    last = 0
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return 0, 0
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = item.get("message") or {}
        timestamp_ns = _timestamp_ns(
            message.get("timestamp") or item.get("timestamp")
        )
        if not timestamp_ns:
            continue
        first = timestamp_ns if not first else min(first, timestamp_ns)
        last = max(last, timestamp_ns)
    return first, last


def _transcript_last_execution_time(path: Optional[Path]) -> tuple[int, str, str]:
    """Return the latest durable child execution event, excluding metadata.

    Session/index timestamps and replayed receipts prove that storage changed,
    not that the child executed. A child runtime lower bound therefore advances
    only on an assistant/model event or a persisted tool result from that
    child's own transcript.
    """
    if path is None or not path.is_file():
        return 0, "", "missing"
    latest_ns = 0
    latest_role = ""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return 0, "", "missing"
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        role = str(message.get("role") or "")
        if role not in {"assistant", "tool", "toolResult"}:
            continue
        timestamp_ns = _timestamp_ns(
            message.get("timestamp") or event.get("timestamp")
        )
        if timestamp_ns >= latest_ns:
            latest_ns = timestamp_ns
            latest_role = role
    if not latest_ns:
        return 0, "", "missing"
    return (
        latest_ns,
        f"transcript_last_{latest_role}_execution",
        "durable_transcript_execution",
    )


def _parent_start_time(
    entry: dict, transcript_first_ns: int
) -> tuple[int, str]:
    """Prefer immutable creation/transcript evidence over mutable startedAt."""
    created_ns, created_field = _entry_time(
        entry, ("createdAt", "created_at")
    )
    candidates = [
        (created_ns, created_field),
        (
            transcript_first_ns,
            "transcript_first_event" if transcript_first_ns else "",
        ),
    ]
    candidates = [(value, source) for value, source in candidates if value]
    if candidates:
        return min(candidates, key=lambda item: item[0])
    return _entry_time(
        entry,
        ("startedAt", "started_at", "sessionStartedAt", "session_started_at"),
    )


def _transcript_explicit_done_time(path: Path) -> int:
    """Return the timestamp when the final durable assistant turn says DONE."""
    if not path.is_file():
        return 0
    done_ns = 0
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = item.get("message") or {}
        if item.get("type") != "message" or message.get("role") != "assistant":
            continue
        timestamp_ns = _timestamp_ns(
            message.get("timestamp") or item.get("timestamp")
        )
        is_done = (
            str(message.get("stopReason") or "").lower() in {"stop", "length"}
            and bool(re.search(
                r"(?im)^\s*DONE(?:\s*$|\b)", _message_text(message)
            ))
        )
        done_ns = timestamp_ns if is_done else 0
    return done_ns


def _child_start_time(
    entry: dict, transcript_first_ns: int
) -> tuple[int, str, str, dict]:
    """Resolve one child's lifecycle start without trusting resumed startedAt.

    ``startedAt`` is mutable in OpenClaw v9 and can move forward when the same
    child is continued. Creation and the first event in that child's own
    transcript are durable lifecycle evidence. A transcript that predates the
    child's creation by more than a small serialization tolerance is treated as
    inherited/replayed history rather than blindly selected as the minimum.
    """
    created_ns, created_field = _entry_time(
        entry, ("createdAt", "created_at")
    )
    started_ns, started_field = _entry_time(
        entry, ("startedAt", "started_at", "sessionStartedAt", "session_started_at")
    )
    evidence = {
        key: value
        for key, value in (
            (created_field, created_ns),
            (started_field, started_ns),
            ("transcript_first_event", transcript_first_ns),
        )
        if key and value
    }
    durable_starts = [
        value for value in (created_ns, transcript_first_ns) if value
    ]
    conflict = bool(
        started_ns
        and durable_starts
        and started_ns > min(durable_starts) + 5_000_000_000
    )

    if created_ns:
        # A much older transcript is likely inherited history. A same-session
        # header/message close to creation is useful corroborating evidence,
        # but creation remains the stable lifecycle boundary.
        transcript_inherited = bool(
            transcript_first_ns
            and transcript_first_ns + 5_000_000_000 < created_ns
        )
        return created_ns, created_field, "durable_session_creation", {
            "evidence": evidence,
            "conflict": conflict or transcript_inherited,
            "transcript_inherited_or_replayed": transcript_inherited,
        }
    if transcript_first_ns:
        return (
            transcript_first_ns,
            "transcript_first_event",
            "durable_transcript_creation",
            {
                "evidence": evidence,
                "conflict": conflict,
                "transcript_inherited_or_replayed": False,
            },
        )
    if started_ns:
        return started_ns, started_field, "mutable_started_at_fallback", {
            "evidence": evidence,
            "conflict": False,
            "transcript_inherited_or_replayed": False,
        }
    return 0, "", "missing", {
        "evidence": evidence,
        "conflict": False,
        "transcript_inherited_or_replayed": False,
    }


def _transcript_terminal_time(path: Optional[Path]) -> tuple[int, str, str]:
    """Return a final terminal assistant event from one durable transcript.

    A stop/length/error in the middle of a transcript is not terminal evidence.
    This helper only accepts the last durable message and is used together with
    a terminal session state; unknown sessions require stronger runtime evidence.
    """
    if path is None or not path.is_file():
        return 0, "", "missing"
    latest: Optional[dict] = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return 0, "", "missing"
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "message":
            latest = event
    if latest is None:
        return 0, "", "missing"
    message = latest.get("message") or {}
    if message.get("role") != "assistant":
        return 0, "", "missing"
    stop_reason = str(message.get("stopReason") or "").strip().lower()
    if stop_reason not in {"stop", "length", "error"}:
        return 0, "", "missing"
    timestamp_ns = _timestamp_ns(
        message.get("timestamp") or latest.get("timestamp")
    )
    if not timestamp_ns:
        return 0, "", "missing"
    return timestamp_ns, "transcript_final_assistant", (
        f"transcript_terminal_{stop_reason}"
    )


def _child_terminal_time(
    entry: dict, transcript: Optional[Path]
) -> tuple[int, str, str]:
    timestamp_ns, field = _entry_time(
        entry,
        (
            "endedAt", "ended_at", "completedAt", "completed_at",
            "finishedAt", "finished_at",
        ),
    )
    if timestamp_ns:
        return timestamp_ns, field, "session_state"
    return _transcript_terminal_time(transcript)


def _gateway_child_failures(run_dir: Path) -> dict[str, dict]:
    """Map children to definitive failed embedded-run terminal evidence."""
    trace = run_dir / "conversation" / "cache-trace.jsonl"
    gateway = run_dir / "openclaw_gateway.log"
    if not trace.is_file() or not gateway.is_file():
        return {}

    run_identity: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for line in trace.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = str(event.get("runId") or "").strip()
        session_id = str(event.get("sessionId") or "").strip()
        session_key = str(event.get("sessionKey") or "").strip()
        if not run_id or not session_id or not session_key:
            continue
        identity = (session_id, session_key)
        if run_id in run_identity and run_identity[run_id] != identity:
            ambiguous.add(run_id)
        else:
            run_identity[run_id] = identity
    for run_id in ambiguous:
        run_identity.pop(run_id, None)

    pattern = re.compile(
        r"^(\S+)\s+\[agent/embedded\]\s+embedded run agent end:\s+"
        r"runId=(\S+)\s+isError=true\b(?P<rest>.*)$"
    )
    failures: dict[str, dict] = {}
    for line in gateway.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        identity = run_identity.get(match.group(2))
        timestamp_ns = _timestamp_ns(match.group(1))
        if identity is None or not timestamp_ns:
            continue
        session_id, session_key = identity
        raw_error = re.search(r"\brawError=(.*)$", match.group("rest"))
        evidence = {
            "session_id": session_id,
            "session_key": session_key,
            "run_id": match.group(2),
            "timestamp_ns": timestamp_ns,
            "error": raw_error.group(1).strip() if raw_error else None,
            "source": "gateway_embedded_run_error",
        }
        failures[session_key] = evidence
        failures[session_id] = evidence
    return failures


def _child_state_time(
    entry: dict, transcript_last_ns: int
) -> tuple[int, str, str]:
    timestamp_ns, field = _entry_time(
        entry, ("updatedAt", "updated_at", "lastActivityAt", "last_activity_at")
    )
    if transcript_last_ns > timestamp_ns:
        return transcript_last_ns, "transcript_last_event", "transcript_inferred"
    if timestamp_ns:
        return timestamp_ns, field, "session_state"
    return 0, "", "missing"


def _lifecycle_observation_boundary(
    run_dir: Path, fallback_ns: int
) -> tuple[int, str, str]:
    """Return the fence after which agent sessions cannot keep running."""
    path = run_dir / "openclaw_response.json"
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        payload = {}

    lifecycle = None
    pending = [payload]
    while pending:
        value = pending.pop()
        if not isinstance(value, dict):
            continue
        candidate = value.get("agentBenchLifecycle")
        if isinstance(candidate, dict):
            lifecycle = candidate
            break
        pending.extend(item for item in value.values() if isinstance(item, dict))

    if lifecycle:
        for field in ("workspaceQuiescedNs", "parentTerminalObservedNs"):
            timestamp_ns = _timestamp_ns(lifecycle.get(field))
            if timestamp_ns:
                quality = (
                    "harness_observed_censor"
                    if field == "workspaceQuiescedNs"
                    else "watcher_observed_censor"
                )
                return timestamp_ns, field, quality
    return fallback_ns, "measurement_observed", "measurement_observed_censor"


def _append(
    events: list[dict],
    *,
    timestamp_ns: int,
    session_id: str,
    session_key: str,
    role: str,
    event_type: str,
    **fields: Any,
) -> None:
    event = {
        "timestamp": _iso_from_ns(timestamp_ns),
        "timestamp_ns": timestamp_ns or None,
        "session_id": session_id or None,
        "session_key": session_key or None,
        "role": role,
        "event_type": event_type,
    }
    event.update(fields)
    events.append(event)


def _session_events(run_dir: Path, agent_id: str) -> list[dict]:
    sessions_dir = run_dir / "conversation" / "sessions"
    if not sessions_dir.is_dir():
        return []
    index = _session_index(sessions_dir)
    events: list[dict] = []
    session_bounds: dict[str, tuple[int, int]] = {}

    for path, session_key, entry in _session_sources(sessions_dir):
        session_id = str(entry.get("sessionId") or path.stem)
        role = _session_role(session_key, entry, agent_id)
        session_state = str(entry.get("status") or "unknown")
        seen_assistant: set[str] = set()
        seen_tool_results: set[str] = set()
        seen_resumes: set[tuple[int, str]] = set()
        heartbeat_turn = False
        transcript_first_ns = 0
        transcript_last_ns = 0
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = item.get("message") or {}
            timestamp_ns = _timestamp_ns(
                message.get("timestamp") or item.get("timestamp")
            )
            if timestamp_ns:
                transcript_first_ns = (
                    timestamp_ns if not transcript_first_ns
                    else min(transcript_first_ns, timestamp_ns)
                )
                transcript_last_ns = max(transcript_last_ns, timestamp_ns)
            item_type = item.get("type")
            custom_type = str(item.get("customType") or "")
            if item_type == "custom_message" and custom_type == "openclaw.sessions_yield":
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role, event_type="yield",
                    session_state=session_state,
                )
                continue
            if item_type == "custom" and custom_type == "openclaw:prompt-error":
                data = item.get("data")
                error = data.get("error") if isinstance(data, dict) else data
                error = str(error or "OpenClaw prompt error")
                event_type = (
                    "context_overflow"
                    if "context overflow" in error.lower().replace("_", " ")
                    else "prompt_error"
                )
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role, event_type=event_type,
                    error=error, session_state=session_state,
                )
                continue
            if item_type != "message":
                continue

            message_role = str(message.get("role") or "")
            if message_role == "user":
                text = _message_text(message)
                if role == "parent" and _is_heartbeat_request(text, message):
                    heartbeat_turn = True
                    _append(
                        events, timestamp_ns=timestamp_ns,
                        session_id=session_id, session_key=session_key,
                        role=role, event_type="heartbeat_request",
                        text=text, session_state=session_state,
                    )
                    continue

                if role == "parent":
                    heartbeat_turn = False

                resume_key = (timestamp_ns, text)
                if resume_key in seen_resumes:
                    continue
                seen_resumes.add(resume_key)
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role, event_type="resume",
                    text=text, session_state=session_state,
                )
                continue
            if message_role == "toolResult":
                tool_result_id = str(message.get("toolCallId") or "")
                if tool_result_id and tool_result_id in seen_tool_results:
                    continue
                if tool_result_id:
                    seen_tool_results.add(tool_result_id)
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role, event_type="tool_result",
                    tool_name=message.get("toolName"),
                    tool_call_id=message.get("toolCallId"),
                    is_error=bool(message.get("isError")),
                    duration_ms=(message.get("details") or {}).get("durationMs"),
                    session_state=session_state,
                )
                continue
            if message_role != "assistant":
                continue

            response_id = str(message.get("responseId") or "")
            if response_id and response_id in seen_assistant:
                continue
            if response_id:
                seen_assistant.add(response_id)

            text = _message_text(message)
            for content in message.get("content") or []:
                if not isinstance(content, dict) or content.get("type") != "toolCall":
                    continue
                tool_name = str(content.get("name") or "unknown")
                event_type = "spawn" if tool_name == "sessions_spawn" else "tool_call"
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role, event_type=event_type,
                    tool_name=tool_name, tool_call_id=content.get("id"),
                    arguments=content.get("arguments"), session_state=session_state,
                )
            stop_reason = str(message.get("stopReason") or "").lower()

            if role == "parent" and heartbeat_turn:
                _append(
                    events, timestamp_ns=timestamp_ns,
                    session_id=session_id, session_key=session_key,
                    role=role, event_type="heartbeat_response",
                    text=text, stop_reason=stop_reason,
                    session_state=session_state,
                )
                if stop_reason == "stop":
                    heartbeat_turn = False
                continue

            if stop_reason in {"stop", "length"}:
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role,
                    event_type="assistant_stop", text=text,
                    stop_reason=stop_reason, session_state=session_state,
                )
                strict_done = bool(
                    re.search(r"(?im)^\s*DONE(?:\s*$|\b)", text)
                )
                markdown_done = bool(
                    re.search(
                        r"(?im)^\s*(?:\*\*|__)+DONE(?:\*\*|__)+(?:\s*$|\b)",
                        text,
                    )
                )
                if strict_done:
                    _append(
                        events, timestamp_ns=timestamp_ns, session_id=session_id,
                        session_key=session_key, role=role, event_type="done",
                        text=text, session_state=session_state,
                        marker_style="strict_bare_done",
                    )
                elif markdown_done:
                    _append(
                        events, timestamp_ns=timestamp_ns, session_id=session_id,
                        session_key=session_key, role=role,
                        event_type="done_declaration", text=text,
                        session_state=session_state,
                        marker_style="markdown_done_declaration",
                    )

        session_bounds[session_key] = (
            transcript_first_ns, transcript_last_ns
        )

    for session_key, entry in index.items():
        if not isinstance(entry, dict) or not entry.get("spawnedBy"):
            continue
        status = str(entry.get("status") or "unknown").lower()
        event_type = {
            "done": "child_completion",
            "completed": "child_completion",
            "timeout": "child_timeout",
            "timed_out": "child_timeout",
        }.get(status, "child_state")
        _first_ns, last_ns = session_bounds.get(str(session_key), (0, 0))
        name = _session_file_name(entry)
        transcript = sessions_dir / name if name else None
        if status in {
            "done", "completed", "failed", "error", "timeout", "timed_out",
            "cancelled", "canceled",
        }:
            source_ns, source_field, timestamp_quality = _child_terminal_time(
                entry, transcript
            )
        else:
            source_ns, source_field, timestamp_quality = _child_state_time(
                entry, last_ns
            )
        measurement_observed_ns = time.time_ns()
        _append(
            events, timestamp_ns=source_ns,
            session_id=str(entry.get("sessionId") or ""),
            session_key=str(session_key), role="child", event_type=event_type,
            session_state=status, spawned_by=entry.get("spawnedBy"),
            source_timestamp=_iso_from_ns(source_ns),
            source_timestamp_ns=source_ns or None,
            source_timestamp_field=source_field or None,
            measurement_observed_timestamp=_iso_from_ns(measurement_observed_ns),
            measurement_observed_timestamp_ns=measurement_observed_ns,
            timestamp_quality=timestamp_quality,
        )
    return events


def _child_runtime_observability(run_dir: Path, agent_id: str) -> dict:
    sessions_dir = run_dir / "conversation" / "sessions"
    index = _session_index(sessions_dir) if sessions_dir.is_dir() else {}
    children: dict[str, dict] = {}
    intervals: list[tuple[int, int]] = []
    lower_bound_intervals: list[tuple[int, int]] = []
    terminal_states = {
        "done", "completed", "failed", "error", "timeout", "timed_out",
        "cancelled", "canceled",
    }
    measurement_observed_ns = time.time_ns()
    observation_end_ns, observation_end_source, observation_end_quality = (
        _lifecycle_observation_boundary(run_dir, measurement_observed_ns)
    )
    child_failures = _gateway_child_failures(run_dir)

    for session_key, entry in index.items():
        if not isinstance(entry, dict) or not entry.get("spawnedBy"):
            continue
        session_key = str(session_key)
        name = _session_file_name(entry)
        transcript = sessions_dir / name if name else None
        first_ns, last_ns = (
            _transcript_time_bounds(transcript)
            if transcript is not None else (0, 0)
        )
        (
            start_ns,
            start_source,
            start_quality,
            start_diagnostic,
        ) = _child_start_time(entry, first_ns)
        observed_status = str(entry.get("status") or "unknown").lower()
        measured_status = observed_status
        is_terminal = observed_status in terminal_states
        terminal_evidence = None
        execution_last_ns, execution_last_source, execution_last_quality = (
            _transcript_last_execution_time(transcript)
        )
        if is_terminal:
            end_ns, end_source, end_quality = _child_terminal_time(
                entry, transcript
            )
            terminal_evidence = (
                end_source if end_ns else None
            )
        else:
            failure = child_failures.get(session_key) or child_failures.get(
                str(entry.get("sessionId") or "")
            )
            current_run_id = str(entry.get("lastRunId") or "").strip()
            if (
                isinstance(failure, dict)
                and failure.get("session_key") == session_key
                and failure.get("session_id")
                == str(entry.get("sessionId") or "")
                and (
                    not current_run_id
                    or failure.get("run_id") == current_run_id
                )
                and _int(failure.get("timestamp_ns")) >= execution_last_ns
            ):
                measured_status = "failed"
                is_terminal = True
                end_ns = _int(failure.get("timestamp_ns"))
                end_source = str(failure.get("source") or "")
                end_quality = "matched_runtime_terminal_evidence"
                terminal_evidence = {
                    **failure,
                    "identity_quality": (
                        "session_and_current_run_match"
                        if current_run_id
                        else "session_match_run_unavailable"
                    ),
                }
            else:
                end_ns = 0
                end_source = ""
                end_quality = "missing"

        last_observed_ns = execution_last_ns
        last_observed_source = execution_last_source
        last_observed_quality = execution_last_quality

        runtime_seconds = None
        if start_ns and end_ns and end_ns >= start_ns:
            runtime_seconds = (end_ns - start_ns) / 1e9
            intervals.append((start_ns, end_ns))
            lower_bound_intervals.append((start_ns, end_ns))
        runtime_lower_bound_seconds = None
        if (
            runtime_seconds is None
            and start_ns
            and last_observed_ns
            and last_observed_ns >= start_ns
        ):
            runtime_lower_bound_seconds = (
                last_observed_ns - start_ns
            ) / 1e9
            lower_bound_intervals.append((start_ns, last_observed_ns))

        children[session_key] = {
            "session_id": str(entry.get("sessionId") or "") or None,
            "spawned_by": entry.get("spawnedBy"),
            "status": measured_status,
            "observed_status": observed_status,
            "status_source": (
                "matched_runtime_terminal_evidence"
                if measured_status != observed_status else "session_index"
            ),
            "start_timestamp": _iso_from_ns(start_ns),
            "start_timestamp_ns": start_ns or None,
            "start_timestamp_source": start_source or None,
            "start_timestamp_quality": start_quality,
            "start_timestamp_evidence": start_diagnostic["evidence"],
            "start_timestamp_conflict": start_diagnostic["conflict"],
            "start_transcript_inherited_or_replayed": start_diagnostic[
                "transcript_inherited_or_replayed"
            ],
            "terminal_timestamp": _iso_from_ns(end_ns) if is_terminal else None,
            "terminal_timestamp_ns": (end_ns or None) if is_terminal else None,
            "terminal_timestamp_source": (
                (end_source or None) if is_terminal else None
            ),
            "terminal_timestamp_quality": (
                end_quality if is_terminal else "missing"
            ),
            "terminal_evidence": terminal_evidence,
            "interval_end_timestamp": (
                _iso_from_ns(end_ns) if runtime_seconds is not None else None
            ),
            "interval_end_timestamp_ns": (
                end_ns if runtime_seconds is not None else None
            ),
            "interval_end_timestamp_source": (
                (end_source or None) if runtime_seconds is not None else None
            ),
            "interval_end_timestamp_quality": (
                end_quality if runtime_seconds is not None else "missing"
            ),
            "last_observed_timestamp": _iso_from_ns(last_observed_ns),
            "last_observed_timestamp_ns": last_observed_ns or None,
            "last_observed_timestamp_source": last_observed_source or None,
            "last_observed_timestamp_quality": last_observed_quality,
            "measurement_observed_timestamp": _iso_from_ns(measurement_observed_ns),
            "measurement_observed_timestamp_ns": measurement_observed_ns,
            "runtime_seconds": runtime_seconds,
            "runtime_lower_bound_seconds": runtime_lower_bound_seconds,
            "runtime_complete": is_terminal and runtime_seconds is not None,
            "runtime_censored": runtime_seconds is None,
            "label": entry.get("label"),
        }

    complete = [
        child for child in children.values()
        if child.get("runtime_complete")
    ]
    observed = [
        child for child in children.values()
        if child.get("runtime_seconds") is not None
    ]
    runtime_sum = sum(float(child["runtime_seconds"]) for child in observed)
    runtime_lower_bound_sum = sum(
        float(child.get("runtime_seconds") or 0.0)
        if child.get("runtime_seconds") is not None
        else float(child.get("runtime_lower_bound_seconds") or 0.0)
        for child in children.values()
    )

    union_ns = 0
    if intervals:
        merged: list[list[int]] = []
        for start_ns, end_ns in sorted(intervals):
            if not merged or start_ns > merged[-1][1]:
                merged.append([start_ns, end_ns])
            else:
                merged[-1][1] = max(merged[-1][1], end_ns)
        union_ns = sum(end - start for start, end in merged)

    lower_bound_union_ns = 0
    if lower_bound_intervals:
        merged_lower: list[list[int]] = []
        for start_ns, end_ns in sorted(lower_bound_intervals):
            if not merged_lower or start_ns > merged_lower[-1][1]:
                merged_lower.append([start_ns, end_ns])
            else:
                merged_lower[-1][1] = max(merged_lower[-1][1], end_ns)
        lower_bound_union_ns = sum(end - start for start, end in merged_lower)

    max_concurrent = 0
    concurrent = 0
    overlap_wall_ns = 0
    concurrency_excess_ns = 0
    previous_timestamp_ns: Optional[int] = None
    points: list[tuple[int, int]] = []
    for start_ns, end_ns in intervals:
        points.append((start_ns, 1))
        points.append((end_ns, -1))
    for timestamp_ns_value, delta in sorted(
        points, key=lambda item: (item[0], item[1])
    ):
        if previous_timestamp_ns is not None and timestamp_ns_value > previous_timestamp_ns:
            span_ns = timestamp_ns_value - previous_timestamp_ns
            if concurrent >= 2:
                overlap_wall_ns += span_ns
            if concurrent > 1:
                concurrency_excess_ns += (concurrent - 1) * span_ns
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
        previous_timestamp_ns = timestamp_ns_value

    lower_bound_max_concurrent = 0
    lower_bound_concurrent = 0
    lower_bound_points = []
    for start_ns, end_ns in lower_bound_intervals:
        lower_bound_points.append((start_ns, 1))
        lower_bound_points.append((end_ns, -1))
    for _timestamp_ns_value, delta in sorted(
        lower_bound_points, key=lambda item: (item[0], item[1])
    ):
        lower_bound_concurrent += delta
        lower_bound_max_concurrent = max(
            lower_bound_max_concurrent, lower_bound_concurrent
        )

    timeline_agents = []
    parent_candidates = [
        (key, entry) for key, entry in index.items()
        if isinstance(entry, dict) and not entry.get("spawnedBy")
    ]
    if parent_candidates:
        parent_key, parent_entry = parent_candidates[0]
        parent_name = _session_file_name(parent_entry)
        parent_transcript = sessions_dir / parent_name if parent_name else None
        parent_first_ns, parent_last_ns = (
            _transcript_time_bounds(parent_transcript)
            if parent_transcript is not None else (0, 0)
        )
        parent_start_ns, parent_start_source = _parent_start_time(
            parent_entry, parent_first_ns
        )
        parent_status = str(parent_entry.get("status") or "unknown").lower()
        explicit_done_ns = (
            _transcript_explicit_done_time(parent_transcript)
            if parent_transcript is not None else 0
        )
        parent_terminal = parent_status in terminal_states or bool(explicit_done_ns)
        if parent_status in terminal_states:
            parent_end_ns, parent_end_source, parent_end_quality = (
                _child_terminal_time(parent_entry, parent_transcript)
            )
        elif explicit_done_ns:
            parent_end_ns = explicit_done_ns
            parent_end_source = "transcript_explicit_done"
            parent_end_quality = "transcript_terminal_evidence"
        else:
            parent_end_ns = observation_end_ns
            parent_end_source = observation_end_source
            parent_end_quality = observation_end_quality
        if parent_start_ns and parent_end_ns >= parent_start_ns:
            timeline_agents.append({
                "role": "parent",
                "label": agent_id,
                "session_key": str(parent_key),
                "session_id": str(parent_entry.get("sessionId") or "") or None,
                "status": parent_status,
                "start_timestamp": _iso_from_ns(parent_start_ns),
                "start_timestamp_ns": parent_start_ns,
                "start_timestamp_source": parent_start_source or None,
                "end_timestamp": _iso_from_ns(parent_end_ns),
                "end_timestamp_ns": parent_end_ns,
                "end_timestamp_source": parent_end_source or None,
                "end_timestamp_quality": parent_end_quality,
                "duration_seconds": (parent_end_ns - parent_start_ns) / 1e9,
                "runtime_censored": not parent_terminal,
            })
    for session_key, child in children.items():
        start_ns = _int(child.get("start_timestamp_ns"))
        end_ns = _int(child.get("interval_end_timestamp_ns"))
        lower_bound_end_ns = _int(child.get("last_observed_timestamp_ns"))
        display_end_ns = end_ns or lower_bound_end_ns
        if not start_ns or not display_end_ns or display_end_ns < start_ns:
            continue
        timeline_agents.append({
            "role": "child",
            "label": child.get("label") or child.get("session_id") or session_key,
            "session_key": session_key,
            "session_id": child.get("session_id"),
            "status": child.get("status"),
            "start_timestamp": child.get("start_timestamp"),
            "start_timestamp_ns": start_ns,
            "start_timestamp_source": child.get("start_timestamp_source"),
            "end_timestamp": (
                child.get("interval_end_timestamp")
                or child.get("last_observed_timestamp")
            ),
            "end_timestamp_ns": display_end_ns,
            "end_timestamp_source": (
                child.get("interval_end_timestamp_source")
                or child.get("last_observed_timestamp_source")
            ),
            "end_timestamp_quality": (
                child.get("interval_end_timestamp_quality")
                if end_ns else "observed_lower_bound"
            ),
            "duration_seconds": child.get("runtime_seconds"),
            "runtime_lower_bound_seconds": child.get(
                "runtime_lower_bound_seconds"
            ),
            "runtime_censored": child.get("runtime_censored", False),
        })
    timeline_agents.sort(
        key=lambda item: (
            0 if item["role"] == "parent" else 1,
            item["start_timestamp_ns"],
            str(item["session_key"]),
        )
    )

    union_seconds = union_ns / 1e9
    unknown_count = sum(
        not child.get("runtime_complete") for child in children.values()
    )
    lifecycle = None
    response_path = run_dir / "openclaw_response.json"
    try:
        response = json.loads(response_path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        response = {}
    pending = [response]
    while pending:
        value = pending.pop()
        if not isinstance(value, dict):
            continue
        candidate = value.get("agentBenchLifecycle")
        if isinstance(candidate, dict):
            lifecycle = candidate
            break
        pending.extend(item for item in value.values() if isinstance(item, dict))

    state_counts: dict[str, int] = defaultdict(int)
    for child in children.values():
        state_counts[str(child.get("status") or "unknown")] += 1
    start_quality_counts: dict[str, int] = defaultdict(int)
    terminal_quality_counts: dict[str, int] = defaultdict(int)
    start_conflict_count = 0
    qualified_terminal_identity_count = 0
    for child in children.values():
        start_quality_counts[
            str(child.get("start_timestamp_quality") or "missing")
        ] += 1
        terminal_quality_counts[
            str(child.get("terminal_timestamp_quality") or "missing")
        ] += 1
        start_conflict_count += bool(child.get("start_timestamp_conflict"))
        evidence = child.get("terminal_evidence")
        if (
            isinstance(evidence, dict)
            and evidence.get("identity_quality")
            == "session_match_run_unavailable"
        ):
            qualified_terminal_identity_count += 1
    has_qualified_timestamps = bool(
        start_conflict_count
        or start_quality_counts.get("mutable_started_at_fallback", 0)
        or qualified_terminal_identity_count
    )
    projection_sources = {
        str(entry.get("agentBenchProjectionSource") or "")
        for entry in index.values()
        if isinstance(entry, dict) and entry.get("spawnedBy")
    }
    projection_source_known = bool(projection_sources) and projection_sources <= {
        "online_index",
        "restored_archived_transcript",
    }
    online_index_child_count = None
    restored_archived_child_count = None
    if not children:
        online_index_child_count = 0
        restored_archived_child_count = 0
        child_count_source_quality = "exact_no_materialized_children"
    elif projection_source_known:
        online_index_child_count = sum(
            entry.get("agentBenchProjectionSource") == "online_index"
            for entry in index.values()
            if isinstance(entry, dict) and entry.get("spawnedBy")
        )
        restored_archived_child_count = sum(
            entry.get("agentBenchProjectionSource")
            == "restored_archived_transcript"
            for entry in index.values()
            if isinstance(entry, dict) and entry.get("spawnedBy")
        )
        child_count_source_quality = "projection_sources_exact"
    else:
        child_count_source_quality = (
            "legacy_projection_online_split_unavailable"
        )
    return {
        "child_session_count": len(children),
        "materialized_child_session_count": len(children),
        "online_index_child_count_at_archive": online_index_child_count,
        "restored_archived_child_count": restored_archived_child_count,
        "child_count_source_quality": child_count_source_quality,
        "child_runtime_complete_count": len(complete),
        "child_runtime_unknown_count": unknown_count,
        "child_runtime_observed_count": len(observed),
        "child_runtime_by_session": children,
        "child_sum_runtime_seconds": runtime_sum,
        "child_sum_runtime_lower_bound_seconds": runtime_lower_bound_sum,
        "child_union_active_seconds": union_seconds,
        "child_union_active_lower_bound_seconds": lower_bound_union_ns / 1e9,
        "child_overlap_wall_seconds": overlap_wall_ns / 1e9,
        "child_concurrency_excess_seconds": concurrency_excess_ns / 1e9,
        "max_concurrent_children": max_concurrent,
        "max_concurrent_children_lower_bound": lower_bound_max_concurrent,
        "child_derived_metrics_quality": (
            "partial_excludes_unknown"
            if unknown_count
            else (
                "complete_with_qualified_timestamps"
                if has_qualified_timestamps
                else "complete"
            )
        ),
        "child_start_timestamp_quality_counts": dict(start_quality_counts),
        "child_terminal_timestamp_quality_counts": dict(
            terminal_quality_counts
        ),
        "child_start_timestamp_conflict_count": start_conflict_count,
        "child_terminal_identity_qualified_count": (
            qualified_terminal_identity_count
        ),
        "reconstructed_child_state_counts": dict(state_counts),
        "parent_terminal_child_snapshot": {
            "total_children": (
                lifecycle.get("totalChildren") if lifecycle else None
            ),
            "completed_children": (
                lifecycle.get("completedChildren") if lifecycle else None
            ),
            "pending_children": (
                lifecycle.get("pendingChildrenAtParentTerminal")
                if lifecycle else None
            ),
            "child_states": (
                lifecycle.get("childStatesAtParentTerminal")
                if lifecycle else None
            ),
            "source": "agentBenchLifecycle" if lifecycle else None,
        },
        "agent_timeline": {
            "observation_end_timestamp": _iso_from_ns(observation_end_ns),
            "observation_end_timestamp_ns": observation_end_ns,
            "observation_end_timestamp_source": observation_end_source,
            "observation_end_timestamp_quality": observation_end_quality,
            "agents": timeline_agents,
        },
        "child_runtime_policy": {
            "definition": (
                "durable child lifecycle start to matched terminal evidence; "
                "this is elapsed lifecycle time, not GPU-active time"
            ),
            "censored_runtime": (
                "unknown terminal time is reported only as an observed lower "
                "bound and excluded from precise runtime/concurrency aggregates"
            ),
            "overlap_wall_definition": (
                "wall-clock time with at least two child sessions active"
            ),
            "concurrency_excess_definition": (
                "integral of max(active_children - 1, 0) over child-active time"
            ),
            "failed_spawn_runtime": None,
        },
    }


def _gateway_events(run_dir: Path) -> list[dict]:
    path = run_dir / "openclaw_gateway.log"
    if not path.is_file():
        return []
    events = []
    line_re = re.compile(r"^(\S+)\s+(.*)$")
    for line in path.read_text(errors="replace").splitlines():
        match = line_re.match(line)
        if not match:
            continue
        timestamp_ns = _timestamp_ns(match.group(1))
        text = match.group(2)
        lowered = text.lower()
        event_type = None
        fields: dict[str, Any] = {"message": text}
        run_id_match = re.search(r"runId=(announce:\S+)", text, re.IGNORECASE)
        announce_run_id = run_id_match.group(1) if run_id_match else ""
        child_match = re.search(r"childSessionKey=(\S+)", text)
        child_session_key = child_match.group(1) if child_match else ""
        if announce_run_id:
            fields["announce_run_id"] = announce_run_id
            prefix = "announce:v1:"
            remainder = (
                announce_run_id[len(prefix):]
                if announce_run_id.startswith(prefix)
                else announce_run_id
            )
            if ":" in remainder:
                child_session_key = remainder.rsplit(":", 1)[0]
        if child_session_key:
            fields["child_session_key"] = child_session_key
        requester_settle_timeout = (
            "requester settle wake" in lowered
            and "gateway request timeout for agent" in lowered
        )
        if (
            "subagent announce completion" in lowered and "timeout" in lowered
        ) or requester_settle_timeout:
            event_type = "announce_timeout"
            timeout = re.search(r"timeout after (\d+)ms", lowered)
            fields["latency_ms"] = _int(timeout.group(1)) if timeout else None
            retry = re.search(r"retrying (\d+)/(\d+)", lowered)
            settle_attempt = re.search(
                r"requester settle wake attempt (\d+) failed", lowered
            )
            fields["retry_attempt"] = (
                _int(retry.group(1))
                if retry
                else _int(settle_attempt.group(1)) if settle_attempt else None
            )
            fields["max_attempts"] = _int(retry.group(2)) if retry else None
            retry_delay = re.search(r"retrying in (\d+)s", lowered)
            fields["retry_delay_seconds"] = (
                _int(retry_delay.group(1)) if retry_delay else None
            )
            if requester_settle_timeout:
                fields["announce_kind"] = "requester_settle_wake"
        elif "runid=announce:" in lowered and "res ✓ agent" in lowered:
            event_type = "announce"
            latency = re.search(r"agent\s+(\d+)ms", lowered)
            fields["latency_ms"] = _int(latency.group(1)) if latency else None
        elif "stuck session:" in lowered:
            event_type = "stuck_session"
            session = re.search(r"sessionKey=(\S+)", text)
            age = re.search(r"age=(\d+)s", text)
            fields["session_key"] = session.group(1) if session else None
            fields["age_seconds"] = _int(age.group(1)) if age else None
        elif "context-overflow-diag" in lowered:
            event_type = "context_overflow"
        elif "tool loop:" in lowered:
            event_type = "loop_detected"
            session_id = re.search(r"\bsessionId=([^\s]+)", text)
            session_key = re.search(r"\bsessionKey=([^\s]+)", text)
            detector = re.search(r"\bdetector=([^\s]+)", text)
            tool = re.search(r"\btool=([^\s]+)", text)
            count = re.search(r"\bcount=(\d+)", text)
            level = re.search(r"\blevel=([^\s]+)", text)
            action = re.search(r"\baction=([^\s]+)", text)
            parsed_action = action.group(1).lower() if action else None
            fields.update({
                "detector": detector.group(1) if detector else "tool_loop",
                "tool_name": tool.group(1) if tool else None,
                "count": _int(count.group(1)) if count else None,
                "level": level.group(1).lower() if level else None,
                "action": parsed_action,
                "terminal": parsed_action in {
                    "abort", "block", "error", "stop", "terminate",
                },
            })
            if session_key:
                fields["session_key"] = session_key.group(1)
            if session_id:
                fields["session_id"] = session_id.group(1)
        elif "compaction_loop_persisted" in lowered:
            event_type = "loop_detected"
            fields["detector"] = "compaction_loop_persisted"
            fields["terminal"] = True
            tool = re.search(r"\btool=([^\s]+)", text)
            count = re.search(r"\bcount=(\d+)", text)
            fields["tool_name"] = tool.group(1) if tool else None
            fields["count"] = _int(count.group(1)) if count else None
        elif "compaction" in lowered:
            count_match = re.search(
                r"\b(?:compactions|compactionAttempts|compactionCount|"
                r"compaction_count)\s*[=:]\s*(\d+)",
                text,
                re.IGNORECASE,
            )
            compaction_count = _int(count_match.group(1)) if count_match else 0
            if compaction_count > 0:
                event_type = "compaction"
                fields["count"] = compaction_count
        if event_type:
            _append(
                events, timestamp_ns=timestamp_ns,
                session_id=str(fields.pop("session_id", "") or ""),
                session_key=str(fields.pop("session_key", "") or ""),
                role="runtime", event_type=event_type, **fields,
            )
            if event_type == "context_overflow" and "compaction" in lowered:
                attempts = re.search(r"compactionAttempts=(\d+)", text)
                attempt_count = _int(attempts.group(1)) if attempts else 0
                if attempt_count > 0:
                    _append(
                        events, timestamp_ns=timestamp_ns, session_id="",
                        session_key="", role="runtime", event_type="compaction",
                        attempts=attempt_count, count=attempt_count,
                        message=text,
                    )
    return events


def _read_spans(run_dir: Path) -> Iterable[dict]:
    path = run_dir / "spans.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _logical_tokens_from_transcripts(
    run_dir: Path, agent_id: str
) -> tuple[dict, list[dict]]:
    sessions_dir = run_dir / "conversation" / "sessions"
    sessions: dict[str, dict] = defaultdict(
        lambda: {"input": 0, "output": 0, "cache_read": 0, "turn_count": 0}
    )
    usage_events: list[dict] = []
    if not sessions_dir.is_dir():
        return {}, []

    for path, session_key, entry in _session_sources(sessions_dir):
        session_id = str(entry.get("sessionId") or path.stem)
        role = _session_role(session_key, entry, agent_id)
        seen_responses: set[str] = set()
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = item.get("message") or {}
            if item.get("type") != "message" or message.get("role") != "assistant":
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            response_id = str(message.get("responseId") or "")
            if response_id and response_id in seen_responses:
                continue
            if response_id:
                seen_responses.add(response_id)
            uncached_input = _int(usage.get("input") or usage.get("prompt_tokens"))
            cache_read = _int(usage.get("cacheRead") or usage.get("cache_read"))
            input_tokens = uncached_input + cache_read
            output_tokens = _int(
                usage.get("output") or usage.get("completion_tokens")
            )
            bucket_key = (
                f"agent:{agent_id}:main"
                if role == "parent"
                else session_key or f"session:{session_id}"
            )
            bucket = sessions[bucket_key]
            bucket["input"] += input_tokens
            bucket["output"] += output_tokens
            bucket["cache_read"] += cache_read
            bucket["turn_count"] += 1
            _append(
                usage_events,
                timestamp_ns=_timestamp_ns(
                    message.get("timestamp") or item.get("timestamp")
                ),
                session_id=session_id,
                session_key=session_key,
                role=role,
                event_type="model_usage",
                input_tokens=input_tokens,
                uncached_input_tokens=uncached_input,
                cache_read_tokens=cache_read,
                output_tokens=output_tokens,
                request_latency_ms=None,
            )
    return dict(sessions), usage_events


def _parent_resume_count(
    run_dir: Path,
    agent_id: str,
    *,
    parent_session_key: Optional[str] = None,
) -> int:
    """Count requester episodes after the initial parent request.

    OpenClaw v9 persists a stable event id, which distinguishes genuine
    byte-identical resumes while deduplicating replay. Older transcripts lack
    that id, so their text remains the best available fallback identity.
    """
    sessions_dir = run_dir / "conversation" / "sessions"
    if not sessions_dir.is_dir():
        return 0
    requester_episodes = 0
    seen: set[tuple[str, str]] = set()
    for path, session_key, entry in _session_sources(sessions_dir):
        if parent_session_key is not None:
            if session_key != parent_session_key:
                continue
        elif _session_role(session_key, entry, agent_id) != "parent":
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = item.get("message") or {}
            if item.get("type") != "message" or message.get("role") != "user":
                continue
            text = _message_text(message)
            if _is_heartbeat_request(text, message):
                continue
            event_id = str(item.get("id") or message.get("id") or "").strip()
            identity = ("id", event_id) if event_id else ("text", text)
            if identity in seen:
                continue
            seen.add(identity)
            requester_episodes += 1
    return max(0, requester_episodes - 1)


def _logical_tokens(run_dir: Path, agent_id: str) -> tuple[dict, list[dict]]:
    transcript_sessions, transcript_events = _logical_tokens_from_transcripts(
        run_dir, agent_id
    )
    if transcript_sessions:
        sessions = transcript_sessions
        usage_events = transcript_events
    else:
        sessions, usage_events = _logical_tokens_from_spans(run_dir, agent_id)

    parent_key = f"agent:{agent_id}:main"
    empty = {
        "input": 0, "output": 0, "cache_read": 0, "turn_count": 0
    }
    parent = dict(sessions.get(parent_key, empty))
    parent["resumed_turns"] = _parent_resume_count(run_dir, agent_id)
    children = {
        key: dict(value) for key, value in sessions.items() if key != parent_key
    }
    return {
        "parent": parent,
        "children": children,
        "total_logical_input": parent["input"] + sum(v["input"] for v in children.values()),
        "total_logical_output": parent["output"] + sum(v["output"] for v in children.values()),
    }, usage_events


def _logical_tokens_from_spans(
    run_dir: Path, agent_id: str
) -> tuple[dict, list[dict]]:
    sessions: dict[str, dict] = defaultdict(
        lambda: {"input": 0, "output": 0, "turn_count": 0}
    )
    usage_events = []
    recent: dict[tuple, int] = {}
    for span in _read_spans(run_dir):
        attrs = span.get("attrs") or {}
        if span.get("name") != "openclaw.model.usage":
            continue
        if attrs.get("openclaw.span.source") == "raw_response_log":
            continue
        session_key = str(attrs.get("openclaw.sessionKey") or "unknown")
        session_id = str(attrs.get("openclaw.sessionId") or "")
        input_tokens = _int(attrs.get("openclaw.tokens.input"))
        output_tokens = _int(attrs.get("openclaw.tokens.output"))
        start_ns = _int(span.get("start_unix_nano"))
        fingerprint = (session_key, session_id, input_tokens, output_tokens)
        previous = recent.get(fingerprint)
        if previous is not None and abs(start_ns - previous) <= 5_000_000_000:
            continue
        recent[fingerprint] = start_ns
        bucket = sessions[session_key]
        bucket["input"] += input_tokens
        bucket["output"] += output_tokens
        bucket["turn_count"] += 1
        role = "parent" if session_key == f"agent:{agent_id}:main" else "child"
        _append(
            usage_events, timestamp_ns=start_ns, session_id=session_id,
            session_key=session_key, role=role, event_type="model_usage",
            input_tokens=input_tokens, output_tokens=output_tokens,
            request_latency_ms=span.get("duration_ms"),
        )

    return dict(sessions), usage_events


def _serving_tokens(run_dir: Path) -> tuple[dict, list[dict]]:
    path = run_dir / "conversation" / "llm-raw-responses.jsonl"
    totals = {"processed_input": 0, "generated_output": 0, "request_count": 0}
    events = []
    if not path.is_file():
        return totals, events
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = row.get("response") or {}
        usage = response.get("usage") or {}
        prompt = _int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion = _int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )
        totals["processed_input"] += prompt
        totals["generated_output"] += completion
        totals["request_count"] += 1
        metrics = row.get("metrics") or {}
        timestamp_ns = _timestamp_ns(row.get("ts"))
        _append(
            events, timestamp_ns=timestamp_ns,
            session_id=str(row.get("sessionId") or ""), session_key="",
            role="serving", event_type="model_request",
            input_tokens=prompt, output_tokens=completion,
            request_latency_ms=metrics.get("total_ms"),
            http_status=row.get("httpStatus"), model=row.get("model"),
        )
    return totals, events


def _active_intervals(run_dir: Path) -> list[dict]:
    intervals = []
    for span in _read_spans(run_dir):
        if span.get("name") not in {"openclaw.model.usage", "openclaw.tool.call"}:
            continue
        attrs = span.get("attrs") or {}
        duration_ms = float(span.get("duration_ms") or 0)
        # OpenClaw also emits run-level aggregate model.usage spans whose
        # interval covers the whole session. They are token summaries, not an
        # in-flight request, and would classify every stuck diagnostic as
        # legitimate activity. Per-request synthesized spans are authoritative.
        if (
            span.get("name") == "openclaw.model.usage"
            and attrs.get("openclaw.span.source") != "raw_response_log"
            and duration_ms > 300_000
        ):
            continue
        start = _int(span.get("start_unix_nano"))
        end = _int(span.get("end_unix_nano"))
        if not end and start:
            end = start + int(duration_ms * 1e6)
        intervals.append({
            "start": start,
            "end": end,
            "session_key": str(attrs.get("openclaw.sessionKey") or ""),
            "activity": (
                "model_request"
                if span.get("name") == "openclaw.model.usage"
                else f"tool:{attrs.get('openclaw.tool.name') or 'unknown'}"
            ),
        })
    return intervals


def _interval_union_seconds(
    intervals: Iterable[dict], start_ns: int, end_ns: int
) -> float:
    clipped = []
    for interval in intervals:
        start = max(start_ns, _int(interval.get("start")))
        end = min(end_ns, _int(interval.get("end")))
        if start and end > start:
            clipped.append((start, end))
    if not clipped:
        return 0.0
    merged: list[list[int]] = []
    for start, end in sorted(clipped):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged) / 1e9


def _execution_time_observability(
    run_dir: Path,
    events: list[dict],
    execution_start_ns: int,
    execution_end_ns: int,
) -> dict:
    """Separate observed work from orchestration gaps without hiding latency."""
    if not execution_start_ns or execution_end_ns <= execution_start_ns:
        return {}

    intervals = _active_intervals(run_dir)
    model_intervals = [
        interval for interval in intervals
        if interval.get("activity") == "model_request"
    ]
    tool_intervals = [
        interval for interval in intervals
        if str(interval.get("activity") or "").startswith("tool:")
    ]
    observed_seconds = _interval_union_seconds(
        intervals, execution_start_ns, execution_end_ns
    )
    wall_seconds = (execution_end_ns - execution_start_ns) / 1e9

    parent_events = sorted(
        (
            event for event in events
            if event.get("role") == "parent"
            and event.get("event_type") in {"yield", "resume"}
            and _int(event.get("timestamp_ns"))
        ),
        key=lambda event: _int(event.get("timestamp_ns")),
    )
    yield_wait_intervals = []
    waiting_since = 0
    for event in parent_events:
        timestamp_ns = _int(event.get("timestamp_ns"))
        if event.get("event_type") == "yield":
            if not waiting_since:
                waiting_since = timestamp_ns
        elif waiting_since and timestamp_ns >= waiting_since:
            yield_wait_intervals.append({
                "start": waiting_since,
                "end": timestamp_ns,
            })
            waiting_since = 0
    if waiting_since:
        yield_wait_intervals.append({
            "start": waiting_since,
            "end": execution_end_ns,
        })

    return {
        "agent_end_to_end_seconds": wall_seconds,
        "observed_operation_union_seconds": observed_seconds,
        "observed_model_union_seconds": _interval_union_seconds(
            model_intervals, execution_start_ns, execution_end_ns
        ),
        "observed_tool_union_seconds": _interval_union_seconds(
            tool_intervals, execution_start_ns, execution_end_ns
        ),
        "parent_yield_wait_seconds": _interval_union_seconds(
            yield_wait_intervals, execution_start_ns, execution_end_ns
        ),
        "unattributed_orchestration_seconds": max(
            0.0, wall_seconds - observed_seconds
        ),
        "policy": {
            "primary_latency": (
                "agent_end_to_end_seconds includes yield, queue, continuation, "
                "retry, and quiescence overhead"
            ),
            "active_latency": (
                "union of observed model-request and tool-call intervals "
                "across parent and children"
            ),
            "unattributed_warning": (
                "not a pure idle metric; it also includes runtime activity "
                "missing from available spans"
            ),
        },
    }


def _loop_termination(events: list[dict]) -> dict:
    loop_events = [
        event for event in events
        if event.get("event_type") == "loop_detected"
        and event.get("terminal") is True
    ]
    if not loop_events:
        return {}
    terminal = loop_events[-1]
    detector = str(terminal.get("detector") or "compaction_loop_persisted")
    return {
        "termination_reason": detector,
        "loop_detector": detector,
        "loop_tool": terminal.get("tool_name"),
        "loop_count": terminal.get("count"),
    }


def _runtime_observability(
    run_dir: Path,
    events: list[dict],
    agent_id: str,
    *,
    execution_start_ns: int = 0,
    execution_end_ns: int = 0,
) -> dict:
    child_timing = _child_runtime_observability(run_dir, agent_id)
    spawn_attempt_count = sum(
        event.get("event_type") == "spawn" for event in events
    )
    announce_timeouts = [
        event for event in events if event["event_type"] == "announce_timeout"
    ]
    announces = [event for event in events if event["event_type"] == "announce"]
    loop_events = [
        event for event in events if event.get("event_type") == "loop_detected"
    ]
    strict_done_events = [
        event for event in events if event.get("event_type") == "done"
    ]
    markdown_done_events = [
        event for event in events
        if event.get("event_type") == "done_declaration"
        and event.get("marker_style") == "markdown_done_declaration"
    ]
    announce_groups: dict[str, dict] = {}
    for event in [*announce_timeouts, *announces]:
        group_key = str(
            event.get("child_session_key")
            or event.get("announce_run_id")
            or "unattributed"
        )
        group = announce_groups.setdefault(group_key, {
            "child_session_key": event.get("child_session_key"),
            "announce_run_ids": [],
            "attempt_count": 0,
            "success_count": 0,
            "timeout_count": 0,
            "retry_count": 0,
            "latency_ms": [],
        })
        run_id = event.get("announce_run_id")
        if run_id and run_id not in group["announce_run_ids"]:
            group["announce_run_ids"].append(run_id)
        group["attempt_count"] += 1
        latency = event.get("latency_ms")
        if latency is not None:
            group["latency_ms"].append(_int(latency))
        if event["event_type"] == "announce_timeout":
            group["timeout_count"] += 1
            if event.get("retry_attempt") is not None:
                group["retry_count"] += 1
        else:
            group["success_count"] += 1

    intervals = _active_intervals(run_dir)
    stuck_events = [event for event in events if event["event_type"] == "stuck_session"]
    stuck_by_session: dict[str, int] = defaultdict(int)
    classifications: dict[str, int] = defaultdict(int)
    for event in stuck_events:
        timestamp_ns = _int(event.get("timestamp_ns"))
        session_key = str(event.get("session_key") or "")
        activity = sorted({
            interval["activity"]
            for interval in intervals
            if interval["start"] <= timestamp_ns <= interval["end"]
            and (
                not session_key
                or not interval["session_key"]
                or interval["session_key"] == session_key
            )
        })
        classification = "active_operation" if activity else "unresolved"
        event["concurrent_activity"] = activity
        event["diagnostic_classification"] = classification
        stuck_by_session[session_key or "unknown"] += 1
        classifications[classification] += 1

    announce_latencies = [
        _int(event.get("latency_ms"))
        for event in [*announce_timeouts, *announces]
        if event.get("latency_ms") is not None
    ]
    unattributed_timeouts = sum(
        not event.get("child_session_key")
        and not event.get("announce_run_id")
        for event in announce_timeouts
    )
    return {
        **child_timing,
        "execution_time": _execution_time_observability(
            run_dir, events, execution_start_ns, execution_end_ns
        ),
        "spawn_attempt_count": spawn_attempt_count,
        "unmaterialized_spawn_attempt_count": max(
            0, spawn_attempt_count - child_timing["child_session_count"]
        ),
        "spawn_attempt_attribution": (
            "best-effort aggregate: durable transcript spawn calls minus "
            "durable child sessions; no per-attempt child identity inferred"
        ),
        "spawn_runtime_policy": (
            "failed/unmaterialized spawn waits remain in agent_execution_seconds "
            "but never create child runtime"
        ),
        "announce_attempt_count": len(announce_timeouts) + len(announces),
        "announce_timeout_count": len(announce_timeouts),
        "announce_retry_count": sum(
            event.get("retry_attempt") is not None for event in announce_timeouts
        ),
        "announce_success_count": len(announces),
        "announce_latency_ms": announce_latencies,
        "announce_by_child": announce_groups,
        "announce_attribution": {
            "unattributed_timeout_count": unattributed_timeouts,
            "identity_source": "explicit gateway log fields only",
            "limitation": (
                "OpenClaw v2026.4.22 retry warnings omit childSessionKey "
                "and announce runId; Agent_Bench does not infer identity "
                "from timing."
            ),
        },
        "loop_detection_count": len(loop_events),
        "strict_done_marker_count": len(strict_done_events),
        "markdown_done_declaration_count": len(markdown_done_events),
        "done_marker_policy": {
            "strict_bare_done": "runtime completion evidence",
            "markdown_done_declaration": "measurement-only declaration",
        },
        "loop_detections": [
            {
                "timestamp": event.get("timestamp"),
                "detector": event.get("detector"),
                "tool_name": event.get("tool_name"),
                "count": event.get("count"),
                "level": event.get("level"),
                "action": event.get("action"),
                "terminal": event.get("terminal", False),
            }
            for event in loop_events
        ],
        "stuck_session_count": len(stuck_events),
        "stuck_session_by_session": dict(stuck_by_session),
        "stuck_session_classification": dict(classifications),
        "stuck_diagnostic_policy": {
            "runtime_version": "v2026.4.22",
            "default_warn_after_ms": 120000,
            "heartbeat_interval_ms": 30000,
            "trigger": "session state=processing with no diagnostic activity",
            "is_failure": False,
        },
    }


def build_openclaw_measurement(
    run_dir: Path,
    *,
    agent_id: str,
    execution_start_ns: int = 0,
    execution_end_ns: int = 0,
) -> dict:
    """Write trajectory/token artifacts and return their manifest payload."""
    logical, usage_events = _logical_tokens(run_dir, agent_id)
    serving, request_events = _serving_tokens(run_dir)
    events = _session_events(run_dir, agent_id)
    events.extend(_gateway_events(run_dir))
    events.extend(usage_events)
    events.extend(request_events)
    runtime_observability = _runtime_observability(
        run_dir,
        events,
        agent_id,
        execution_start_ns=execution_start_ns,
        execution_end_ns=execution_end_ns,
    )
    termination = _loop_termination(events)
    events.sort(key=lambda event: (event.get("timestamp_ns") or 0, event["event_type"]))

    trajectory_path = run_dir / "trajectory.jsonl"
    with trajectory_path.open("w") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    accounting = {
        "trajectory_tokens": logical,
        "serving_tokens": serving,
        "runtime_observability": runtime_observability,
        "sources": {
            "trajectory_tokens": (
                "deduplicated assistant usage in durable session transcripts; "
                "session-attributed diagnostics OTel fallback"
            ),
            "serving_tokens": "raw OpenAI-compatible HTTP responses",
        },
    }
    accounting_path = run_dir / "token_accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2))
    observability_path = run_dir / "runtime_observability.json"
    observability_path.write_text(json.dumps(runtime_observability, indent=2))
    result = {
        **accounting,
        "trajectory_path": str(trajectory_path),
        "token_accounting_path": str(accounting_path),
        "runtime_observability_path": str(observability_path),
    }
    result.update(termination)
    return result


def append_trajectory_event(run_dir: Path, event_type: str, **fields: Any) -> None:
    path = run_dir / "trajectory.jsonl"
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_ns": int(datetime.now(timezone.utc).timestamp() * 1e9),
        "session_id": None,
        "session_key": None,
        "role": "harness",
        "event_type": event_type,
        **fields,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
