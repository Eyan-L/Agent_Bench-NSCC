"""Build machine-readable OpenClaw trajectory and token artifacts."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
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
    parts = []
    for item in message.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _is_heartbeat_request(text: str) -> bool:
    """Recognize OpenClaw heartbeat requester turns."""
    normalized = " ".join(str(text or "").lower().split())
    return (
        "read heartbeat.md" in normalized
        and "heartbeat_ok" in normalized
    )


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

    for path, session_key, entry in _session_sources(sessions_dir):
        session_id = str(entry.get("sessionId") or path.stem)
        role = "parent" if session_key == f"agent:{agent_id}:main" else "child"
        session_state = str(entry.get("status") or "unknown")
        seen_assistant: set[str] = set()
        seen_tool_results: set[str] = set()
        seen_resumes: set[tuple[int, str]] = set()
        heartbeat_turn = False
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
                if role == "parent" and _is_heartbeat_request(text):
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

            if stop_reason == "stop":
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role,
                    event_type="assistant_stop", text=text,
                    stop_reason=stop_reason, session_state=session_state,
                )
                if re.search(r"(?im)^\s*DONE(?:\s*$|\b)", text):
                    _append(
                        events, timestamp_ns=timestamp_ns, session_id=session_id,
                        session_key=session_key, role=role, event_type="done",
                        text=text, session_state=session_state,
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
        _append(
            events, timestamp_ns=0,
            session_id=str(entry.get("sessionId") or ""),
            session_key=str(session_key), role="child", event_type=event_type,
            session_state=status, spawned_by=entry.get("spawnedBy"),
        )
    return events


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
        if "subagent announce completion" in lowered and "timeout" in lowered:
            event_type = "announce_timeout"
            timeout = re.search(r"timeout after (\d+)ms", lowered)
            fields["latency_ms"] = _int(timeout.group(1)) if timeout else None
            retry = re.search(r"retrying (\d+)/(\d+)", lowered)
            fields["retry_attempt"] = _int(retry.group(1)) if retry else None
            fields["max_attempts"] = _int(retry.group(2)) if retry else None
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
        elif "compaction" in lowered:
            event_type = "compaction"
        if event_type:
            _append(
                events, timestamp_ns=timestamp_ns, session_id="",
                session_key=str(fields.pop("session_key", "") or ""),
                role="runtime", event_type=event_type, **fields,
            )
            if event_type == "context_overflow" and "compaction" in lowered:
                attempts = re.search(r"compactionAttempts=(\d+)", text)
                _append(
                    events, timestamp_ns=timestamp_ns, session_id="",
                    session_key="", role="runtime", event_type="compaction",
                    attempts=_int(attempts.group(1)) if attempts else None,
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
        role = "parent" if session_key == f"agent:{agent_id}:main" else "child"
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
            bucket = sessions[session_key or f"session:{session_id}"]
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


def _parent_resume_count(run_dir: Path, agent_id: str) -> int:
    """Count requester episodes after the initial parent request.

    Compaction can replay the initial requester message at a new timestamp, so
    text, rather than timestamp, is the durable identity used here. OpenClaw
    v2026.4.22 does not persist a stable id on requester messages; two genuine
    byte-identical resumes are therefore indistinguishable and intentionally
    collapse until the runtime exposes a durable event identity.
    """
    sessions_dir = run_dir / "conversation" / "sessions"
    if not sessions_dir.is_dir():
        return 0
    messages: list[str] = []
    seen: set[str] = set()
    for path, session_key, _entry in _session_sources(sessions_dir):
        if session_key != f"agent:{agent_id}:main":
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
            if _is_heartbeat_request(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            messages.append(text)
    return max(0, len(messages) - 1)


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


def _runtime_observability(run_dir: Path, events: list[dict]) -> dict:
    announce_timeouts = [
        event for event in events if event["event_type"] == "announce_timeout"
    ]
    announces = [event for event in events if event["event_type"] == "announce"]
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
    return {
        "announce_attempt_count": len(announce_timeouts) + len(announces),
        "announce_timeout_count": len(announce_timeouts),
        "announce_retry_count": sum(
            event.get("retry_attempt") is not None for event in announce_timeouts
        ),
        "announce_success_count": len(announces),
        "announce_latency_ms": announce_latencies,
        "announce_by_child": announce_groups,
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
) -> dict:
    """Write trajectory/token artifacts and return their manifest payload."""
    logical, usage_events = _logical_tokens(run_dir, agent_id)
    serving, request_events = _serving_tokens(run_dir)
    events = _session_events(run_dir, agent_id)
    events.extend(_gateway_events(run_dir))
    events.extend(usage_events)
    events.extend(request_events)
    runtime_observability = _runtime_observability(run_dir, events)
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
    return {
        **accounting,
        "trajectory_path": str(trajectory_path),
        "token_accounting_path": str(accounting_path),
        "runtime_observability_path": str(observability_path),
    }


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
