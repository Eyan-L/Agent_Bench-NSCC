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
    by_file = {
        _session_file_name(entry): (key, entry)
        for key, entry in index.items()
        if isinstance(entry, dict) and _session_file_name(entry)
    }
    events: list[dict] = []

    for path in sorted(sessions_dir.glob("*.jsonl")):
        if path.name.endswith(".trajectory.jsonl"):
            continue
        session_key, entry = by_file.get(path.name, ("", {}))
        session_id = str(entry.get("sessionId") or path.stem)
        role = "parent" if session_key == f"agent:{agent_id}:main" else "child"
        session_state = str(entry.get("status") or "unknown")
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
                _append(
                    events, timestamp_ns=timestamp_ns, session_id=session_id,
                    session_key=session_key, role=role, event_type="resume",
                    text=_message_text(message), session_state=session_state,
                )
                continue
            if message_role == "toolResult":
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


def _logical_tokens(run_dir: Path, agent_id: str) -> tuple[dict, list[dict]]:
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

    parent_key = f"agent:{agent_id}:main"
    parent = dict(sessions.get(parent_key, {"input": 0, "output": 0, "turn_count": 0}))
    parent["resumed_turns"] = max(0, parent["turn_count"] - 1)
    children = {
        key: dict(value) for key, value in sessions.items() if key != parent_key
    }
    return {
        "parent": parent,
        "children": children,
        "total_logical_input": parent["input"] + sum(v["input"] for v in children.values()),
        "total_logical_output": parent["output"] + sum(v["output"] for v in children.values()),
    }, usage_events


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
        if attrs.get("openclaw.span.source") == "raw_response_log":
            continue
        start = _int(span.get("start_unix_nano"))
        end = _int(span.get("end_unix_nano"))
        if not end and start:
            end = start + int(float(span.get("duration_ms") or 0) * 1e6)
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
    attempts = [
        _int(event.get("retry_attempt")) for event in announce_timeouts
        if event.get("retry_attempt") is not None
    ]
    attempt_count = max(
        attempts or [len(announce_timeouts) + len(announces)]
    )

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
            and (not session_key or interval["session_key"] == session_key)
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
        "announce_attempt_count": attempt_count,
        "announce_timeout_count": len(announce_timeouts),
        "announce_retry_count": sum(
            event.get("retry_attempt") is not None for event in announce_timeouts
        ),
        "announce_success_count": len(announces),
        "announce_latency_ms": announce_latencies,
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
            "trajectory_tokens": "session-attributed diagnostics OTel usage",
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
