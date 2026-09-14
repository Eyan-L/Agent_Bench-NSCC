"""RoadmapBench measurement compatibility for OpenClaw runtimes.

The durable OpenClaw session transcript remains the primary trajectory source.
Runtime recovery diagnostics can be emitted either on the agent CLI stderr or
on the Gateway log depending on the OpenClaw/SIF build. This adapter merges the
timestamped recovery records from both sources after the shared OpenClaw
measurement builder has run.

OpenClaw v2026.4.22 can rely on Agent_Bench's raw-response log for exact
per-request serving totals. OpenClaw v2026.9.2 exposes native per-request model
call OpenTelemetry spans with prompt/output usage and latency. When the raw
response log is absent, this adapter uses those native model-call spans as the
serving-token/request fallback instead of patching pi-ai internals.

Some runtimes can omit assistant ``usage`` objects from durable transcripts. In
that case the serving aggregate can still provide an explicitly labelled,
unattributed logical-token fallback.

Older OpenClaw builds also do not always emit the gateway announce diagnostics
consumed by the shared measurement parser. For RoadmapBench we can still infer a
successful completion announce conservatively from durable semantics: a real
child-completion event followed by a later parent resume.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_LINE_RE = re.compile(r"^(\S+)\s+(.*)$")
_OVERFLOW_MARKERS = (
    "context overflow",
    "context_overflow",
    "context-overflow",
    "context length exceeded",
    "maximum context length",
    "context size exceeds",
)
_COMPACTION_COMPLETED_RE = re.compile(
    r"(?:"
    r"\b(?:auto[- ]?)?compaction\b.{0,180}\b(?:succeeded|successful(?:ly)?|completed)\b"
    r"|\bsuccessfully\s+compact(?:ed|ing)\b"
    r")"
)
_COMPACTION_NONTERMINAL_MARKERS = (
    "guard",
    "armed",
    "arming",
    "starting",
    "started",
    "attempting",
    "triggering",
    "triggered",
    "diagnostic",
    "candidate",
    "scheduled",
)
_RECOVERY_LOGS = (
    ("openclaw_stderr.log", "openclaw_stderr"),
    ("openclaw_gateway.log", "openclaw_gateway"),
)
_COMPACTION_DEDUP_NS = 10 * 1_000_000_000
_ANNOUNCE_PAIR_WINDOW_NS = 5 * 60 * 1_000_000_000


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


def _iso_from_ns(value: int) -> str | None:
    if not value:
        return None
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc).isoformat()


def _runtime_event(
    timestamp_ns: int,
    event_type: str,
    text: str,
    *,
    source: str,
    **fields: Any,
) -> dict:
    return {
        "timestamp": _iso_from_ns(timestamp_ns),
        "timestamp_ns": timestamp_ns,
        "session_id": None,
        "session_key": None,
        "role": "runtime",
        "event_type": event_type,
        "source": source,
        "message": text,
        **fields,
    }


def _is_completed_compaction(lowered: str) -> bool:
    """Return true only for a terminal successful compaction diagnostic."""
    if not _COMPACTION_COMPLETED_RE.search(lowered):
        return False
    return not any(marker in lowered for marker in _COMPACTION_NONTERMINAL_MARKERS)


def _tool_result_truncation(lowered: str) -> tuple[bool, int | None]:
    normalized = lowered.replace("tool-result", "tool result").replace(
        "tool_result", "tool result"
    )
    if "tool result" not in normalized:
        return False, None
    if not any(
        marker in normalized
        for marker in ("truncat", "prun", "remov", "drop", "discard")
    ):
        return False, None

    patterns = (
        r"(?:truncat\w*|prun\w*|remov\w*|drop\w*|discard\w*)[^\d]{0,40}(\d+)\s+(?:old\s+)?tool\s+results?",
        r"(\d+)\s+(?:old\s+)?tool\s+results?[^\n]{0,50}(?:truncat\w*|prun\w*|remov\w*|drop\w*|discard\w*)",
        r"tool\s+results?[^\d]{0,60}(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return True, int(match.group(1))
    return True, None


def _timestamped_recovery_lines(run_dir: Path) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for filename, source in _RECOVERY_LOGS:
        path = Path(run_dir) / filename
        if not path.is_file():
            continue
        for raw_line in path.read_text(errors="replace").splitlines():
            line = _ANSI_RE.sub("", raw_line).strip()
            match = _LINE_RE.match(line)
            if not match:
                continue
            timestamp_ns = _timestamp_ns(match.group(1))
            if not timestamp_ns:
                continue
            rows.append((timestamp_ns, source, match.group(2).strip()))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return rows


def stderr_recovery_events(run_dir: Path) -> list[dict]:
    """Parse timestamped recovery diagnostics from stderr and Gateway logs.

    The legacy function name is retained because existing Roadmap measurement
    code imports it directly. Untimestamped lines are deliberately ignored so
    the synthesized trajectory events always retain a real temporal position.
    Duplicate diagnostics mirrored to both logs are collapsed by timestamp,
    event type, and message.
    """
    events: list[dict] = []
    seen: set[tuple[int, str, str]] = set()
    last_compaction_ns = 0

    for timestamp_ns, source, text in _timestamped_recovery_lines(Path(run_dir)):
        lowered = " ".join(text.lower().split())
        truncation, truncated_count = _tool_result_truncation(lowered)

        candidates: list[dict] = []
        overflow = any(marker in lowered for marker in _OVERFLOW_MARKERS)
        if overflow:
            candidates.append(
                _runtime_event(
                    timestamp_ns,
                    "context_overflow",
                    text,
                    source=source,
                    recovery=(
                        "compaction"
                        if "compaction" in lowered
                        else "tool_result_truncation"
                        if truncation
                        else None
                    ),
                )
            )

        if _is_completed_compaction(lowered):
            if (
                not last_compaction_ns
                or timestamp_ns - last_compaction_ns > _COMPACTION_DEDUP_NS
            ):
                candidates.append(
                    _runtime_event(
                        timestamp_ns,
                        "compaction",
                        text,
                        source=source,
                        count=1,
                        outcome="completed",
                    )
                )
                last_compaction_ns = timestamp_ns

        if truncation:
            candidates.append(
                _runtime_event(
                    timestamp_ns,
                    "tool_result_truncation",
                    text,
                    source=source,
                    truncated_tool_results=truncated_count,
                )
            )

        for event in candidates:
            key = (
                int(event["timestamp_ns"]),
                str(event["event_type"]),
                str(event["message"]),
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
    return events


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_trajectory(path: Path, events: list[dict]) -> None:
    events.sort(
        key=lambda event: (
            int(event.get("timestamp_ns") or 0),
            str(event.get("event_type") or ""),
        )
    )
    with path.open("w") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _merge_recovery_events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "trajectory.jsonl"
    existing = _read_jsonl(path)
    additions = stderr_recovery_events(run_dir)
    if not additions:
        return []

    exact = {
        (
            int(event.get("timestamp_ns") or 0),
            str(event.get("event_type") or ""),
            str(event.get("message") or event.get("error") or ""),
        )
        for event in existing
    }
    added: list[dict] = []
    for event in additions:
        key = (
            int(event.get("timestamp_ns") or 0),
            str(event.get("event_type") or ""),
            str(event.get("message") or ""),
        )
        if key in exact:
            continue
        exact.add(key)
        existing.append(event)
        added.append(event)

    _write_trajectory(path, existing)
    return added


def _is_native_model_request_span(span: dict) -> bool:
    attrs = span.get("attrs") or {}
    observation = str(attrs.get("openclaw.model_call.observation_unit") or "")
    if observation:
        return observation == "request"
    if span.get("name") == "openclaw.model.call":
        return True
    return "openclaw.model_call.usage.prompt_tokens" in attrs


def _native_model_call_serving(run_dir: Path) -> tuple[dict, list[dict]]:
    totals = {"processed_input": 0, "generated_output": 0, "request_count": 0}
    events: list[dict] = []
    seen: set[tuple[Any, ...]] = set()

    for span in _read_jsonl(Path(run_dir) / "spans.jsonl"):
        if not _is_native_model_request_span(span):
            continue
        attrs = span.get("attrs") or {}
        trace_id = span.get("trace_id") or span.get("traceId")
        span_id = span.get("span_id") or span.get("spanId")
        start_ns = int(span.get("start_unix_nano") or 0)
        end_ns = int(span.get("end_unix_nano") or 0)
        prompt = int(
            attrs.get("openclaw.model_call.usage.prompt_tokens")
            or attrs.get("gen_ai.usage.input_tokens")
            or 0
        )
        completion = int(
            attrs.get("openclaw.model_call.usage.output_tokens")
            or attrs.get("gen_ai.usage.output_tokens")
            or 0
        )
        fingerprint: tuple[Any, ...]
        if trace_id and span_id:
            fingerprint = ("ids", trace_id, span_id)
        else:
            fingerprint = (
                "content",
                start_ns,
                end_ns,
                attrs.get("openclaw.provider"),
                attrs.get("openclaw.model"),
                prompt,
                completion,
            )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        totals["processed_input"] += prompt
        totals["generated_output"] += completion
        totals["request_count"] += 1
        duration_ms = span.get("duration_ms")
        if duration_ms is None and start_ns and end_ns and end_ns >= start_ns:
            duration_ms = (end_ns - start_ns) / 1e6
        events.append(
            {
                "timestamp": _iso_from_ns(start_ns),
                "timestamp_ns": start_ns or None,
                "session_id": attrs.get("openclaw.sessionId"),
                "session_key": attrs.get("openclaw.sessionKey"),
                "role": "serving",
                "event_type": "model_request",
                "source": "native_model_call_otel",
                "input_tokens": prompt,
                "output_tokens": completion,
                "request_latency_ms": duration_ms,
                "time_to_first_byte_ms": attrs.get(
                    "openclaw.model_call.time_to_first_byte_ms"
                ),
                "request_payload_bytes": attrs.get(
                    "openclaw.model_call.request_bytes"
                ),
                "response_stream_bytes": attrs.get(
                    "openclaw.model_call.response_bytes"
                ),
                "provider": attrs.get("openclaw.provider"),
                "model": attrs.get("openclaw.model")
                or attrs.get("gen_ai.request.model"),
            }
        )
    return totals, events


def _apply_native_serving_fallback(run_dir: Path, result: dict) -> list[dict]:
    serving = result.get("serving_tokens") or {}
    if int(serving.get("request_count") or 0) > 0:
        result.setdefault("serving_measurement_source", "raw_response_log")
        return []

    totals, events = _native_model_call_serving(Path(run_dir))
    if int(totals.get("request_count") or 0) <= 0:
        return []

    result["serving_tokens"] = totals
    result["serving_measurement_source"] = "native_model_call_otel"

    path = Path(run_dir) / "trajectory.jsonl"
    existing = _read_jsonl(path)
    exact = {
        (
            int(event.get("timestamp_ns") or 0),
            str(event.get("event_type") or ""),
            str(event.get("source") or ""),
            str(event.get("model") or ""),
        )
        for event in existing
    }
    added: list[dict] = []
    for event in events:
        key = (
            int(event.get("timestamp_ns") or 0),
            str(event.get("event_type") or ""),
            str(event.get("source") or ""),
            str(event.get("model") or ""),
        )
        if key in exact:
            continue
        exact.add(key)
        existing.append(event)
        added.append(event)
    if added:
        _write_trajectory(path, existing)
    return added


def _infer_announce_observability(run_dir: Path, runtime: dict) -> dict:
    """Infer successful completion announces from durable child/resume order."""
    rows = _read_jsonl(Path(run_dir) / "trajectory.jsonl")
    completions = sorted(
        int(event.get("timestamp_ns") or 0)
        for event in rows
        if event.get("event_type") == "child_completion"
        and event.get("role") == "child"
        and int(event.get("timestamp_ns") or 0) > 0
    )
    resumes = sorted(
        int(event.get("timestamp_ns") or 0)
        for event in rows
        if event.get("event_type") == "resume"
        and event.get("role") == "parent"
        and int(event.get("timestamp_ns") or 0) > 0
    )

    used: set[int] = set()
    inferred_latencies: list[float] = []
    for completion_ns in completions:
        for index, resume_ns in enumerate(resumes):
            if index in used or resume_ns <= completion_ns:
                continue
            if resume_ns - completion_ns > _ANNOUNCE_PAIR_WINDOW_NS:
                break
            used.add(index)
            inferred_latencies.append((resume_ns - completion_ns) / 1e9)
            break

    inferred_attempts = len(completions)
    inferred_successes = len(inferred_latencies)
    if inferred_attempts == 0:
        return {}

    existing_attempts = int(runtime.get("announce_attempt_count") or 0)
    existing_successes = int(runtime.get("announce_success_count") or 0)
    attempts = max(existing_attempts, inferred_attempts)
    successes = max(existing_successes, inferred_successes)
    update: dict[str, Any] = {
        "announce_attempt_count": attempts,
        "announce_success_count": successes,
        "announce_failure_count": max(0, attempts - successes),
        "announce_inferred_attempt_count": inferred_attempts,
        "announce_inferred_success_count": inferred_successes,
    }
    if inferred_successes > existing_successes:
        update["announce_measurement_source"] = (
            "durable_trajectory_child_completion_to_parent_resume"
        )
        update["announce_inferred_latency_seconds"] = inferred_latencies
    return update


def _logical_usage_missing(tokens: Any) -> bool:
    if not isinstance(tokens, dict):
        return True
    total_input = int(tokens.get("total_logical_input") or 0)
    total_output = int(tokens.get("total_logical_output") or 0)
    parent = tokens.get("parent") or {}
    child_turns = sum(
        int(value.get("turn_count") or 0)
        for value in (tokens.get("children") or {}).values()
        if isinstance(value, dict)
    )
    turns = int(parent.get("turn_count") or 0) + child_turns
    return turns == 0 and total_input == 0 and total_output == 0


def _apply_serving_token_fallback(result: dict) -> tuple[int | None, int | None]:
    tokens = result.get("trajectory_tokens")
    serving = result.get("serving_tokens") or {}
    request_count = int(serving.get("request_count") or 0)
    if not _logical_usage_missing(tokens) or request_count <= 0:
        return None, None

    input_tokens = int(serving.get("processed_input") or 0)
    output_tokens = int(serving.get("generated_output") or 0)
    if input_tokens <= 0 and output_tokens <= 0:
        return None, None

    serving_source = str(result.get("serving_measurement_source") or "raw_response_log")
    accounting_source = (
        "raw_response_fallback_unattributed"
        if serving_source == "raw_response_log"
        else f"{serving_source}_fallback_unattributed"
    )
    fallback = {
        "parent": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "turn_count": 0,
            "resumed_turns": int(
                ((tokens or {}).get("parent") or {}).get("resumed_turns") or 0
            ),
        },
        "children": {},
        "unattributed": {
            "input": input_tokens,
            "output": output_tokens,
            "request_count": request_count,
        },
        "total_logical_input": input_tokens,
        "total_logical_output": output_tokens,
        "accounting_source": accounting_source,
        "attribution_quality": "aggregate_only",
    }
    result["trajectory_tokens"] = fallback
    result["total_prompt_tokens"] = input_tokens
    result["total_completion_tokens"] = output_tokens
    return input_tokens, output_tokens


def _recovery_source_filenames(run_dir: Path) -> list[str]:
    return [
        filename
        for filename, _ in _RECOVERY_LOGS
        if (Path(run_dir) / filename).is_file()
    ]


def _rewrite_accounting_artifacts(
    run_dir: Path,
    result: dict,
    added_events: list[dict],
) -> None:
    runtime = dict(result.get("runtime_observability") or {})
    runtime.update(_infer_announce_observability(run_dir, runtime))
    source_files = _recovery_source_filenames(run_dir)
    runtime.update(
        {
            # Keep the legacy stderr_* names for report compatibility. The
            # values now cover both supported OpenClaw recovery log surfaces.
            "stderr_recovery_event_count": len(added_events),
            "stderr_compaction_count": sum(
                event.get("event_type") == "compaction" for event in added_events
            ),
            "stderr_context_overflow_count": sum(
                event.get("event_type") == "context_overflow"
                for event in added_events
            ),
            "tool_result_truncation_count": sum(
                event.get("event_type") == "tool_result_truncation"
                for event in added_events
            ),
            "truncated_tool_result_total": sum(
                int(event.get("truncated_tool_results") or 0)
                for event in added_events
                if event.get("event_type") == "tool_result_truncation"
            ),
            "stderr_recovery_source": "+".join(source_files),
            "recovery_log_sources": source_files,
        }
    )
    result["runtime_observability"] = runtime

    observability_path = Path(str(result.get("runtime_observability_path") or ""))
    if not observability_path.is_absolute():
        observability_path = Path(run_dir) / (
            observability_path.name or "runtime_observability.json"
        )
    observability_path.write_text(json.dumps(runtime, indent=2))
    result["runtime_observability_path"] = str(observability_path)

    accounting_path = Path(str(result.get("token_accounting_path") or ""))
    if not accounting_path.is_absolute():
        accounting_path = Path(run_dir) / (
            accounting_path.name or "token_accounting.json"
        )
    accounting: dict[str, Any] = {}
    if accounting_path.is_file():
        try:
            value = json.loads(accounting_path.read_text())
            if isinstance(value, dict):
                accounting = value
        except json.JSONDecodeError:
            accounting = {}
    accounting["trajectory_tokens"] = result.get("trajectory_tokens") or {}
    accounting["serving_tokens"] = result.get("serving_tokens") or {}
    accounting["runtime_observability"] = runtime
    sources = dict(accounting.get("sources") or {})
    trajectory_source = (
        (result.get("trajectory_tokens") or {}).get("accounting_source")
        if isinstance(result.get("trajectory_tokens"), dict)
        else None
    )
    if trajectory_source == "raw_response_fallback_unattributed":
        sources["trajectory_tokens"] = (
            "raw OpenAI-compatible response usage fallback; aggregate totals "
            "only because durable session assistant usage was unavailable"
        )
    elif trajectory_source == "native_model_call_otel_fallback_unattributed":
        sources["trajectory_tokens"] = (
            "native OpenClaw model-call OTel usage fallback; aggregate totals "
            "only because durable session assistant usage was unavailable"
        )
    serving_source = str(result.get("serving_measurement_source") or "")
    if serving_source == "native_model_call_otel":
        sources["serving_tokens"] = (
            "OpenClaw v2026.9.2 native per-request model-call OTel spans"
        )
    elif serving_source == "raw_response_log":
        sources["serving_tokens"] = "raw OpenAI-compatible HTTP responses"
    sources["runtime_recovery_events"] = (
        "timestamped " + ", ".join(source_files)
        if source_files
        else "no supported OpenClaw recovery log found"
    )
    if runtime.get("announce_measurement_source"):
        sources["announce_measurement"] = runtime["announce_measurement_source"]
    accounting["sources"] = sources
    accounting_path.write_text(json.dumps(accounting, indent=2))
    result["token_accounting_path"] = str(accounting_path)


def augment_roadmap_measurement(run_dir: Path, result: dict) -> dict:
    """Add Roadmap-only runtime recovery, serving, announce, and token compatibility data."""
    result = dict(result)
    _apply_native_serving_fallback(Path(run_dir), result)
    added_events = _merge_recovery_events(Path(run_dir))
    _apply_serving_token_fallback(result)
    _rewrite_accounting_artifacts(Path(run_dir), result, added_events)
    return result


def install_roadmap_measurement_compat() -> None:
    """Wrap the shared builder once, affecting only ``agent_id=roadmap_bench``."""
    from agent_bench.benchmarks.conditions import openclaw_measurement as shared

    current: Callable[..., dict] = shared.build_openclaw_measurement
    if getattr(current, "_roadmap_measurement_compat", False):
        return

    def wrapped(run_dir: Path, *, agent_id: str) -> dict:
        result = current(run_dir, agent_id=agent_id)
        if agent_id != "roadmap_bench":
            return result
        return augment_roadmap_measurement(Path(run_dir), result)

    setattr(wrapped, "_roadmap_measurement_compat", True)
    shared.build_openclaw_measurement = wrapped
