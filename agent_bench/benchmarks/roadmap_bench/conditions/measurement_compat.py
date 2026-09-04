"""RoadmapBench measurement compatibility for OpenClaw v2026.4.22+.

The durable OpenClaw session transcript remains the primary trajectory source.
Some runtime recovery diagnostics, however, are emitted only on the agent CLI
stderr in the NSCC SIF (not in the session trajectory): context-overflow
recovery, successful compaction, and tool-result truncation. This adapter adds
those machine-readable events after the shared OpenClaw measurement builder has
run.

The same SIF can omit assistant ``usage`` objects from durable transcripts. In
that case the shared logical-token accounting has no attributable usage even
though ``llm-raw-responses.jsonl`` has exact per-request totals. We preserve the
normal transcript/OTel preference and use the already-computed raw serving
aggregate only as an explicitly labelled, unattributed fallback.
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
_COMPACTION_SUCCESS_MARKERS = (
    "succeed",
    "success",
    "complete",
    "completed",
    "retrying prompt",
    "retry prompt",
)


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


def _runtime_event(timestamp_ns: int, event_type: str, text: str, **fields: Any) -> dict:
    return {
        "timestamp": _iso_from_ns(timestamp_ns),
        "timestamp_ns": timestamp_ns,
        "session_id": None,
        "session_key": None,
        "role": "runtime",
        "event_type": event_type,
        "source": "openclaw_stderr",
        "message": text,
        **fields,
    }


def stderr_recovery_events(run_dir: Path) -> list[dict]:
    """Parse timestamped recovery diagnostics from ``openclaw_stderr.log``.

    Untimestamped stderr lines are deliberately ignored: trajectory events must
    retain a real temporal position rather than receiving a measurement-time
    timestamp that would distort concurrency/timeline analysis.
    """
    path = Path(run_dir) / "openclaw_stderr.log"
    if not path.is_file():
        return []

    events: list[dict] = []
    seen: set[tuple[int, str, str]] = set()
    for raw_line in path.read_text(errors="replace").splitlines():
        line = _ANSI_RE.sub("", raw_line).strip()
        match = _LINE_RE.match(line)
        if not match:
            continue
        timestamp_ns = _timestamp_ns(match.group(1))
        if not timestamp_ns:
            continue
        text = match.group(2).strip()
        lowered = " ".join(text.lower().split())

        candidates: list[dict] = []
        overflow = any(marker in lowered for marker in _OVERFLOW_MARKERS)
        if overflow:
            candidates.append(_runtime_event(
                timestamp_ns,
                "context_overflow",
                text,
                recovery=(
                    "compaction" if "compaction" in lowered
                    else "tool_result_truncation"
                    if "truncat" in lowered and "tool result" in lowered
                    else None
                ),
            ))

        if (
            "compaction" in lowered
            and any(marker in lowered for marker in _COMPACTION_SUCCESS_MARKERS)
            and not any(marker in lowered for marker in ("failed", "failure", "error"))
        ):
            candidates.append(_runtime_event(
                timestamp_ns,
                "compaction",
                text,
                outcome="completed",
            ))

        if "truncat" in lowered and "tool result" in lowered:
            count_match = re.search(
                r"(?:truncat\w*\s+)?(\d+)\s+tool\s+results?",
                lowered,
            )
            if count_match is None:
                count_match = re.search(
                    r"tool\s+results?[^\d]{0,40}(\d+)", lowered
                )
            candidates.append(_runtime_event(
                timestamp_ns,
                "tool_result_truncation",
                text,
                truncated_tool_results=(
                    int(count_match.group(1)) if count_match else None
                ),
            ))

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

    existing.sort(
        key=lambda event: (
            int(event.get("timestamp_ns") or 0),
            str(event.get("event_type") or ""),
        )
    )
    with path.open("w") as stream:
        for event in existing:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return added


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


def _apply_raw_response_token_fallback(result: dict) -> tuple[int | None, int | None]:
    tokens = result.get("trajectory_tokens")
    serving = result.get("serving_tokens") or {}
    request_count = int(serving.get("request_count") or 0)
    if not _logical_usage_missing(tokens) or request_count <= 0:
        return None, None

    input_tokens = int(serving.get("processed_input") or 0)
    output_tokens = int(serving.get("generated_output") or 0)
    if input_tokens <= 0 and output_tokens <= 0:
        return None, None

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
        "accounting_source": "raw_response_fallback_unattributed",
        "attribution_quality": "aggregate_only",
    }
    result["trajectory_tokens"] = fallback
    # ``OpenClawAgentCondition.run`` expands measurement last, so these keys
    # deliberately override the zero run-level agentMeta usage in the returned
    # result and therefore in manifest.total_*_tokens.
    result["total_prompt_tokens"] = input_tokens
    result["total_completion_tokens"] = output_tokens
    return input_tokens, output_tokens


def _rewrite_accounting_artifacts(run_dir: Path, result: dict, added_events: list[dict]) -> None:
    runtime = dict(result.get("runtime_observability") or {})
    runtime.update({
        "stderr_recovery_event_count": len(added_events),
        "stderr_compaction_count": sum(
            event.get("event_type") == "compaction" for event in added_events
        ),
        "stderr_context_overflow_count": sum(
            event.get("event_type") == "context_overflow" for event in added_events
        ),
        "tool_result_truncation_count": sum(
            event.get("event_type") == "tool_result_truncation" for event in added_events
        ),
        "stderr_recovery_source": "openclaw_stderr.log",
    })
    result["runtime_observability"] = runtime

    observability_path = Path(
        str(result.get("runtime_observability_path") or "")
    )
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
    if (
        isinstance(result.get("trajectory_tokens"), dict)
        and result["trajectory_tokens"].get("accounting_source")
        == "raw_response_fallback_unattributed"
    ):
        sources["trajectory_tokens"] = (
            "raw OpenAI-compatible response usage fallback; aggregate totals "
            "only because durable session assistant usage was unavailable"
        )
    sources["runtime_recovery_events"] = "timestamped openclaw_stderr.log"
    accounting["sources"] = sources
    accounting_path.write_text(json.dumps(accounting, indent=2))
    result["token_accounting_path"] = str(accounting_path)


def augment_roadmap_measurement(run_dir: Path, result: dict) -> dict:
    """Add Roadmap-only runtime recovery and token compatibility data."""
    result = dict(result)
    added_events = _merge_recovery_events(Path(run_dir))
    _apply_raw_response_token_fallback(result)
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
