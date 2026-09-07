"""Roadmap compatibility for OpenClaw OTel span-name evolution.

OpenClaw v2026.4.22/Agent_Bench used ``openclaw.model.usage`` and
``openclaw.tool.call`` as the active model/tool spans consumed by the shared
stuck-session classifier.  OpenClaw v2026.9.2 natively emits per-request
``openclaw.model.call`` spans and ``openclaw.tool.execution`` spans.

Keep the shared measurement module untouched and install a small, idempotent
compatibility wrapper when RoadmapBench conditions are imported.  The wrapper
accepts both schemas and otherwise preserves the old interval semantics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


_MODEL_SPANS = {"openclaw.model.usage", "openclaw.model.call"}
_TOOL_SPANS = {"openclaw.tool.call", "openclaw.tool.execution"}


def install_roadmap_otel_activity_compat() -> None:
    from agent_bench.benchmarks.conditions import openclaw_measurement as shared

    current: Callable[[Path], list[dict]] = shared._active_intervals
    if getattr(current, "_roadmap_v2026_9_2_otel_compat", False):
        return

    def compatible_active_intervals(run_dir: Path) -> list[dict]:
        intervals: list[dict] = []
        for span in shared._read_spans(Path(run_dir)):
            name = str(span.get("name") or "")
            if name not in _MODEL_SPANS | _TOOL_SPANS:
                continue
            attrs = span.get("attrs") or {}
            duration_ms = float(span.get("duration_ms") or 0)

            # Old aggregate usage spans can cover the whole session and are not
            # an in-flight model request. Preserve the shared parser's filter.
            if (
                name == "openclaw.model.usage"
                and attrs.get("openclaw.span.source") != "raw_response_log"
                and duration_ms > 300_000
            ):
                continue

            # v2026.9.2 may also use model-call diagnostics for turn-level
            # lifecycle boundaries. Only request-level calls are serving work.
            if name == "openclaw.model.call":
                observation = str(
                    attrs.get("openclaw.model_call.observation_unit") or ""
                )
                if observation and observation != "request":
                    continue

            start = shared._int(span.get("start_unix_nano"))
            end = shared._int(span.get("end_unix_nano"))
            if not end and start:
                end = start + int(duration_ms * 1e6)

            if name in _MODEL_SPANS:
                activity = "model_request"
            else:
                tool_name = (
                    attrs.get("openclaw.tool.name")
                    or attrs.get("openclaw.toolName")
                    or attrs.get("gen_ai.tool.name")
                    or "unknown"
                )
                activity = f"tool:{tool_name}"

            intervals.append(
                {
                    "start": start,
                    "end": end,
                    "session_key": str(attrs.get("openclaw.sessionKey") or ""),
                    "activity": activity,
                }
            )
        return intervals

    setattr(
        compatible_active_intervals,
        "_roadmap_v2026_9_2_otel_compat",
        True,
    )
    shared._active_intervals = compatible_active_intervals
