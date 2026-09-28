"""Hermes-only recovery of delegated child work from durable trajectories."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def _recover_hermes_child_timing(record, original: dict[str, Any]) -> dict[str, Any]:
    """Recover child execution intervals from durable messages and tool spans.

    Hermes leaves session ``ended_at`` empty for delegated children. A final
    assistant stop message is the child terminal observation; the matching
    delegate_task span bounds it. Failed starts have no model work and are
    tracked separately from children that actually ran.
    """
    sessions_path = record.run_dir / "hermes_sessions.json"
    spans_path = record.run_dir / "spans.jsonl"
    db_path = record.run_dir / "hermes_state" / "state.db"
    children = original.get("child_runtime_by_session")
    if not isinstance(children, dict):
        return original
    try:
        sessions_doc = json.loads(sessions_path.read_text(errors="replace"))
        sessions = {str(s["id"]): s for s in sessions_doc["sessions"]}
        spans = [
            span for line in spans_path.open(errors="replace")
            if (span := json.loads(line)).get("attrs", {}).get("agent.tool.name")
            == "delegate_task"
        ]
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except (OSError, ValueError, KeyError, sqlite3.Error, json.JSONDecodeError):
        return original

    result = dict(original)
    recovered = {key: dict(value) for key, value in children.items()}
    completed: list[tuple[int, int]] = []
    failed: list[str] = []
    unknown: list[str] = []
    duration_discrepancies: list[float] = []
    try:
        tool_results = {
            call_id: content
            for call_id, content in connection.execute(
                "SELECT tool_call_id, content FROM messages "
                "WHERE role='tool' AND tool_call_id IS NOT NULL"
            )
        }
        span_by_child: dict[str, tuple[dict[str, Any], int]] = {}
        for span in spans:
            matching = sorted(
                (
                    (key, child) for key, child in recovered.items()
                    if isinstance(child.get("start_timestamp_ns"), int)
                    and span["start_unix_nano"] <= child["start_timestamp_ns"]
                    <= span["end_unix_nano"]
                ),
                key=lambda pair: pair[1]["start_timestamp_ns"],
            )
            for index, (key, _child) in enumerate(matching):
                span_by_child[key] = (span, index)

        for key, child in recovered.items():
            chain = [str(child["session_id"])] + list(
                child.get("continuation_session_ids") or []
            )
            ledger = [sessions.get(session_id) for session_id in chain]
            if any(session is None for session in ledger):
                unknown.append(key)
                continue
            api_calls = sum(int(session.get("api_call_count") or 0) for session in ledger)
            input_tokens = sum(int(session.get("input_tokens") or 0) for session in ledger)
            output_tokens = sum(int(session.get("output_tokens") or 0) for session in ledger)
            child["model_api_call_count"] = api_calls
            child["model_input_tokens"] = input_tokens
            child["model_output_tokens"] = output_tokens
            matched = span_by_child.get(key)
            result_item = None
            if matched:
                span, index = matched
                child["delegation_call_id"] = span["attrs"].get("agent.tool.call_id")
                try:
                    raw_result = tool_results.get(child["delegation_call_id"])
                    if raw_result is None and not span["attrs"].get(
                        "agent.tool.result_truncated"
                    ):
                        raw_result = span["attrs"].get(
                            "agent.tool.result_preview"
                        )
                    payload = json.loads(raw_result)
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    result_item = payload["results"][index]
                    child["delegation_reported_duration_seconds"] = result_item.get(
                        "duration_seconds"
                    )
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            if api_calls == 0 and input_tokens == 0 and output_tokens == 0:
                child["status"] = "failed_start"
                child["execution_status"] = "no_model_work"
                if isinstance(result_item, dict):
                    child["failure_reason"] = result_item.get("summary")
                failed.append(key)
                continue
            if not matched:
                unknown.append(key)
                continue
            span, _index = matched
            message = connection.execute(
                "SELECT timestamp, finish_reason, length(content), tool_calls "
                "FROM messages WHERE session_id=? AND role='assistant' "
                "ORDER BY id DESC LIMIT 1",
                (chain[-1],),
            ).fetchone()
            if not message or message[1] != "stop" or not message[2] or message[3] not in (None, "", "[]"):
                unknown.append(key)
                continue
            start_ns = child.get("start_timestamp_ns")
            end_ns = round(message[0] * 1_000_000_000)
            if not isinstance(start_ns, int) or not (
                span["start_unix_nano"] <= start_ns <= end_ns
                <= span["end_unix_nano"]
            ):
                unknown.append(key)
                continue
            child.update({
                "status": "completed",
                "execution_status": "model_work_completed",
                "terminal_timestamp_ns": end_ns,
                "terminal_timestamp_source": "hermes_state.messages.final_assistant.timestamp",
                "terminal_timestamp_quality": "durable_final_assistant_stop_bounded_by_delegate_span",
                "runtime_seconds": (end_ns - start_ns) / 1_000_000_000,
                "runtime_complete": True,
                "runtime_censored": False,
                "delegation_span_end_timestamp_ns": span["end_unix_nano"],
            })
            reported = child.get("delegation_reported_duration_seconds")
            if isinstance(reported, (int, float)):
                discrepancy = child["runtime_seconds"] - reported
                child["duration_crosscheck_discrepancy_seconds"] = discrepancy
                duration_discrepancies.append(discrepancy)
            completed.append((start_ns, end_ns))
    except sqlite3.Error:
        return original
    finally:
        connection.close()

    points = sorted(
        [(start, 1) for start, _end in completed]
        + [(end, -1) for _start, end in completed]
    )
    active = maximum = 0
    union = overlap = excess = 0.0
    for index, (timestamp_ns, delta) in enumerate(points):
        active += delta
        maximum = max(maximum, active)
        if index + 1 == len(points):
            continue
        duration = (points[index + 1][0] - timestamp_ns) / 1_000_000_000
        if active:
            union += duration
        if active > 1:
            overlap += duration
            excess += (active - 1) * duration

    complete_for_work = not unknown
    result.update({
        "source": "hermes_state/state.db + spans.jsonl (report-time recovery)",
        "child_runtime_by_session": recovered,
        "effective_child_session_count": len(completed),
        "failed_child_start_count": len(failed),
        "failed_child_session_ids": failed,
        "unresolved_child_session_ids": unknown,
        "child_duration_crosscheck_count": len(duration_discrepancies),
        "child_duration_crosscheck_max_abs_discrepancy_seconds": (
            max(map(abs, duration_discrepancies))
            if duration_discrepancies else None
        ),
        "child_runtime_complete_count": len(completed),
        "child_runtime_unknown_count": len(children) - len(completed),
        "child_terminal_identity_qualified_count": len(completed),
        "child_derived_metrics_quality": (
            "recovered_effective_children;failed_starts_excluded"
            if complete_for_work else "partial_recovery"
        ),
        "max_concurrent_children": maximum if complete_for_work else None,
        "max_concurrent_children_lower_bound": maximum,
        "child_sum_runtime_seconds": (
            sum((end - start) / 1_000_000_000 for start, end in completed)
            if complete_for_work else None
        ),
        "child_sum_runtime_lower_bound_seconds": sum(
            (end - start) / 1_000_000_000 for start, end in completed
        ),
        "child_union_active_seconds": union if complete_for_work else None,
        "child_union_active_lower_bound_seconds": union,
        "child_overlap_wall_seconds": overlap if complete_for_work else None,
        "child_concurrency_excess_seconds": excess if complete_for_work else None,
    })
    return result


def _hermes_compaction_count(record) -> int | None:
    """Count durable Hermes session compressions, not OpenClaw trace events."""
    if record.manifest.get("agent_runtime_mode") != "hermes":
        return None
    path = record.run_dir / "hermes_sessions.json"
    try:
        document = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    sessions = document.get("sessions") if isinstance(document, dict) else None
    if not isinstance(sessions, list):
        return None
    return sum(
        isinstance(session, dict) and session.get("end_reason") == "compression"
        for session in sessions
    )


def install_comparison(base) -> None:
    """Enable Hermes metrics only in the Hermes comparison process."""
    if getattr(base, "_hermes_telemetry_installed", False):
        return
    base._hermes_telemetry_installed = True
    original_runtime = base._runtime_observability
    original_record = base.record_to_dict
    original_summarize = base.summarize_arm
    original_compare = base.compare_paired
    original_render = base.render_markdown

    def runtime_observability(record):
        cached = record.__dict__.get("_hermes_recovered_observability_cache")
        if cached is not None:
            return cached
        observation, path = original_runtime(record)
        if path is not None and record.manifest.get("agent_runtime_mode") == "hermes":
            observation = _recover_hermes_child_timing(record, observation)
        result = (observation, path)
        record.__dict__["_hermes_recovered_observability_cache"] = result
        return result

    def record_to_dict(record):
        row = original_record(record)
        if record.manifest.get("agent_runtime_mode") != "hermes":
            return row
        obs, _path = runtime_observability(record)
        row.update({
            "compaction_count": _hermes_compaction_count(record),
            "compaction_count_source": "hermes_sessions.json",
            "effective_subagent_count": obs.get("effective_child_session_count"),
            "failed_subagent_start_count": obs.get("failed_child_start_count"),
            "failed_subagent_session_ids": obs.get("failed_child_session_ids"),
            "unresolved_subagent_session_ids": obs.get("unresolved_child_session_ids"),
            "child_timing_source": obs.get("source"),
            "child_duration_crosscheck_count": obs.get(
                "child_duration_crosscheck_count"
            ),
            "child_duration_crosscheck_max_abs_discrepancy_seconds": obs.get(
                "child_duration_crosscheck_max_abs_discrepancy_seconds"
            ),
            "child_runtime_complete_count": obs.get("child_runtime_complete_count"),
            "child_runtime_unknown_count": obs.get("child_runtime_unknown_count"),
            "child_derived_metrics_quality": obs.get("child_derived_metrics_quality"),
            "child_terminal_identity_qualified_count": obs.get(
                "child_terminal_identity_qualified_count"
            ),
            "max_concurrent_subagents": obs.get("max_concurrent_children"),
            "max_concurrent_subagents_lower_bound": obs.get(
                "max_concurrent_children_lower_bound"
            ),
            "child_sum_runtime_seconds": obs.get("child_sum_runtime_seconds"),
            "child_sum_runtime_lower_bound_seconds": obs.get(
                "child_sum_runtime_lower_bound_seconds"
            ),
            "child_union_active_seconds": obs.get("child_union_active_seconds"),
            "child_union_active_lower_bound_seconds": obs.get(
                "child_union_active_lower_bound_seconds"
            ),
            "child_overlap_wall_seconds": obs.get("child_overlap_wall_seconds"),
            "child_concurrency_excess_seconds": obs.get(
                "child_concurrency_excess_seconds"
            ),
            "child_intervals": base._child_intervals(record),
            "failed_subagent_details": [
                {
                    "session_id": session_id,
                    "delegation_call_id": (
                        obs.get("child_runtime_by_session", {})
                        .get(session_id, {}).get("delegation_call_id")
                    ),
                    "reason": (
                        obs.get("child_runtime_by_session", {})
                        .get(session_id, {}).get("failure_reason")
                    ),
                }
                for session_id in (obs.get("failed_child_session_ids") or [])
            ],
        })
        return row

    def summarize_arm(records):
        summary = original_summarize(records)
        if not records or not all(
            record.manifest.get("agent_runtime_mode") == "hermes"
            for record in records
        ):
            return summary
        rows = [record_to_dict(record) for record in records]
        compactions = [row.get("compaction_count") for row in rows
                       if row.get("benchmark_valid")]
        summary["efficiency"]["total_compactions"] = (
            sum(compactions) if compactions
            and all(value is not None for value in compactions) else None
        )
        summary["efficiency"]["total_spawns"] = None
        activity = summary["subagent_activity"]
        activity["tasks_using_subagents"] = sum(
            (row.get("effective_subagent_count") or 0) > 0 for row in rows
        )
        activity["mean_effective_subagents_per_task"] = base._impl._mean([
            row["effective_subagent_count"] for row in rows
            if row.get("effective_subagent_count") is not None
        ])
        activity["failed_subagent_starts"] = sum(
            row.get("failed_subagent_start_count") or 0 for row in rows
        )
        activity["max_concurrent_subagents_observed"] = max(
            (row["max_concurrent_subagents"] for row in rows
             if row.get("max_concurrent_subagents") is not None),
            default=None,
        )
        return summary

    def compare_paired(subagent, single_agent, *, bootstrap_samples):
        summary, rows = original_compare(
            subagent, single_agent, bootstrap_samples=bootstrap_samples
        )
        if not subagent or not all(
            record.manifest.get("agent_runtime_mode") == "hermes"
            for record in subagent
        ):
            return summary, rows
        sub_by_key = {record.key: record for record in subagent}
        for row in rows:
            record = sub_by_key[(row["instance_id"], row["seed"])]
            flat = record_to_dict(record)
            for field in ("effective_subagent_count", "failed_subagent_start_count"):
                row[f"subagent_{field}"] = flat.get(field)
        return summary, rows

    def render_markdown(*args, **kwargs):
        markdown = original_render(*args, **kwargs)
        records = args[0]
        if not records or not all(
            record.manifest.get("agent_runtime_mode") == "hermes"
            for record in records
        ):
            return markdown
        markdown = markdown.replace(
            "Subagent usage means materialized child sessions from "
            "`runtime_observability.json`, not `sessions_spawn` call count.",
            "Hermes reports materialized child sessions separately from children "
            "that made model calls and returned a final assistant reply. "
            "Child timing is recovered from durable messages and matching "
            "`delegate_task` spans.",
            1,
        )
        rows = [record_to_dict(record) for record in records]
        audit = [
            "## Hermes child execution audit", "",
            f"- Tasks with an executing child: "
            f"**{sum((row.get('effective_subagent_count') or 0) > 0 for row in rows)}"
            f"/{len(rows)}**.",
            f"- Materialized children: "
            f"**{sum(row.get('materialized_subagent_count') or 0 for row in rows)}**; "
            f"executing children: "
            f"**{sum(row.get('effective_subagent_count') or 0 for row in rows)}**; "
            f"failed starts: "
            f"**{sum(row.get('failed_subagent_start_count') or 0 for row in rows)}**.",
            "", "| Task | Materialized | Executing | Failed starts |",
            "|---|---:|---:|---:|",
        ]
        for row in rows:
            audit.append(
                f"| {row['instance_id']} | "
                f"{base._fmt(row.get('materialized_subagent_count'), 0)} | "
                f"{base._fmt(row.get('effective_subagent_count'), 0)} | "
                f"{base._fmt(row.get('failed_subagent_start_count'), 0)} |"
            )
        return markdown.replace(
            "## Input PBS jobs", "\n".join(audit) + "\n\n## Input PBS jobs", 1
        )

    base._runtime_observability = runtime_observability
    base.record_to_dict = base._impl.record_to_dict = record_to_dict
    base.summarize_arm = base._impl.summarize_arm = summarize_arm
    base.compare_paired = base._impl.compare_paired = compare_paired
    base.render_markdown = base._impl.render_markdown = render_markdown


def install_analysis(analysis) -> None:
    """Add Hermes-only audit fields to the single-arm report."""
    if getattr(analysis, "_hermes_telemetry_installed", False):
        return
    analysis._hermes_telemetry_installed = True
    install_comparison(analysis.base)
    original_details = analysis._single_arm_details
    original_render = analysis._render_markdown

    def single_arm_details(rows):
        details = original_details(rows)
        if not rows or not all(
            "effective_subagent_count" in row for row in rows
        ):
            return details
        observed = [row for row in rows
                    if row.get("effective_subagent_count") is not None]
        used = [row for row in observed if row["effective_subagent_count"] > 0]
        details.update({
            "tasks_using_subagents": len(used),
            "subagent_usage_rate": len(used) / len(observed) if observed else None,
            "tasks_without_subagents_ids": [
                row["instance_id"] for row in observed
                if row["effective_subagent_count"] == 0
            ],
            "total_effective_subagents": sum(
                row["effective_subagent_count"] for row in observed
            ),
            "total_failed_subagent_starts": sum(
                row.get("failed_subagent_start_count") or 0 for row in rows
            ),
            "tasks_with_failed_subagent_starts_ids": [
                row["instance_id"] for row in rows
                if (row.get("failed_subagent_start_count") or 0) > 0
            ],
            "max_concurrent_subagents_observed": max(
                (row["max_concurrent_subagents"] for row in rows
                 if row.get("max_concurrent_subagents") is not None),
                default=None,
            ),
            "total_compactions": (
                sum(row["compaction_count"] for row in rows)
                if all(row.get("compaction_count") is not None for row in rows)
                else None
            ),
            "subagent_count_policy": "executing_child_sessions_with_model_calls_and_final_reply;materialized_sessions_separate",
        })
        return details

    def render_markdown(records, summary, details, rows, **kwargs):
        if not records or not all(
            record.manifest.get("agent_runtime_mode") == "hermes"
            for record in records
        ):
            return original_render(records, summary, details, rows, **kwargs)
        safe = dict(details)
        if safe.get("max_concurrent_subagents_observed") is None:
            safe["max_concurrent_subagents_observed"] = "NA"
        if safe.get("total_compactions") is None:
            safe["total_compactions"] = "NA"
        markdown = original_render(records, summary, safe, rows, **kwargs)
        markdown = markdown.replace(
            "Actual subagent usage is the number of materialized child sessions from "
            "`runtime_observability.json`, not the number of `sessions_spawn` calls.",
            "Hermes executing subagents made model calls and returned a final "
            "assistant reply. Materialized sessions and failed starts are separate. "
            "Child timing comes from durable final messages bounded by "
            "`delegate_task` spans.",
            1,
        ).replace(
            "tasks that actually materialized >=1 subagent",
            "tasks with >=1 executing subagent",
            1,
        ).replace(
            "tasks with zero materialized subagents",
            "tasks with zero executing subagents",
            1,
        )
        warning = (
            f"Across {len(rows)} scored tasks, "
            f"{details['tasks_using_subagents']} had an executing child. "
            "The overall reward and latency summaries include every scored "
            "task; a strict 20-task subagent cohort requires rerunning the "
            "task with no child model work."
        )
        markdown = markdown.replace(
            "## Summary", warning + "\n\n## Summary", 1
        )
        audit = [
            "## Hermes child execution audit", "",
            f"- Executing children: **{details['total_effective_subagents']}**; "
            f"materialized children: **{details['total_materialized_subagents']}**; "
            f"failed starts: **{details['total_failed_subagent_starts']}**.",
            "- Failed-start tasks: "
            + (", ".join(details["tasks_with_failed_subagent_starts_ids"])
               or "none"),
            "- Recovered child intervals use final assistant stop timestamps. "
            "The raw `runtime_observability.json` lacks those ends; this report "
            "adds them from the included session database and tool spans.",
            "", "| Task | Materialized | Executing | Failed starts | "
            "Child runtime (s) | Max concurrent |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            audit.append(
                f"| {row['instance_id']} | "
                f"{analysis.base._fmt(row.get('materialized_subagent_count'), 0)} | "
                f"{analysis.base._fmt(row.get('effective_subagent_count'), 0)} | "
                f"{analysis.base._fmt(row.get('failed_subagent_start_count'), 0)} | "
                f"{analysis.base._fmt(row.get('child_sum_runtime_seconds'), 1)} | "
                f"{analysis.base._fmt(row.get('max_concurrent_subagents'), 0)} |"
            )
        return markdown.replace(
            "## Per-task results",
            "\n".join(audit) + "\n\n## Per-task results",
            1,
        )

    analysis._single_arm_details = single_arm_details
    analysis._render_markdown = render_markdown
