#!/usr/bin/env python3
"""RoadmapBench paired comparison with strict agent latency and subagent telemetry.

The heavy artifact-discovery/statistics implementation remains in
``compare_roadmap_runs_impl.py``. This entry point defines the Roadmap baseline
policy used by the NSCC experiments:

* duplicate ``(instance_id, seed)`` records: rightmost CLI input wins;
* latency: ``agent_execution_seconds`` only (no legacy duration fallback);
* subagent usage: materialized child sessions, not spawn tool-call count;
* model-request counts are intentionally omitted from reports and figures;
* runtime child intervals are visualized as a concurrency-over-trajectory plot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

try:
    from . import compare_roadmap_runs_impl as _impl
    from . import roadmap_result_timing as result_timing
except ImportError:
    import compare_roadmap_runs_impl as _impl  # type: ignore
    import roadmap_result_timing as result_timing  # type: ignore


for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


_original_record_to_dict = _impl.record_to_dict
_original_summarize_arm = _impl.summarize_arm
_original_compare_paired = _impl.compare_paired

TRAJECTORY_COMPARE_FIELDS = (
    ("tool_call_count", "Tool calls"),
    ("total_prompt_tokens", "Prompt tokens"),
    ("total_completion_tokens", "Completion tokens"),
    ("trajectory_session_count", "Trajectory sessions"),
    ("materialized_subagent_count", "Materialized subagents"),
    ("max_concurrent_subagents", "Max concurrent subagents"),
    ("child_union_active_seconds", "Child union active (s)"),
    ("child_overlap_wall_seconds", "Child overlap wall (s)"),
    ("child_concurrency_excess_seconds", "Child concurrency excess (s)"),
    ("compaction_count", "Compactions"),
    ("context_overflow_count", "Context overflows"),
    ("loop_detected_count", "Loop detections"),
    ("serving_processed_input_tokens", "Serving input tokens"),
    ("serving_generated_output_tokens", "Serving output tokens"),
)


def _strict_latency_seconds(record) -> float | None:
    return record.agent_execution_seconds


def _strict_latency_source(record) -> str | None:
    return (
        "agent_execution_seconds"
        if record.agent_execution_seconds is not None
        else None
    )


_impl.RunRecord.latency_seconds = property(_strict_latency_seconds)
_impl.RunRecord.latency_source = property(_strict_latency_source)
_impl.TRAJECTORY_COMPARE_FIELDS = TRAJECTORY_COMPARE_FIELDS


def select_best_records(records):
    """Select duplicate task+seed records by CLI order: rightmost wins."""
    grouped = {}
    for record in records:
        grouped.setdefault(record.key, []).append(record)

    selected = []
    duplicates = {}
    for key, group in grouped.items():
        ordered = list(reversed(group))
        selected.append(ordered[0])
        if len(ordered) > 1:
            duplicates[f"{key[0]}::seed={key[1]}"] = ordered
    return sorted(selected, key=lambda record: record.key), duplicates


def _runtime_observability(record) -> tuple[dict[str, Any], Path | None]:
    cached = record.__dict__.get("_roadmap_runtime_observability_cache")
    if cached is not None:
        return cached

    raw = str(record.manifest.get("runtime_observability_path") or "").strip()
    # Keep the measurement source in the same resolved run tree. A copied
    # Scratch run can retain a historical Home path in its source manifest.
    candidates: list[Path] = [record.run_dir / "runtime_observability.json"]
    if raw:
        raw_path = Path(raw).expanduser()
        if not raw_path.is_absolute():
            candidates.append(record.run_dir / raw_path)
        candidates.append(record.run_dir / raw_path.name)
        candidates.append(raw_path)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            result = (value, candidate.resolve())
            record.__dict__["_roadmap_runtime_observability_cache"] = result
            return result

    result = ({}, None)
    record.__dict__["_roadmap_runtime_observability_cache"] = result
    return result


def _child_intervals(record) -> list[dict[str, Any]]:
    """Return exact child lifetimes plus execution-evidence lower bounds."""
    obs, _path = _runtime_observability(record)
    children = obs.get("child_runtime_by_session") or {}
    if not isinstance(children, dict):
        return []

    intervals: list[dict[str, Any]] = []
    for session_key, child in children.items():
        if not isinstance(child, dict):
            continue

        start_ns = _impl._number(child.get("start_timestamp_ns"))
        terminal_ns = _impl._number(child.get("terminal_timestamp_ns"))
        interval_end_ns = _impl._number(child.get("interval_end_timestamp_ns"))
        lower_bound_end_ns = _impl._number(
            child.get("last_observed_timestamp_ns")
        )

        if terminal_ns is not None:
            end_ns = terminal_ns
            end_source = child.get("terminal_timestamp_source")
            end_quality = child.get("terminal_timestamp_quality")
        else:
            end_ns = interval_end_ns or lower_bound_end_ns
            end_source = (
                child.get("interval_end_timestamp_source")
                or child.get("last_observed_timestamp_source")
            )
            end_quality = (
                child.get("interval_end_timestamp_quality")
                if interval_end_ns is not None
                else child.get("last_observed_timestamp_quality")
            )

        if start_ns is None or end_ns is None or end_ns < start_ns:
            continue

        intervals.append(
            {
                "session_key": str(session_key),
                "session_id": str(child.get("session_id") or ""),
                "label": str(child.get("label") or session_key),
                "status": str(child.get("status") or "unknown"),
                "start_timestamp_ns": int(start_ns),
                "end_timestamp_ns": int(end_ns),
                "terminal_timestamp_ns": (
                    int(terminal_ns) if terminal_ns is not None else None
                ),
                "interval_end_timestamp_ns": (
                    int(interval_end_ns) if interval_end_ns is not None else None
                ),
                "runtime_seconds": _impl._number(child.get("runtime_seconds")),
                "runtime_lower_bound_seconds": _impl._number(
                    child.get("runtime_lower_bound_seconds")
                ),
                "runtime_complete": bool(child.get("runtime_complete")),
                "runtime_censored": bool(child.get("runtime_censored")),
                "start_timestamp_source": child.get("start_timestamp_source"),
                "start_timestamp_quality": child.get("start_timestamp_quality"),
                "start_timestamp_conflict": bool(
                    child.get("start_timestamp_conflict")
                ),
                "end_timestamp_source": end_source,
                "end_timestamp_quality": end_quality,
            }
        )

    return sorted(
        intervals,
        key=lambda item: (
            item["start_timestamp_ns"],
            item["end_timestamp_ns"],
            item["session_key"],
        ),
    )


def record_to_dict(record) -> dict[str, Any]:
    """Flatten one run while using materialized child sessions as subagent usage."""
    result = _original_record_to_dict(record)

    for key in (
        "model_request_count",
        "model_request_latency_mean_ms",
        "model_request_latency_p50_ms",
        "model_request_latency_p95_ms",
        "model_request_latency_max_ms",
        "spawn_count",
    ):
        result.pop(key, None)

    obs, obs_path = _runtime_observability(record)
    child_count = _impl._number(
        obs.get("materialized_child_session_count")
        if obs.get("materialized_child_session_count") is not None
        else obs.get("child_session_count")
    )
    max_concurrent = _impl._number(obs.get("max_concurrent_children"))
    phase = record.manifest.get("phase_timing") or {}
    if not isinstance(phase, dict):
        phase = {}

    result.update(
        {
            "failure_phase": record.manifest.get("failure_phase"),
            "verifier_skipped": record.manifest.get("verifier_skipped"),
            "runtime_observability_present": obs_path is not None,
            "runtime_observability_path": str(obs_path) if obs_path else None,
            "materialized_subagent_count": (
                int(child_count) if child_count is not None else None
            ),
            "child_runtime_complete_count": _impl._number(
                obs.get("child_runtime_complete_count")
            ),
            "child_runtime_unknown_count": _impl._number(
                obs.get("child_runtime_unknown_count")
            ),
            "child_count_source_quality": obs.get("child_count_source_quality"),
            "child_derived_metrics_quality": obs.get(
                "child_derived_metrics_quality"
            ),
            "child_start_timestamp_conflict_count": _impl._number(
                obs.get("child_start_timestamp_conflict_count")
            ),
            "child_terminal_identity_qualified_count": _impl._number(
                obs.get("child_terminal_identity_qualified_count")
            ),
            "max_concurrent_subagents": (
                int(max_concurrent) if max_concurrent is not None else None
            ),
            "max_concurrent_subagents_lower_bound": _impl._number(
                obs.get("max_concurrent_children_lower_bound")
            ),
            "child_sum_runtime_seconds": _impl._number(
                obs.get("child_sum_runtime_seconds")
            ),
            "child_sum_runtime_lower_bound_seconds": _impl._number(
                obs.get("child_sum_runtime_lower_bound_seconds")
            ),
            "child_union_active_seconds": _impl._number(
                obs.get("child_union_active_seconds")
            ),
            "child_union_active_lower_bound_seconds": _impl._number(
                obs.get("child_union_active_lower_bound_seconds")
            ),
            "child_overlap_wall_seconds": _impl._number(
                obs.get("child_overlap_wall_seconds")
            ),
            "child_concurrency_excess_seconds": _impl._number(
                obs.get("child_concurrency_excess_seconds")
            ),
            "agent_execution_start_ns": _impl._number(
                phase.get("agent_execution_start_ns")
            ),
            "agent_execution_end_ns": _impl._number(
                phase.get("agent_execution_end_ns")
            ),
            "child_intervals": _child_intervals(record),
            "artifact_links": _artifact_links(record),
        }
    )
    return result


def _artifact_links(record) -> dict[str, str | None]:
    """Return only links confirmed by the completed artifact exporter."""
    exported = record.__dict__.get("_roadmap_artifact_bundle_links")
    if not isinstance(exported, dict):
        exported = {}
    return {
        key: str(exported[key]) if exported.get(key) else None
        for key in (
            "details",
            "spans_timeline",
            "conversation",
            "resources",
            "execution_log",
            "candidate",
            "verdict",
            "job_evidence",
        )
    }


def summarize_arm(records) -> dict[str, Any]:
    summary = _original_summarize_arm(records)
    rows = [record_to_dict(record) for record in records]

    latency = summary["latency_seconds"]
    latency["primary_metric"] = "agent_execution_seconds"
    latency["legacy_fallback"] = None

    efficiency = summary["efficiency"]
    efficiency.pop("mean_model_requests_per_task", None)
    trajectory = summary["trajectory"]
    trajectory.pop("total_model_request_count", None)
    if isinstance(trajectory.get("event_type_counts"), dict):
        trajectory["event_type_counts"].pop("model_request", None)

    def values(field: str) -> list[float]:
        return [
            value
            for row in rows
            if (value := _impl._number(row.get(field))) is not None
        ]

    subagent_counts = values("materialized_subagent_count")
    max_concurrency = values("max_concurrent_subagents")
    serving_input = values("serving_processed_input_tokens")
    serving_output = values("serving_generated_output_tokens")

    summary["subagent_activity"] = {
        "runtime_observability_present_count": sum(
            bool(row.get("runtime_observability_present")) for row in rows
        ),
        "tasks_using_subagents": sum(value > 0 for value in subagent_counts),
        "mean_materialized_subagents_per_task": _impl._mean(subagent_counts),
        "max_materialized_subagents_in_task": (
            max(subagent_counts) if subagent_counts else None
        ),
        "mean_max_concurrent_subagents": _impl._mean(max_concurrency),
        "max_concurrent_subagents_observed": (
            max(max_concurrency) if max_concurrency else None
        ),
        "mean_child_union_active_seconds": _impl._mean(
            values("child_union_active_seconds")
        ),
        "mean_child_overlap_wall_seconds": _impl._mean(
            values("child_overlap_wall_seconds")
        ),
    }
    summary["serving"] = {
        "mean_processed_input_tokens_per_task": _impl._mean(serving_input),
        "mean_generated_output_tokens_per_task": _impl._mean(serving_output),
    }
    return summary


def compare_paired(
    subagent,
    single_agent,
    *,
    bootstrap_samples: int,
):
    summary, rows = _original_compare_paired(
        subagent,
        single_agent,
        bootstrap_samples=bootstrap_samples,
    )
    sub_map = {record.key: record for record in subagent}
    single_map = {record.key: record for record in single_agent}
    for row in rows:
        key = (row["instance_id"], row["seed"])
        record = sub_map[key]
        flat = record_to_dict(record)
        row["subagent_agent_execution_start_ns"] = flat.get(
            "agent_execution_start_ns"
        )
        row["subagent_agent_execution_end_ns"] = flat.get(
            "agent_execution_end_ns"
        )
        row["subagent_child_intervals"] = flat.get("child_intervals") or []
        for field in (
            "child_count_source_quality",
            "child_derived_metrics_quality",
            "child_runtime_unknown_count",
            "child_start_timestamp_conflict_count",
            "child_terminal_identity_qualified_count",
            "child_sum_runtime_lower_bound_seconds",
            "child_union_active_lower_bound_seconds",
            "max_concurrent_subagents_lower_bound",
        ):
            row[f"subagent_{field}"] = flat.get(field)
        row["subagent_artifact_links"] = flat.get("artifact_links") or {}
        single_record = single_map.get(key)
        row["single_agent_artifact_links"] = (
            record_to_dict(single_record).get("artifact_links") or {}
            if single_record is not None
            else {}
        )
    summary["latency"]["metric"] = "agent_execution_seconds"
    summary["latency"]["legacy_fallback"] = None
    return summary, rows


def _arm_markdown(name: str, summary: dict[str, Any]) -> str:
    accuracy = summary["accuracy"]
    latency = summary["latency_seconds"]
    trajectory = summary["trajectory"]
    subagents = summary.get("subagent_activity") or {}
    serving = summary.get("serving") or {}
    return (
        f"| {name} | {summary['task_count']} | {summary['evaluated_count']} | "
        f"{summary['infra_error_count']} | {_impl._fmt(accuracy['mean_reward'])} | "
        f"{_impl._fmt(accuracy['mean_test_pass_rate'])} | "
        f"{_impl._fmt(accuracy['full_pass_rate'])} | "
        f"{_impl._fmt(latency['median'], 1)} | {_impl._fmt(latency['p95'], 1)} | "
        f"{_impl._fmt(trajectory['coverage_rate'])} | "
        f"{_impl._fmt(subagents.get('mean_materialized_subagents_per_task'), 1)} | "
        f"{_impl._fmt(subagents.get('max_concurrent_subagents_observed'), 0)} | "
        f"{_impl._fmt(serving.get('mean_processed_input_tokens_per_task'), 0)} | "
        f"{_impl._fmt(serving.get('mean_generated_output_tokens_per_task'), 0)} |"
    )


def render_markdown(
    subagent,
    sub_summary,
    *,
    single_agent=None,
    single_summary=None,
    paired_summary=None,
    paired_rows=None,
    duplicate_count: int = 0,
    figure_names: Sequence[str] = (),
    selection_diagnostics: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# RoadmapBench paired accuracy / latency / trajectory report",
        "",
        "Accuracy uses verifier `reward`. Latency uses "
        "`manifest.agent_execution_seconds` **only**; records without that "
        "measurement are excluded from latency statistics rather than falling "
        "back to `duration_seconds`.",
        "",
        "Subagent usage means materialized child sessions from "
        "`runtime_observability.json`, not `sessions_spawn` call count.",
        "",
        "## Arm-level summary",
        "",
        "| Arm | Tasks | Evaluated | Infra errors | Mean reward | Mean test pass | "
        "Full-pass rate | Median latency (s) | P95 (s) | Trajectory coverage | "
        "Mean materialized subagents | Max concurrent subagents | Mean serving "
        "input tokens | Mean serving output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _arm_markdown("subagent", sub_summary),
    ]
    if single_summary is not None:
        lines.append(_arm_markdown("single-agent", single_summary))

    lines += [
        "",
        "## Input PBS jobs",
        "",
        f"- subagent: `{', '.join(_impl._jobs(subagent))}`",
    ]
    if single_agent is not None:
        lines.append(
            f"- single-agent: `{', '.join(_impl._jobs(single_agent))}`"
        )

    if paired_summary is not None:
        accuracy = paired_summary["accuracy"]
        latency = paired_summary["latency"]
        trajectory = paired_summary["trajectory"]
        lines += [
            "",
            "## Paired primary outcomes",
            "",
            f"- paired tasks: **{paired_summary['paired_task_count']}**",
            f"- paired seeds: **{paired_summary['paired_seeds']}**",
            f"- reward-evaluable pairs: **{accuracy['paired_evaluable_count']}**; "
            f"latency-evaluable pairs with `agent_execution_seconds` on both arms: "
            f"**{latency['paired_evaluable_count']}**",
            "- mean reward delta (subagent - single-agent): "
            f"**{_impl._fmt(accuracy['mean_reward_delta_subagent_minus_single'])}** "
            f"(95% task-bootstrap CI "
            f"{_impl._fmt_ci(accuracy['mean_reward_delta_95pct_bootstrap_ci'])})",
            "- accuracy W/T/L for subagent: "
            f"**{accuracy['subagent_wins']}/{accuracy['ties']}/"
            f"{accuracy['subagent_losses']}**; exact paired sign-test "
            f"p={_impl._fmt(accuracy['exact_sign_test_two_sided_p'])}",
            "- mean latency delta (single-agent - subagent): "
            f"**{_impl._fmt(latency['mean_delta_single_minus_subagent_seconds'], 1)} s** "
            f"(95% task-bootstrap CI "
            f"{_impl._fmt_ci(latency['mean_delta_95pct_bootstrap_ci_seconds'], 1)})",
            "- geometric-mean speedup (single-agent / subagent): "
            f"**{_impl._fmt(latency['geomean_speedup_single_over_subagent'])}x** "
            f"(95% task-bootstrap CI "
            f"{_impl._fmt_ci(latency['geomean_speedup_95pct_bootstrap_ci'])})",
            "- latency faster/tie/slower for subagent: "
            f"**{latency['subagent_faster']}/{latency['ties']}/"
            f"{latency['subagent_slower']}**",
            "- both trajectories present: "
            f"**{trajectory['both_present_count']}/"
            f"{paired_summary['paired_task_count']}**",
            "",
            "Positive reward delta favors subagent; latency delta > 0 and "
            "speedup > 1 mean subagent is faster.",
            "",
            "## Trajectory / serving efficiency",
            "",
            *_impl._efficiency_markdown(paired_summary),
        ]

        if figure_names:
            lines += ["", "## Figures", ""]
            for name in figure_names:
                lines += [f"![{name}]({name})", ""]

        if paired_rows:
            lines += [
                "",
                "## Per-task paired results",
                "",
                "| Task | Seed | Sub reward | Single reward | Δ reward | Sub latency "
                "(s) | Single latency (s) | Speedup | Materialized subagents | "
                "Max concurrent subagents | Child timing quality | Trajectory | Artifacts |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
            for row in paired_rows:
                trajectory_ok = (
                    "yes"
                    if row["subagent_trajectory"]
                    and row["single_agent_trajectory"]
                    else "NO"
                )
                lines.append(
                    f"| {row['instance_id']} | {row['seed']} | "
                    f"{_impl._fmt(row['subagent_reward'])} | "
                    f"{_impl._fmt(row['single_agent_reward'])} | "
                    f"{_impl._fmt(row['reward_delta_subagent_minus_single'])} | "
                    f"{_impl._fmt(row['subagent_latency_seconds'], 1)} | "
                    f"{_impl._fmt(row['single_agent_latency_seconds'], 1)} | "
                    f"{_impl._fmt(row['subagent_speedup_single_over_subagent'])}x | "
                    f"{_impl._fmt(row.get('subagent_materialized_subagent_count'), 0)} | "
                    f"{_impl._fmt(row.get('subagent_max_concurrent_subagents'), 0)} | "
                    f"{row.get('subagent_child_derived_metrics_quality') or 'unknown'} | "
                    f"{trajectory_ok} | "
                    f"{_paired_artifact_markdown(row)} |"
                )

        if paired_summary["subagent_only"] or paired_summary["single_agent_only"]:
            lines += ["", "## Unpaired tasks", ""]
            if paired_summary["subagent_only"]:
                lines.append(
                    "- subagent only: "
                    + ", ".join(
                        item["instance_id"]
                        for item in paired_summary["subagent_only"]
                    )
                )
            if paired_summary["single_agent_only"]:
                lines.append(
                    "- single-agent only: "
                    + ", ".join(
                        item["instance_id"]
                        for item in paired_summary["single_agent_only"]
                    )
                )

    if selection_diagnostics:
        reasons = selection_diagnostics.get("excluded_reason_counts") or {}
        lines += [
            "",
            "## Selection coverage",
            "",
            f"- candidate task+seed keys: **{selection_diagnostics.get('candidate_key_count')}**",
            f"- included pairs: **{selection_diagnostics.get('included_pair_count')}**",
            f"- excluded pairs: **{selection_diagnostics.get('excluded_pair_count')}**",
        ]
        if reasons:
            lines.append(
                "- exclusion reasons: "
                + ", ".join(f"`{key}`={value}" for key, value in sorted(reasons.items()))
            )

    selected_records = [*subagent, *(single_agent or [])]
    if selected_records:
        lines += [
            "",
            "## Selected run artifacts",
            "",
            "| Task | Seed | Arm | Job | Portable artifacts |",
            "|---|---:|---|---|---|",
        ]
        for record in sorted(
            selected_records,
            key=lambda item: (item.instance_id, item.seed, item.arm),
        ):
            links = record_to_dict(record).get("artifact_links") or {}
            lines.append(
                f"| {record.instance_id} | {record.seed} | {record.arm} | "
                f"{record.job_id} | {_artifact_markdown(links, 'view')} |"
            )

    if duplicate_count:
        lines += [
            "",
            "## Retry / duplicate selection",
            "",
            f"{duplicate_count} earlier manifest(s) were superseded. "
            "For each `instance_id + seed`, the rightmost matching command-line "
            "PBS input is authoritative regardless of reward/status.",
        ]

    lines.append("")
    return "\n".join(lines)


def _artifact_markdown(links: dict[str, Any], label: str) -> str:
    parts = []
    for short, key in (
        ("details", "details"),
        ("spans", "spans_timeline"),
        ("chat", "conversation"),
        ("util", "resources"),
        ("logs", "execution_log"),
        ("candidate", "candidate"),
        ("verdict", "verdict"),
        ("job", "job_evidence"),
    ):
        target = links.get(key)
        if target:
            parts.append(f"[{short}]({target})")
    return f"{label}: " + ("/".join(parts) if parts else "not collected")


def _paired_artifact_markdown(row: dict[str, Any]) -> str:
    return "<br>".join(
        (
            _artifact_markdown(
                row.get("subagent_artifact_links") or {}, "subagent"
            ),
            _artifact_markdown(
                row.get("single_agent_artifact_links") or {}, "single"
            ),
        )
    )


def _plot_subagent_count(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    path: Path,
    dpi: int,
) -> bool:
    clean = [
        row
        for row in rows
        if _impl._number(row.get("subagent_materialized_subagent_count"))
        is not None
    ]
    if not clean:
        return False
    clean = sorted(
        clean,
        key=lambda row: float(
            _impl._number(row.get("subagent_materialized_subagent_count")) or 0
        ),
    )
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.34 * len(clean) + 1.5)))
    y = list(range(len(clean)))
    values = [
        float(row.get("subagent_materialized_subagent_count") or 0)
        for row in clean
    ]
    ax.scatter(values, y, marker="o")
    ax.set_yticks(y, [row["instance_id"] for row in clean], fontsize=8)
    ax.set_xlabel("Materialized subagents")
    ax.set_title("Actual subagent usage by task")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_subagent_concurrency_trajectory(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    path: Path,
    dpi: int,
) -> bool:
    """Heatmap of exact child concurrency over normalized agent execution."""
    bins = 120
    prepared: list[tuple[str, list[int]]] = []
    maximum = 0

    for row in sorted(rows, key=lambda item: str(item["instance_id"])):
        start = _impl._number(row.get("subagent_agent_execution_start_ns"))
        end = _impl._number(row.get("subagent_agent_execution_end_ns"))
        intervals = [
            interval
            for interval in (row.get("subagent_child_intervals") or [])
            if isinstance(interval, dict)
            and not bool(interval.get("runtime_censored"))
        ]
        if start is None or end is None or end <= start:
            continue

        counts = [0] * bins
        span = end - start
        for interval in intervals:
            if not isinstance(interval, dict):
                continue
            child_start = _impl._number(interval.get("start_timestamp_ns"))
            child_end = _impl._number(
                interval.get("end_timestamp_ns")
                if interval.get("end_timestamp_ns") is not None
                else interval.get("terminal_timestamp_ns")
            )
            if (
                child_start is None
                or child_end is None
                or child_end < child_start
            ):
                continue
            clipped_start = max(start, child_start)
            clipped_end = min(end, child_end)
            if clipped_end < clipped_start:
                continue
            for index in range(bins):
                midpoint = start + span * ((index + 0.5) / bins)
                if clipped_start <= midpoint <= clipped_end:
                    counts[index] += 1

        maximum = max(maximum, max(counts, default=0))
        prepared.append((str(row["instance_id"]), counts))

    if not prepared or maximum <= 0:
        return False

    matrix = [counts for _label, counts in prepared]
    labels = [label for label, _counts in prepared]
    fig, ax = plt.subplots(
        figsize=(11, max(5.0, 0.36 * len(prepared) + 1.8))
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=max(1, maximum),
    )
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xticks(
        [0, bins * 0.25, bins * 0.50, bins * 0.75, bins - 1],
        ["0%", "25%", "50%", "75%", "100%"],
    )
    ax.set_xlabel("Normalized agent execution trajectory")
    ax.set_title("Exact child concurrency over the agent trajectory")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Active materialized subagents")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True



def _plot_agent_subagent_timeline(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    path: Path,
    dpi: int,
) -> bool:
    """Plot parent and materialized-child lifetimes as horizontal timelines."""
    tracks: list[dict[str, Any]] = []
    child_track_count = 0

    for row in sorted(rows, key=lambda item: str(item["instance_id"])):
        instance_id = str(row["instance_id"])
        parent_start = _impl._number(
            row.get("subagent_agent_execution_start_ns")
        )
        parent_end = _impl._number(
            row.get("subagent_agent_execution_end_ns")
        )
        if (
            parent_start is None
            or parent_end is None
            or parent_end <= parent_start
        ):
            continue

        parent_seconds = (parent_end - parent_start) / 1e9
        tracks.append(
            {
                "label": f"{instance_id} / Parent",
                "start": 0.0,
                "end": parent_seconds,
                "censored": False,
                "parent": True,
            }
        )

        intervals = row.get("subagent_child_intervals") or []
        for index, interval in enumerate(intervals, start=1):
            if not isinstance(interval, dict):
                continue

            child_start = _impl._number(interval.get("start_timestamp_ns"))
            child_end = _impl._number(
                interval.get("end_timestamp_ns")
                if interval.get("end_timestamp_ns") is not None
                else interval.get("terminal_timestamp_ns")
            )
            if (
                child_start is None
                or child_end is None
                or child_end < child_start
            ):
                continue

            start_seconds = (child_start - parent_start) / 1e9
            end_seconds = (child_end - parent_start) / 1e9
            censored = bool(interval.get("runtime_censored"))
            child_label = str(
                interval.get("label")
                or interval.get("session_key")
                or f"Child {index}"
            )
            suffix = " [censored]" if censored else ""

            tracks.append(
                {
                    "label": (
                        f"{instance_id} / Child {index}: "
                        f"{child_label}{suffix}"
                    ),
                    "start": start_seconds,
                    "end": end_seconds,
                    "censored": censored,
                    "parent": False,
                }
            )
            child_track_count += 1

    if not tracks or child_track_count == 0:
        return False

    fig, ax = plt.subplots(
        figsize=(12, max(5.0, 0.58 * len(tracks) + 1.8))
    )

    for y, track in enumerate(tracks):
        start = float(track["start"])
        end = float(track["end"])
        censored = bool(track["censored"])
        is_parent = bool(track["parent"])

        ax.hlines(
            y,
            start,
            end,
            linewidth=6 if is_parent else 4,
            linestyles="--" if censored else "-",
        )
        ax.scatter([start], [y], marker="|", s=90)

        if censored:
            ax.scatter([end], [y], marker="x", s=45)
        else:
            ax.scatter([end], [y], marker="|", s=90)

        duration = end - start
        ax.annotate(
            f"{duration:.1f}s",
            (end, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )

    ax.set_yticks(
        range(len(tracks)),
        [str(track["label"]) for track in tracks],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.axvline(0.0, linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Seconds since parent agent start")
    ax.set_title("Parent and materialized-subagent execution timeline")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_plots(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    basename: str,
    *,
    dpi: int,
) -> list[Path]:
    """Generate paired outcome plots plus actual subagent-concurrency plots."""
    plt = _impl._load_pyplot()
    if plt is None or not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        (
            output_dir / f"{basename}-accuracy-paired.png",
            lambda path: _impl._plot_accuracy(plt, rows, path, dpi),
        ),
        (
            output_dir / f"{basename}-latency-speedup.png",
            lambda path: _impl._plot_ratio_by_task(
                plt,
                rows,
                path,
                value_field="subagent_speedup_single_over_subagent",
                title="RoadmapBench agent-execution latency speedup",
                xlabel=(
                    "single-agent agent_execution_seconds / "
                    "subagent agent_execution_seconds (>1 favors subagent)"
                ),
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-accuracy-latency-tradeoff.png",
            lambda path: _impl._plot_tradeoff(plt, rows, path, dpi),
        ),
        (
            output_dir / f"{basename}-materialized-subagents.png",
            lambda path: _plot_subagent_count(plt, rows, path, dpi),
        ),
        (
            output_dir / f"{basename}-subagent-concurrency-trajectory.png",
            lambda path: _plot_subagent_concurrency_trajectory(
                plt, rows, path, dpi
            ),
        ),
        (
            output_dir / f"{basename}-agent-subagent-timeline.png",
            lambda path: _plot_agent_subagent_timeline(
                plt, rows, path, dpi
            ),
        ),
    ]

    generated: list[Path] = []
    for path, render in candidates:
        try:
            if render(path):
                generated.append(path)
        except Exception as exc:
            print(f"WARNING: plot generation failed for {path.name}: {exc}")
    return generated


_impl.select_best_records = select_best_records
_impl.record_to_dict = record_to_dict
_impl.summarize_arm = summarize_arm
_impl.compare_paired = compare_paired
_impl.render_markdown = render_markdown
_impl.generate_plots = generate_plots

globals().update(
    {
        "TRAJECTORY_COMPARE_FIELDS": TRAJECTORY_COMPARE_FIELDS,
        "select_best_records": select_best_records,
        "record_to_dict": record_to_dict,
        "summarize_arm": summarize_arm,
        "compare_paired": compare_paired,
        "render_markdown": render_markdown,
        "generate_plots": generate_plots,
        "generate_subagent_concurrency_plot": _plot_subagent_concurrency_trajectory,
        "generate_agent_subagent_timeline_plot": _plot_agent_subagent_timeline,
    }
)

def _export_result_timing(records, bundle_dir, basename, *, plots, dpi):
    if records and all(
        record.manifest.get("agent_runtime_mode") in {"hermes", "codex"}
        for record in records
    ):
        return None
    tasks = result_timing.analyze_records(records, _runtime_observability)
    export = result_timing.export_analysis(
        tasks, bundle_dir, basename, plots=plots, dpi=dpi
    )
    return export, result_timing.render_markdown(export)


def main(
    argv: Sequence[str] | None = None,
    *,
    omit_candidate_workspace: bool = False,
) -> int:
    return _impl.main(
        argv, omit_candidate_workspace=omit_candidate_workspace,
        result_timing_exporter=_export_result_timing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
