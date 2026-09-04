#!/usr/bin/env python3
"""Analyze one RoadmapBench PBS series with strict latency/subagent telemetry.

This is the one-arm companion to ``compare_roadmap_runs.py``. It shares the
same policies: rightmost retry precedence, ``agent_execution_seconds``-only
latency, materialized child sessions as actual subagent usage, and no
model-request-count reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from . import compare_roadmap_runs as base
except ImportError:
    import compare_roadmap_runs as base  # type: ignore


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one RoadmapBench PBS series."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "PBS job ids or concrete PBS run directories. For duplicate "
            "task+seed records, the rightmost input wins."
        ),
    )
    parser.add_argument(
        "--arm",
        choices=("subagent", "single-agent"),
        default="subagent",
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="reports/roadmap")
    parser.add_argument("--name")
    parser.add_argument(
        "--strict-trajectory",
        action="store_true",
        help="Exit 2 if any selected non-ERROR task lacks trajectory.jsonl.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.plot_dpi < 72:
        parser.error("--plot-dpi must be at least 72")
    return args


def _default_name(records: Sequence[base.RunRecord], arm: str) -> str:
    jobs = "+".join(base._jobs(records)) or arm
    return base._safe_name(f"roadmap-{arm}-analysis-{jobs}")


def _task_rows(records: Sequence[base.RunRecord]) -> list[dict[str, Any]]:
    return [base.record_to_dict(record) for record in records]


def _single_arm_details(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    observed = [
        row
        for row in rows
        if base._number(row.get("materialized_subagent_count")) is not None
    ]
    using = [
        row
        for row in observed
        if float(row["materialized_subagent_count"]) > 0
    ]
    zero_ids = [
        str(row["instance_id"])
        for row in observed
        if float(row["materialized_subagent_count"]) <= 0
    ]
    missing_ids = [
        str(row["instance_id"])
        for row in rows
        if base._number(row.get("materialized_subagent_count")) is None
    ]
    return {
        "tasks_with_runtime_observability": len(observed),
        "tasks_using_subagents": len(using),
        "subagent_usage_rate": len(using) / len(observed) if observed else None,
        "tasks_without_subagents_ids": zero_ids,
        "tasks_missing_runtime_observability_ids": missing_ids,
        "total_materialized_subagents": sum(
            int(base._number(row.get("materialized_subagent_count")) or 0)
            for row in observed
        ),
        "max_concurrent_subagents_observed": max(
            [
                int(base._number(row.get("max_concurrent_subagents")) or 0)
                for row in observed
            ],
            default=0,
        ),
        "total_yields": sum(
            int(base._number(row.get("yield_count")) or 0) for row in rows
        ),
        "total_resumes": sum(
            int(base._number(row.get("resume_count")) or 0) for row in rows
        ),
        "total_compactions": sum(
            int(base._number(row.get("compaction_count")) or 0) for row in rows
        ),
        "total_context_overflows": sum(
            int(base._number(row.get("context_overflow_count")) or 0)
            for row in rows
        ),
        "total_loop_detections": sum(
            int(base._number(row.get("loop_detected_count")) or 0)
            for row in rows
        ),
    }


def _plot_task_metric(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    field: str,
    title: str,
    xlabel: str,
    dpi: int,
    xlim: tuple[float, float] | None = None,
) -> bool:
    clean: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        value = base._number(row.get(field))
        if value is not None:
            clean.append((row, value))
    if not clean:
        return False

    clean.sort(key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.34 * len(clean) + 1.5)))
    y = list(range(len(clean)))
    values = [value for _, value in clean]
    labels = [str(row["instance_id"]) for row, _ in clean]
    ax.scatter(values, y, marker="o")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_reward_latency(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    path: Path,
    dpi: int,
) -> bool:
    clean: list[tuple[dict[str, Any], float, float]] = []
    for row in rows:
        reward = base._number(row.get("reward"))
        latency = base._number(row.get("latency_seconds"))
        if reward is not None and latency is not None:
            clean.append((row, reward, latency))
    if not clean:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    x = [latency for _, _, latency in clean]
    y = [reward for _, reward, _ in clean]
    ax.scatter(x, y, marker="o")
    for row, reward, latency in clean:
        ax.annotate(
            str(row["instance_id"]),
            (latency, reward),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
            alpha=0.75,
        )
    ax.set_xlabel("agent_execution_seconds")
    ax.set_ylabel("Verifier reward")
    ax.set_title("RoadmapBench reward vs agent execution latency")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _concurrency_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        copied = dict(row)
        copied["subagent_agent_execution_start_ns"] = row.get(
            "agent_execution_start_ns"
        )
        copied["subagent_agent_execution_end_ns"] = row.get(
            "agent_execution_end_ns"
        )
        copied["subagent_child_intervals"] = row.get("child_intervals") or []
        result.append(copied)
    return result


def generate_single_arm_plots(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    basename: str,
    *,
    arm: str,
    dpi: int,
) -> list[Path]:
    plt = base._load_pyplot()
    if plt is None or not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        (
            output_dir / f"{basename}-reward-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="reward",
                title=f"RoadmapBench verifier reward by task ({arm})",
                xlabel="Verifier reward",
                xlim=(-0.03, 1.03),
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-latency-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="latency_seconds",
                title=f"RoadmapBench agent execution latency by task ({arm})",
                xlabel="agent_execution_seconds",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-prompt-tokens-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="total_prompt_tokens",
                title=f"RoadmapBench prompt tokens by task ({arm})",
                xlabel="Prompt tokens",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-materialized-subagents-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="materialized_subagent_count",
                title=f"Actual materialized subagents by task ({arm})",
                xlabel="Materialized subagents",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-max-concurrent-subagents-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="max_concurrent_subagents",
                title=f"Maximum concurrent subagents by task ({arm})",
                xlabel="Max concurrent materialized subagents",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-serving-input-tokens-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="serving_processed_input_tokens",
                title=f"Serving processed input tokens by task ({arm})",
                xlabel="Serving processed input tokens",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-reward-latency.png",
            lambda path: _plot_reward_latency(plt, rows, path, dpi),
        ),
        (
            output_dir / f"{basename}-subagent-concurrency-trajectory.png",
            lambda path: base.generate_subagent_concurrency_plot(
                plt,
                _concurrency_rows(rows),
                path,
                dpi,
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


def _render_markdown(
    records: Sequence[base.RunRecord],
    summary: dict[str, Any],
    details: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    arm: str,
    input_specs: Sequence[str],
    duplicate_count: int,
    figure_names: Sequence[str],
) -> str:
    accuracy = summary["accuracy"]
    latency = summary["latency_seconds"]
    trajectory = summary["trajectory"]
    serving = summary.get("serving") or {}

    lines = [
        f"# RoadmapBench single-arm report: {arm}",
        "",
        "Accuracy uses verifier `reward`. Latency uses "
        "`manifest.agent_execution_seconds` **only**; missing measurements remain "
        "`NA` and are not replaced by legacy `duration_seconds`.",
        "",
        "Actual subagent usage is the number of materialized child sessions from "
        "`runtime_observability.json`, not the number of `sessions_spawn` calls.",
        "",
        "## Summary",
        "",
        "| Tasks | Evaluated | Infra errors | Mean reward | Mean test pass | "
        "Full-pass rate | Median latency (s) | P95 latency (s) | Trajectory "
        "coverage | Mean serving input tokens | Mean serving output tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['task_count']} | {summary['evaluated_count']} | "
        f"{summary['infra_error_count']} | "
        f"{base._fmt(accuracy['mean_reward'])} | "
        f"{base._fmt(accuracy['mean_test_pass_rate'])} | "
        f"{base._fmt(accuracy['full_pass_rate'])} | "
        f"{base._fmt(latency['median'], 1)} | "
        f"{base._fmt(latency['p95'], 1)} | "
        f"{base._fmt(trajectory['coverage_rate'])} | "
        f"{base._fmt(serving.get('mean_processed_input_tokens_per_task'), 0)} | "
        f"{base._fmt(serving.get('mean_generated_output_tokens_per_task'), 0)} |",
        "",
        "## Input PBS jobs",
        "",
        f"- precedence (left -> right): `{' -> '.join(input_specs)}`",
        f"- selected {arm} jobs: `{', '.join(base._jobs(records))}`",
        "",
        "## Trajectory / subagent activity",
        "",
        f"- tasks with runtime observability: "
        f"**{details['tasks_with_runtime_observability']}/{summary['task_count']}**",
        f"- tasks that actually materialized >=1 subagent: "
        f"**{details['tasks_using_subagents']}** "
        f"({base._fmt(details['subagent_usage_rate'])})",
        f"- total materialized subagents across tasks: "
        f"**{details['total_materialized_subagents']}**",
        f"- maximum concurrent subagents observed: "
        f"**{details['max_concurrent_subagents_observed']}**",
        f"- total yield/resume events: "
        f"**{details['total_yields']}/{details['total_resumes']}**",
        f"- total compactions: **{details['total_compactions']}**",
        f"- total context overflows: **{details['total_context_overflows']}**",
        f"- total loop detections: **{details['total_loop_detections']}**",
    ]

    if details["tasks_without_subagents_ids"]:
        lines.append(
            "- tasks with zero materialized subagents: "
            + ", ".join(details["tasks_without_subagents_ids"])
        )
    if details["tasks_missing_runtime_observability_ids"]:
        lines.append(
            "- tasks missing runtime observability: "
            + ", ".join(details["tasks_missing_runtime_observability_ids"])
        )

    if figure_names:
        lines += ["", "## Figures", ""]
        for name in figure_names:
            lines += [f"![{name}]({name})", ""]

    lines += [
        "",
        "## Per-task results",
        "",
        "| Task | Seed | Job | Status | Reward | Test pass | Latency (s) | "
        "Materialized subagents | Max concurrent | Prompt tokens | Tool calls | "
        "Compactions | Overflows | Loops | Trajectory |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance_id']} | {row['seed']} | {row['job_id']} | "
            f"{row['status']} | {base._fmt(base._number(row.get('reward')))} | "
            f"{base._fmt(base._number(row.get('test_pass_rate')))} | "
            f"{base._fmt(base._number(row.get('latency_seconds')), 1)} | "
            f"{base._fmt(base._number(row.get('materialized_subagent_count')), 0)} | "
            f"{base._fmt(base._number(row.get('max_concurrent_subagents')), 0)} | "
            f"{base._fmt(base._number(row.get('total_prompt_tokens')), 0)} | "
            f"{base._fmt(base._number(row.get('tool_call_count')), 0)} | "
            f"{base._fmt(base._number(row.get('compaction_count')), 0)} | "
            f"{base._fmt(base._number(row.get('context_overflow_count')), 0)} | "
            f"{base._fmt(base._number(row.get('loop_detected_count')), 0)} | "
            f"{'yes' if row.get('trajectory_present') else 'NO'} |"
        )

    if duplicate_count:
        lines += [
            "",
            "## Retry / duplicate selection",
            "",
            f"{duplicate_count} earlier manifest(s) were superseded. "
            "The rightmost matching PBS input is authoritative.",
        ]

    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = base.collect_records(
        args.runs,
        arm=args.arm,
        runs_root=runs_root,
    )
    records, duplicates = base.select_best_records(raw_records)
    rows = _task_rows(records)
    summary = base.summarize_arm(records)
    details = _single_arm_details(rows)
    duplicate_count = sum(len(group) - 1 for group in duplicates.values())

    basename = base._safe_name(args.name or _default_name(records, args.arm))
    figure_paths: list[Path] = []
    if not args.no_plots:
        figure_paths = generate_single_arm_plots(
            rows,
            output_dir,
            basename,
            arm=args.arm,
            dpi=args.plot_dpi,
        )
    figure_names = [path.name for path in figure_paths]

    report = {
        "schema_version": 2,
        "mode": "single-arm",
        "arm": args.arm,
        "input_specs": list(args.runs),
        "jobs": base._jobs(records),
        "latency_policy": "agent_execution_seconds_only",
        "subagent_count_policy": "materialized_child_sessions",
        "summary": summary,
        "activity": details,
        "tasks": rows,
        "duplicates": base._duplicates_json(duplicates),
        "figures": figure_names,
        "retry_precedence": "rightmost command-line input wins unconditionally",
    }
    markdown = _render_markdown(
        records,
        summary,
        details,
        rows,
        arm=args.arm,
        input_specs=args.runs,
        duplicate_count=duplicate_count,
        figure_names=figure_names,
    )

    json_path = output_dir / f"{basename}.json"
    md_path = output_dir / f"{basename}.md"
    tasks_csv = output_dir / f"{basename}-tasks.csv"
    trajectories_json = output_dir / f"{basename}-trajectories.json"

    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(markdown)
    base._write_csv(tasks_csv, rows)
    trajectories_json.write_text(
        json.dumps(base._trajectory_index(records), indent=2) + "\n"
    )

    generated = [json_path, md_path, tasks_csv, trajectories_json, *figure_paths]

    missing_trajectory = [
        record
        for record in records
        if record.status != "ERROR" and not record.has_trajectory
    ]

    print(markdown)
    print("Generated:")
    for path in generated:
        print(f"  {path}")

    if missing_trajectory:
        print(
            "WARNING: selected non-ERROR tasks missing trajectory: "
            + ", ".join(record.instance_id for record in missing_trajectory)
        )
        if args.strict_trajectory:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
