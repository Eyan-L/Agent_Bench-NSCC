#!/usr/bin/env python3
"""Analyze one RoadmapBench PBS run series and generate single-arm figures.

This is the one-arm companion to ``compare_roadmap_runs.py``. It reuses the
same artifact discovery, manifest/trajectory parsing, arm summary, and retry
selection logic, but it does not require a second comparison arm.

Examples
--------
Analyze the effective subagent series. Later/rightmost PBS inputs override
earlier records with the same ``instance_id + seed``::

    python scripts/analyze_roadmap_runs.py \
      16056107.pbs101 16056108.pbs101 16097006.pbs101 \
      --arm subagent \
      --strict-trajectory

Analyze a single-agent series::

    python scripts/analyze_roadmap_runs.py \
      16104466.pbs101 16104470.pbs101 \
      --arm single-agent

By default outputs are written under ``reports/roadmap``:
  * Markdown summary
  * JSON report
  * per-task CSV
  * trajectory index JSON
  * single-arm PNG figures, when matplotlib is available

Duplicate semantics are intentionally identical to the current compare CLI:
the later/rightmost command-line PBS job or run directory is authoritative,
even if that later record is an ERROR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:  # Package import.
    from . import compare_roadmap_runs as base
except ImportError:  # Direct CLI execution.
    import compare_roadmap_runs as base  # type: ignore


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one RoadmapBench PBS series without requiring a paired "
            "subagent/single-agent arm."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "PBS job ids or concrete PBS run directories. Input order matters: "
            "for duplicate task+seed records, the rightmost input wins."
        ),
    )
    parser.add_argument(
        "--arm",
        choices=("subagent", "single-agent"),
        default="subagent",
        help="Label for this run series (default: subagent).",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root used to resolve PBS job ids (default: runs).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/roadmap",
        help="Directory for JSON/CSV/Markdown/PNG reports.",
    )
    parser.add_argument(
        "--name",
        help="Optional report basename; arm and PBS ids are used by default.",
    )
    parser.add_argument(
        "--strict-trajectory",
        action="store_true",
        help="Exit 2 if any selected non-ERROR task lacks trajectory.jsonl.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate PNG figures.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=180,
        help="PNG figure DPI when plotting is available (default: 180).",
    )
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
    spawned = [
        row
        for row in rows
        if base._number(row.get("spawn_count")) is not None
        and float(row["spawn_count"]) > 0
    ]
    no_spawn = [
        str(row["instance_id"])
        for row in rows
        if (base._number(row.get("spawn_count")) or 0.0) <= 0
    ]
    return {
        "tasks_with_spawn": len(spawned),
        "tasks_without_spawn": len(no_spawn),
        "spawn_usage_rate": len(spawned) / len(rows) if rows else None,
        "tasks_without_spawn_ids": no_spawn,
        "total_spawns": sum(
            int(base._number(row.get("spawn_count")) or 0) for row in rows
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
        duration = base._number(row.get("duration_seconds"))
        if reward is not None and duration is not None:
            clean.append((row, reward, duration))
    if not clean:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    x = [duration for _, _, duration in clean]
    y = [reward for _, reward, _ in clean]
    ax.scatter(x, y, marker="o")
    for row, reward, duration in clean:
        ax.annotate(
            str(row["instance_id"]),
            (duration, reward),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
            alpha=0.75,
        )
    ax.set_xlabel("End-to-end task latency (s)")
    ax.set_ylabel("Verifier reward")
    ax.set_title("RoadmapBench reward vs latency")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_single_arm_plots(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    basename: str,
    *,
    arm: str,
    dpi: int,
) -> list[Path]:
    """Generate useful figures for one run series."""
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
                field="duration_seconds",
                title=f"RoadmapBench end-to-end latency by task ({arm})",
                xlabel="Latency (s)",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-model-requests-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="model_request_count",
                title=f"RoadmapBench model requests by task ({arm})",
                xlabel="Model requests",
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
            output_dir / f"{basename}-spawns-by-task.png",
            lambda path: _plot_task_metric(
                plt,
                rows,
                path,
                field="spawn_count",
                title=f"RoadmapBench sessions_spawn events by task ({arm})",
                xlabel="sessions_spawn events",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-reward-latency.png",
            lambda path: _plot_reward_latency(plt, rows, path, dpi),
        ),
    ]

    generated: list[Path] = []
    for path, render in candidates:
        try:
            if render(path):
                generated.append(path)
        except Exception as exc:  # plotting must not destroy the data report
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
    efficiency = summary["efficiency"]

    lines = [
        f"# RoadmapBench single-arm report: {arm}",
        "",
        "This report analyzes one PBS run series only. Accuracy uses verifier "
        "`reward`; latency uses `manifest.duration_seconds`.",
        "",
        "For duplicate `instance_id + seed` records, command-line input order "
        "is authoritative: the later/rightmost PBS input replaces earlier "
        "records unconditionally.",
        "",
        "## Summary",
        "",
        "| Tasks | Evaluated | Infra errors | Mean reward | Full-pass rate | "
        "Median latency (s) | P95 latency (s) | Trajectory coverage | Mean "
        "model requests | Mean prompt tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['task_count']} | {summary['evaluated_count']} | "
        f"{summary['infra_error_count']} | {base._fmt(accuracy['mean_reward'])} | "
        f"{base._fmt(accuracy['full_pass_rate'])} | "
        f"{base._fmt(latency['median'], 1)} | {base._fmt(latency['p95'], 1)} | "
        f"{base._fmt(trajectory['coverage_rate'])} | "
        f"{base._fmt(efficiency['mean_model_requests_per_task'], 1)} | "
        f"{base._fmt(efficiency['mean_prompt_tokens_per_task'], 0)} |",
        "",
        "## Input PBS jobs",
        "",
        f"- precedence (left -> right): `{' -> '.join(input_specs)}`",
        f"- selected {arm} jobs: `{', '.join(base._jobs(records))}`",
        "",
        "## Trajectory / subagent activity",
        "",
        f"- tasks with `spawn_count > 0`: **{details['tasks_with_spawn']}/"
        f"{summary['task_count']}** "
        f"({base._fmt(details['spawn_usage_rate'])})",
        f"- total spawn events: **{details['total_spawns']}**",
        f"- total yield/resume events: **{details['total_yields']}/"
        f"{details['total_resumes']}**",
        f"- total compactions: **{details['total_compactions']}**",
        f"- total context overflows: **{details['total_context_overflows']}**",
        f"- total loop detections: **{details['total_loop_detections']}**",
    ]
    if details["tasks_without_spawn_ids"]:
        lines += [
            "- tasks with zero spawn events: "
            + ", ".join(details["tasks_without_spawn_ids"])
        ]

    if figure_names:
        lines += ["", "## Figures", ""]
        for name in figure_names:
            lines += [f"![{name}]({name})", ""]

    lines += [
        "",
        "## Per-task results",
        "",
        "| Task | Seed | Job | Status | Reward | Latency (s) | Model requests | "
        "Prompt tokens | Tool calls | Spawns | Sessions | Compactions | "
        "Overflows | Loops | Trajectory |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance_id']} | {row['seed']} | {row['job_id']} | "
            f"{row['status']} | {base._fmt(base._number(row.get('reward')))} | "
            f"{base._fmt(base._number(row.get('duration_seconds')), 1)} | "
            f"{base._fmt(base._number(row.get('model_request_count')), 0)} | "
            f"{base._fmt(base._number(row.get('total_prompt_tokens')), 0)} | "
            f"{base._fmt(base._number(row.get('tool_call_count')), 0)} | "
            f"{base._fmt(base._number(row.get('spawn_count')), 0)} | "
            f"{base._fmt(base._number(row.get('trajectory_session_count')), 0)} | "
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
            f"{duplicate_count} earlier manifest(s) were superseded. The "
            "rightmost matching PBS input is selected regardless of reward, "
            "status, or trajectory quality.",
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
    # Imported from compare_roadmap_runs.py, so this uses the current
    # rightmost-input-wins selection rule rather than the old implementation
    # heuristic preserved in compare_roadmap_runs_impl.py.
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
        "schema_version": 1,
        "mode": "single-arm",
        "arm": args.arm,
        "input_specs": list(args.runs),
        "jobs": base._jobs(records),
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

    if not args.no_plots and not figure_paths:
        print(
            "NOTE: no PNG figures were generated. Install matplotlib or use "
            "--no-plots to silence this note."
        )

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
