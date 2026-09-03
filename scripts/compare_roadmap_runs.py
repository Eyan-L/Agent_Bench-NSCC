#!/usr/bin/env python3
"""Compare RoadmapBench subagent and single-agent PBS runs offline.

This script only reads artifacts already produced by Agent_Bench. It does not
launch a benchmark, model server, verifier, or task.

Examples
--------
Summarize one or more existing subagent PBS jobs::

    python scripts/compare_roadmap_runs.py \
      16056107.pbs101 16056108.pbs101

Compare the formal paired arms (extra jobs may be focused retries/smokes)::

    python scripts/compare_roadmap_runs.py \
      --subagent 16056107.pbs101 16056108.pbs101 1609XXXX.pbs101 \
      --single-agent 1610AAAA.pbs101 1610BBBB.pbs101 \
      --strict-trajectory

Arguments may also be concrete PBS run directories instead of job ids.
Reports are written below reports/roadmap by default and their filenames carry
the PBS job ids used for the analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Sequence


PBS_JOB_RE = re.compile(r"(?P<job>\d+\.pbs\d+)")
CORE_EVENTS = (
    "tool_call",
    "tool_result",
    "spawn",
    "yield",
    "resume",
    "child_completion",
    "model_request",
    "compaction",
    "context_overflow",
    "loop_detected",
    "done",
    "candidate_freeze",
    "cleanup",
)


@dataclass
class RunRecord:
    arm: str
    job_id: str
    pbs_run_dir: Path
    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    trajectory_path: Path | None
    trajectory: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.manifest.get("instance_id") or self.run_dir.name)

    @property
    def seed(self) -> int:
        try:
            return int(self.manifest.get("seed") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def key(self) -> tuple[str, int]:
        return self.instance_id, self.seed

    @property
    def status(self) -> str:
        return str(self.manifest.get("status") or "UNKNOWN").upper()

    @property
    def verifier_status(self) -> str:
        return str(self.manifest.get("verifier_status") or "").upper()

    @property
    def reward(self) -> float | None:
        return _number(self.manifest.get("reward"))

    @property
    def test_pass_rate(self) -> float | None:
        return _number(self.manifest.get("test_pass_rate"))

    @property
    def duration_seconds(self) -> float | None:
        value = _number(self.manifest.get("duration_seconds"))
        if value is not None:
            return value
        start = _number(self.manifest.get("wall_start_ns"))
        end = _number(self.manifest.get("wall_end_ns"))
        if start is not None and end is not None and end >= start:
            return (end - start) / 1e9
        return None

    @property
    def has_trajectory(self) -> bool:
        return bool(self.trajectory_path and self.trajectory_path.is_file())


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_") or "run"


def _job_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = PBS_JOB_RE.search(part)
        if match:
            return match.group("job")
    return "unknown-job"


def resolve_pbs_run(spec: str, runs_root: Path) -> Path:
    """Resolve a PBS job id such as 16056107.pbs101 or an explicit path."""
    explicit = Path(spec).expanduser()
    if explicit.is_dir():
        return explicit.resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")

    exact: list[Path] = []
    fuzzy: list[Path] = []
    for experiment_dir in runs_root.iterdir():
        if not experiment_dir.is_dir():
            continue
        for candidate in experiment_dir.iterdir():
            if not candidate.is_dir():
                continue
            if candidate.name.endswith(f"_{spec}"):
                exact.append(candidate)
            elif spec in candidate.name:
                fuzzy.append(candidate)

    matches = exact or fuzzy
    if not matches:
        raise FileNotFoundError(
            f"could not resolve PBS job {spec!r} below {runs_root}"
        )
    if len(matches) > 1:
        options = "\n".join(f"  - {path}" for path in sorted(matches))
        raise RuntimeError(
            f"PBS job {spec!r} matched multiple run directories:\n{options}"
        )
    return matches[0].resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_trajectory(run_dir: Path, manifest: dict[str, Any]) -> Path | None:
    raw = str(manifest.get("trajectory_path") or "").strip()
    candidates: list[Path] = []
    if raw:
        path = Path(raw).expanduser()
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(run_dir / path)
        candidates.append(run_dir / path.name)
    candidates.append(run_dir / "trajectory.jsonl")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _event_timestamp_ns(event: dict[str, Any]) -> int | None:
    raw = event.get("timestamp_ns")
    try:
        if raw is not None:
            return int(raw)
    except (TypeError, ValueError):
        pass
    timestamp = event.get("timestamp")
    if not timestamp:
        return None
    try:
        return int(
            datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            .timestamp()
            * 1e9
        )
    except ValueError:
        return None


def inspect_trajectory(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "present": False,
            "event_count": 0,
            "event_types": {},
            "timed_event_count": 0,
            "untimed_event_count": 0,
            "span_seconds": None,
            "session_keys": [],
            "model_request_latency_ms": {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            },
        }

    event_types: Counter[str] = Counter()
    session_keys: set[str] = set()
    timestamps: list[int] = []
    request_latencies: list[float] = []
    event_count = 0

    with path.open(errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid trajectory JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                continue
            event_count += 1
            event_type = str(event.get("event_type") or "unknown")
            event_types[event_type] += 1
            session_key = str(event.get("session_key") or "").strip()
            if session_key:
                session_keys.add(session_key)
            timestamp_ns = _event_timestamp_ns(event)
            if timestamp_ns is not None and timestamp_ns > 0:
                timestamps.append(timestamp_ns)
            if event_type == "model_request":
                latency = _number(event.get("request_latency_ms"))
                if latency is not None:
                    request_latencies.append(latency)

    span_seconds = None
    if len(timestamps) >= 2:
        span_seconds = (max(timestamps) - min(timestamps)) / 1e9
    return {
        "present": True,
        "event_count": event_count,
        "event_types": dict(sorted(event_types.items())),
        "timed_event_count": len(timestamps),
        "untimed_event_count": event_count - len(timestamps),
        "span_seconds": span_seconds,
        "session_keys": sorted(session_keys),
        "model_request_latency_ms": {
            "count": len(request_latencies),
            "mean": _mean(request_latencies),
            "p50": _percentile(request_latencies, 0.50),
            "p95": _percentile(request_latencies, 0.95),
            "max": max(request_latencies) if request_latencies else None,
        },
    }


def collect_records(
    specs: Sequence[str], *, arm: str, runs_root: Path
) -> list[RunRecord]:
    records: list[RunRecord] = []
    seen_manifests: set[Path] = set()
    for spec in specs:
        pbs_run_dir = resolve_pbs_run(spec, runs_root)
        job_id = _job_id(pbs_run_dir)
        manifests = sorted(pbs_run_dir.rglob("manifest.json"))
        if not manifests:
            raise FileNotFoundError(
                f"no manifest.json found below {pbs_run_dir}"
            )
        for manifest_path in manifests:
            manifest_path = manifest_path.resolve()
            if manifest_path in seen_manifests:
                continue
            seen_manifests.add(manifest_path)
            manifest = _read_json(manifest_path)
            run_dir = manifest_path.parent
            trajectory_path = _find_trajectory(run_dir, manifest)
            records.append(
                RunRecord(
                    arm=arm,
                    job_id=job_id,
                    pbs_run_dir=pbs_run_dir,
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    trajectory_path=trajectory_path,
                    trajectory=inspect_trajectory(trajectory_path),
                )
            )
    return records


def _selection_score(record: RunRecord) -> tuple[int, int, int, int, int]:
    """Prefer a successful focused retry over an earlier infra ERROR."""
    reward = 1 if record.reward is not None else 0
    non_error = 1 if record.status != "ERROR" else 0
    trajectory = 1 if record.has_trajectory else 0
    wall_end = int(_number(record.manifest.get("wall_end_ns")) or 0)
    try:
        mtime = record.manifest_path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return reward, non_error, trajectory, wall_end, mtime


def select_best_records(
    records: Sequence[RunRecord],
) -> tuple[list[RunRecord], dict[str, list[RunRecord]]]:
    grouped: dict[tuple[str, int], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.key, []).append(record)

    selected: list[RunRecord] = []
    duplicates: dict[str, list[RunRecord]] = {}
    for key, group in grouped.items():
        ordered = sorted(group, key=_selection_score, reverse=True)
        selected.append(ordered[0])
        if len(ordered) > 1:
            duplicates[f"{key[0]}::seed={key[1]}"] = ordered
    return sorted(selected, key=lambda r: r.key), duplicates


def summarize_arm(records: Sequence[RunRecord]) -> dict[str, Any]:
    statuses = Counter(record.status for record in records)
    verifier_statuses = Counter(
        record.verifier_status or "MISSING" for record in records
    )
    rewards = [r.reward for r in records if r.reward is not None]
    pass_rates = [
        r.test_pass_rate for r in records if r.test_pass_rate is not None
    ]
    durations = [
        r.duration_seconds for r in records if r.duration_seconds is not None
    ]
    full_passes = sum(
        1 for record in records
        if record.reward is not None and record.reward >= 1.0
    )
    trajectory_count = sum(record.has_trajectory for record in records)
    events: Counter[str] = Counter()
    total_events = 0
    total_requests = 0
    for record in records:
        total_events += int(record.trajectory.get("event_count") or 0)
        events.update(record.trajectory.get("event_types") or {})
        total_requests += int(
            (record.trajectory.get("model_request_latency_ms") or {}).get(
                "count"
            ) or 0
        )

    return {
        "task_count": len(records),
        "evaluated_count": len(rewards),
        "infra_error_count": statuses.get("ERROR", 0),
        "status_counts": dict(sorted(statuses.items())),
        "verifier_status_counts": dict(sorted(verifier_statuses.items())),
        "accuracy": {
            "mean_reward": _mean(rewards),
            "mean_test_pass_rate": _mean(pass_rates),
            "full_pass_count": full_passes,
            "full_pass_rate": full_passes / len(rewards) if rewards else None,
        },
        "latency_seconds": {
            "count": len(durations),
            "mean": _mean(durations),
            "median": _percentile(durations, 0.50),
            "p90": _percentile(durations, 0.90),
            "p95": _percentile(durations, 0.95),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "trajectory": {
            "present_count": trajectory_count,
            "coverage_rate": (
                trajectory_count / len(records) if records else None
            ),
            "total_event_count": total_events,
            "total_model_request_count": total_requests,
            "event_type_counts": dict(sorted(events.items())),
        },
    }


def compare_paired(
    subagent: Sequence[RunRecord], single_agent: Sequence[RunRecord]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sub_map = {record.key: record for record in subagent}
    single_map = {record.key: record for record in single_agent}
    common = sorted(sub_map.keys() & single_map.keys())

    rows: list[dict[str, Any]] = []
    reward_deltas: list[float] = []
    latency_deltas: list[float] = []
    speedups: list[float] = []
    accuracy_wins = accuracy_ties = accuracy_losses = 0
    latency_wins = latency_ties = latency_losses = 0

    for key in common:
        sub = sub_map[key]
        single = single_map[key]
        reward_delta = None
        if sub.reward is not None and single.reward is not None:
            reward_delta = sub.reward - single.reward
            reward_deltas.append(reward_delta)
            if reward_delta > 1e-12:
                accuracy_wins += 1
            elif reward_delta < -1e-12:
                accuracy_losses += 1
            else:
                accuracy_ties += 1

        latency_delta = None
        speedup = None
        if (
            sub.duration_seconds is not None
            and single.duration_seconds is not None
        ):
            # Positive delta and speedup > 1 mean subagent is faster.
            latency_delta = single.duration_seconds - sub.duration_seconds
            latency_deltas.append(latency_delta)
            if sub.duration_seconds > 0:
                speedup = single.duration_seconds / sub.duration_seconds
                if speedup > 0:
                    speedups.append(speedup)
            if latency_delta > 1e-9:
                latency_wins += 1
            elif latency_delta < -1e-9:
                latency_losses += 1
            else:
                latency_ties += 1

        rows.append(
            {
                "instance_id": key[0],
                "seed": key[1],
                "subagent_job_id": sub.job_id,
                "single_agent_job_id": single.job_id,
                "subagent_status": sub.status,
                "single_agent_status": single.status,
                "subagent_reward": sub.reward,
                "single_agent_reward": single.reward,
                "reward_delta_subagent_minus_single": reward_delta,
                "subagent_duration_seconds": sub.duration_seconds,
                "single_agent_duration_seconds": single.duration_seconds,
                "latency_delta_single_minus_subagent_seconds": latency_delta,
                "subagent_speedup_single_over_subagent": speedup,
                "subagent_trajectory": sub.has_trajectory,
                "single_agent_trajectory": single.has_trajectory,
                "subagent_trajectory_path": (
                    str(sub.trajectory_path) if sub.trajectory_path else None
                ),
                "single_agent_trajectory_path": (
                    str(single.trajectory_path)
                    if single.trajectory_path else None
                ),
            }
        )

    both_trajectories = sum(
        bool(row["subagent_trajectory"])
        and bool(row["single_agent_trajectory"])
        for row in rows
    )
    geo_speedup = None
    if speedups:
        geo_speedup = math.exp(
            statistics.fmean(math.log(value) for value in speedups)
        )

    summary = {
        "paired_task_count": len(common),
        "subagent_only": [
            {"instance_id": key[0], "seed": key[1]}
            for key in sorted(sub_map.keys() - single_map.keys())
        ],
        "single_agent_only": [
            {"instance_id": key[0], "seed": key[1]}
            for key in sorted(single_map.keys() - sub_map.keys())
        ],
        "accuracy": {
            "paired_evaluable_count": len(reward_deltas),
            "mean_reward_delta_subagent_minus_single": _mean(reward_deltas),
            "subagent_wins": accuracy_wins,
            "ties": accuracy_ties,
            "subagent_losses": accuracy_losses,
        },
        "latency": {
            "paired_evaluable_count": len(latency_deltas),
            "mean_delta_single_minus_subagent_seconds": _mean(latency_deltas),
            "median_delta_single_minus_subagent_seconds": _percentile(
                latency_deltas, 0.50
            ),
            "geomean_speedup_single_over_subagent": geo_speedup,
            "subagent_faster": latency_wins,
            "ties": latency_ties,
            "subagent_slower": latency_losses,
        },
        "trajectory": {
            "both_present_count": both_trajectories,
            "both_present_rate": (
                both_trajectories / len(rows) if rows else None
            ),
        },
    }
    return summary, rows


def record_to_dict(record: RunRecord) -> dict[str, Any]:
    latency = record.trajectory.get("model_request_latency_ms") or {}
    events = record.trajectory.get("event_types") or {}
    serving = record.manifest.get("serving_tokens") or {}
    result = {
        "arm": record.arm,
        "job_id": record.job_id,
        "pbs_run_dir": str(record.pbs_run_dir),
        "instance_id": record.instance_id,
        "seed": record.seed,
        "run_id": record.manifest.get("run_id"),
        "status": record.status,
        "verifier_status": record.verifier_status or None,
        "reward": record.reward,
        "test_pass_rate": record.test_pass_rate,
        "duration_seconds": record.duration_seconds,
        "agent_execution_outcome": record.manifest.get(
            "agent_execution_outcome"
        ),
        "trajectory_present": record.has_trajectory,
        "trajectory_path": (
            str(record.trajectory_path) if record.trajectory_path else None
        ),
        "trajectory_event_count": record.trajectory.get("event_count"),
        "trajectory_timed_event_count": record.trajectory.get(
            "timed_event_count"
        ),
        "trajectory_untimed_event_count": record.trajectory.get(
            "untimed_event_count"
        ),
        "trajectory_span_seconds": record.trajectory.get("span_seconds"),
        "trajectory_session_count": len(
            record.trajectory.get("session_keys") or []
        ),
        "model_request_count": latency.get("count"),
        "model_request_latency_mean_ms": latency.get("mean"),
        "model_request_latency_p50_ms": latency.get("p50"),
        "model_request_latency_p95_ms": latency.get("p95"),
        "model_request_latency_max_ms": latency.get("max"),
        "total_prompt_tokens": record.manifest.get("total_prompt_tokens", 0),
        "total_completion_tokens": record.manifest.get(
            "total_completion_tokens", 0
        ),
        "serving_processed_input_tokens": serving.get("processed_input"),
        "serving_generated_output_tokens": serving.get("generated_output"),
        "serving_request_count": serving.get("request_count"),
        "manifest_path": str(record.manifest_path),
    }
    for event in CORE_EVENTS:
        result[f"{event}_count"] = int(events.get(event) or 0)
    return result


def _duplicates_json(
    duplicates: dict[str, list[RunRecord]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [
            {
                "job_id": record.job_id,
                "status": record.status,
                "reward": record.reward,
                "trajectory_present": record.has_trajectory,
                "manifest_path": str(record.manifest_path),
            }
            for record in group
        ]
        for key, group in sorted(duplicates.items())
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jobs(records: Sequence[RunRecord]) -> list[str]:
    return sorted({record.job_id for record in records})


def _default_name(
    subagent: Sequence[RunRecord],
    single_agent: Sequence[RunRecord] | None = None,
) -> str:
    sub = "+".join(_jobs(subagent)) or "subagent"
    if single_agent is None:
        return f"roadmap-results-{_safe_name(sub)}"
    single = "+".join(_jobs(single_agent)) or "single-agent"
    return (
        f"roadmap-comparison-subagent-{_safe_name(sub)}"
        f"--single-agent-{_safe_name(single)}"
    )


def _arm_markdown(name: str, summary: dict[str, Any]) -> str:
    accuracy = summary["accuracy"]
    latency = summary["latency_seconds"]
    trajectory = summary["trajectory"]
    return (
        f"| {name} | {summary['task_count']} | {summary['evaluated_count']} | "
        f"{summary['infra_error_count']} | {_fmt(accuracy['mean_reward'])} | "
        f"{_fmt(accuracy['full_pass_rate'])} | {_fmt(latency['mean'], 1)} | "
        f"{_fmt(latency['median'], 1)} | {_fmt(latency['p95'], 1)} | "
        f"{_fmt(trajectory['coverage_rate'])} |"
    )


def render_markdown(
    subagent: Sequence[RunRecord],
    sub_summary: dict[str, Any],
    *,
    single_agent: Sequence[RunRecord] | None = None,
    single_summary: dict[str, Any] | None = None,
    paired_summary: dict[str, Any] | None = None,
    paired_rows: Sequence[dict[str, Any]] | None = None,
    duplicate_count: int = 0,
) -> str:
    lines = [
        "# RoadmapBench accuracy / latency / trajectory report",
        "",
        "Accuracy uses the authoritative verifier `reward`. Task latency uses "
        "`manifest.duration_seconds`; agent self-reported test claims are not "
        "used as accuracy.",
        "",
        "| Arm | Tasks | Evaluated | Infra errors | Mean reward | Full-pass "
        "rate | Mean latency (s) | Median (s) | P95 (s) | Trajectory coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _arm_markdown("subagent", sub_summary),
    ]
    if single_summary is not None:
        lines.append(_arm_markdown("single-agent", single_summary))

    lines += [
        "",
        "## Input PBS jobs",
        "",
        f"- subagent: `{', '.join(_jobs(subagent))}`",
    ]
    if single_agent is not None:
        lines.append(f"- single-agent: `{', '.join(_jobs(single_agent))}`")

    if paired_summary is not None:
        accuracy = paired_summary["accuracy"]
        latency = paired_summary["latency"]
        trajectory = paired_summary["trajectory"]
        lines += [
            "",
            "## Paired comparison",
            "",
            f"- paired tasks: **{paired_summary['paired_task_count']}**",
            "- mean reward delta (subagent - single-agent): "
            f"**{_fmt(accuracy['mean_reward_delta_subagent_minus_single'])}**",
            "- accuracy W/T/L for subagent: "
            f"**{accuracy['subagent_wins']}/{accuracy['ties']}/"
            f"{accuracy['subagent_losses']}**",
            "- mean latency delta (single-agent - subagent): "
            f"**{_fmt(latency['mean_delta_single_minus_subagent_seconds'], 1)} s**",
            "- geometric mean speedup (single-agent / subagent): "
            f"**{_fmt(latency['geomean_speedup_single_over_subagent'])}x**",
            "- latency faster/tie/slower for subagent: "
            f"**{latency['subagent_faster']}/{latency['ties']}/"
            f"{latency['subagent_slower']}**",
            "- both trajectories present: "
            f"**{trajectory['both_present_count']}/"
            f"{paired_summary['paired_task_count']}**",
            "",
            "Positive reward delta favors subagent. Positive latency delta and "
            "speedup > 1 mean subagent is faster.",
        ]
        if paired_rows:
            lines += [
                "",
                "## Per-task paired results",
                "",
                "| Task | Seed | Sub reward | Single reward | Δ reward | Sub "
                "latency (s) | Single latency (s) | Speedup | Trajectory |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
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
                    f"{_fmt(row['subagent_reward'])} | "
                    f"{_fmt(row['single_agent_reward'])} | "
                    f"{_fmt(row['reward_delta_subagent_minus_single'])} | "
                    f"{_fmt(row['subagent_duration_seconds'], 1)} | "
                    f"{_fmt(row['single_agent_duration_seconds'], 1)} | "
                    f"{_fmt(row['subagent_speedup_single_over_subagent'])}x | "
                    f"{trajectory_ok} |"
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

    if duplicate_count:
        lines += [
            "",
            "## Retry / duplicate selection",
            "",
            f"{duplicate_count} superseded manifest(s) were found. For each "
            "`instance_id + seed`, the analyzer prefers a run with verifier "
            "reward, then non-ERROR status, trajectory presence, and the latest "
            "terminal timestamp. A focused smoke/retry can therefore replace "
            "an earlier infrastructure ERROR without double-counting the task.",
        ]
    lines.append("")
    return "\n".join(lines)


def _trajectory_index(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    return [
        {
            "arm": record.arm,
            "job_id": record.job_id,
            "instance_id": record.instance_id,
            "seed": record.seed,
            "status": record.status,
            "trajectory_present": record.has_trajectory,
            "trajectory_path": (
                str(record.trajectory_path) if record.trajectory_path else None
            ),
            "event_count": record.trajectory.get("event_count"),
            "timed_event_count": record.trajectory.get("timed_event_count"),
            "untimed_event_count": record.trajectory.get("untimed_event_count"),
            "span_seconds": record.trajectory.get("span_seconds"),
            "event_types": record.trajectory.get("event_types"),
        }
        for record in records
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize RoadmapBench PBS artifacts or compare paired subagent "
            "and single-agent runs."
        )
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="PBS job ids/run directories for summary-only mode.",
    )
    parser.add_argument(
        "--subagent", nargs="+", help="Subagent PBS job ids/run directories."
    )
    parser.add_argument(
        "--single-agent",
        nargs="+",
        help="Single-agent PBS job ids/run directories.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root used to resolve PBS job ids (default: runs).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/roadmap",
        help="Directory for JSON/CSV/Markdown reports.",
    )
    parser.add_argument(
        "--name", help="Optional report basename; PBS ids are used by default."
    )
    parser.add_argument(
        "--strict-trajectory",
        action="store_true",
        help="Exit 2 if any selected non-ERROR task lacks trajectory.jsonl.",
    )
    args = parser.parse_args(argv)

    comparison = bool(args.subagent or args.single_agent)
    if comparison:
        if args.runs:
            parser.error(
                "positional runs cannot be mixed with --subagent/--single-agent"
            )
        if not args.subagent or not args.single_agent:
            parser.error(
                "comparison mode requires both --subagent and --single-agent"
            )
    elif not args.runs:
        parser.error(
            "provide positional PBS jobs or both comparison arms"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = bool(args.subagent)
    if comparison:
        raw_sub = collect_records(
            args.subagent, arm="subagent", runs_root=runs_root
        )
        raw_single = collect_records(
            args.single_agent, arm="single-agent", runs_root=runs_root
        )
        subagent, sub_duplicates = select_best_records(raw_sub)
        single_agent, single_duplicates = select_best_records(raw_single)
        sub_summary = summarize_arm(subagent)
        single_summary = summarize_arm(single_agent)
        paired_summary, paired_rows = compare_paired(subagent, single_agent)
        duplicates = {
            **{f"subagent::{k}": v for k, v in sub_duplicates.items()},
            **{
                f"single-agent::{k}": v
                for k, v in single_duplicates.items()
            },
        }
        duplicate_count = sum(len(group) - 1 for group in duplicates.values())
        basename = args.name or _default_name(subagent, single_agent)
        selected = [*subagent, *single_agent]
        report = {
            "mode": "comparison",
            "subagent_jobs": _jobs(subagent),
            "single_agent_jobs": _jobs(single_agent),
            "subagent": sub_summary,
            "single_agent": single_summary,
            "paired": paired_summary,
            "tasks": {
                "subagent": [record_to_dict(r) for r in subagent],
                "single_agent": [record_to_dict(r) for r in single_agent],
                "paired": paired_rows,
            },
            "duplicates": _duplicates_json(duplicates),
        }
        markdown = render_markdown(
            subagent,
            sub_summary,
            single_agent=single_agent,
            single_summary=single_summary,
            paired_summary=paired_summary,
            paired_rows=paired_rows,
            duplicate_count=duplicate_count,
        )
    else:
        raw_sub = collect_records(args.runs, arm="subagent", runs_root=runs_root)
        subagent, duplicates = select_best_records(raw_sub)
        sub_summary = summarize_arm(subagent)
        duplicate_count = sum(len(group) - 1 for group in duplicates.values())
        basename = args.name or _default_name(subagent)
        selected = list(subagent)
        paired_rows = []
        report = {
            "mode": "summary",
            "jobs": _jobs(subagent),
            "summary": sub_summary,
            "tasks": [record_to_dict(r) for r in subagent],
            "duplicates": _duplicates_json(duplicates),
        }
        markdown = render_markdown(
            subagent,
            sub_summary,
            duplicate_count=duplicate_count,
        )

    basename = _safe_name(basename)
    json_path = output_dir / f"{basename}.json"
    md_path = output_dir / f"{basename}.md"
    tasks_csv = output_dir / f"{basename}-tasks.csv"
    trajectories_json = output_dir / f"{basename}-trajectories.json"

    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(markdown)
    _write_csv(tasks_csv, [record_to_dict(record) for record in selected])
    trajectories_json.write_text(
        json.dumps(_trajectory_index(selected), indent=2) + "\n"
    )

    generated = [json_path, md_path, tasks_csv, trajectories_json]
    if comparison:
        paired_csv = output_dir / f"{basename}-paired.csv"
        _write_csv(paired_csv, paired_rows)
        generated.append(paired_csv)

    missing_trajectory = [
        record
        for record in selected
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
