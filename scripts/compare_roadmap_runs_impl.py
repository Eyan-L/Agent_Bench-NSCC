#!/usr/bin/env python3
"""Compare RoadmapBench subagent and single-agent PBS runs offline.

The analyzer is deliberately read-only with respect to benchmark runs: it only
consumes artifacts already produced by Agent_Bench. In paired mode it reports
accuracy, latency, trajectory/resource efficiency, retry selection, and data
coverage, then writes machine-readable tables plus publication-friendly plots.

Examples
--------
Summarize one or more existing subagent PBS jobs::

    python scripts/compare_roadmap_runs.py \
      16056107.pbs101 16056108.pbs101

Compare the formal paired arms (focused retries/smokes may be added; the best
record for each instance_id + seed is selected automatically)::

    python scripts/compare_roadmap_runs.py \
      --subagent 16056107.pbs101 16056108.pbs101 1609XXXX.pbs101 \
      --single-agent 1610AAAA.pbs101 1610BBBB.pbs101 \
      --strict-trajectory

Outputs are written below reports/roadmap by default:
  * Markdown report (also printed to stdout)
  * JSON report
  * per-run tasks CSV
  * paired per-task CSV (comparison mode)
  * trajectory index JSON
  * paired figures (PNG, when matplotlib is available)

The statistical intervals in paired mode are deterministic non-parametric
bootstrap intervals across paired RoadmapBench tasks. They quantify across-task
variation; with one seed they are not a substitute for repeated stochastic runs.
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
import random
import re
import statistics
from typing import Any, Callable, Iterable, Sequence


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
TRAJECTORY_COMPARE_FIELDS = (
    ("model_request_count", "Model requests"),
    ("tool_call_count", "Tool calls"),
    ("total_prompt_tokens", "Prompt tokens"),
    ("total_completion_tokens", "Completion tokens"),
    ("trajectory_session_count", "Sessions"),
    ("compaction_count", "Compactions"),
    ("context_overflow_count", "Context overflows"),
    ("loop_detected_count", "Loop detections"),
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
    def agent_execution_seconds(self) -> float | None:
        value = _number(self.manifest.get("agent_execution_seconds"))
        if value is not None:
            return value
        phase = self.manifest.get("phase_timing") or {}
        return _number(phase.get("agent_execution_seconds")) if isinstance(phase, dict) else None

    @property
    def total_instance_seconds(self) -> float | None:
        return _number(self.manifest.get("total_instance_seconds"))

    @property
    def latency_seconds(self) -> float | None:
        """Primary Roadmap latency: logical agent wall time, with legacy fallback."""
        value = self.agent_execution_seconds
        return value if value is not None else self.duration_seconds

    @property
    def latency_source(self) -> str | None:
        if self.agent_execution_seconds is not None:
            return "agent_execution_seconds"
        if self.duration_seconds is not None:
            return "legacy_duration_seconds"
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


def _int_number(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


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


def _fmt_ci(ci: Sequence[float] | None, digits: int = 3) -> str:
    if not ci or len(ci) != 2:
        return "NA"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_") or "run"


def _job_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = PBS_JOB_RE.search(part)
        if match:
            return match.group("job")
    return "unknown-job"


def _bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    samples: int,
    seed: int = 20260903,
    confidence: float = 0.95,
) -> list[float] | None:
    """Deterministic percentile bootstrap CI over paired-task observations."""
    if not values:
        return None
    if len(values) == 1:
        point = float(statistic(values))
        return [point, point]
    rng = random.Random(seed)
    n = len(values)
    draws: list[float] = []
    for _ in range(max(1, samples)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        result = float(statistic(sample))
        if math.isfinite(result):
            draws.append(result)
    if not draws:
        return None
    alpha = (1.0 - confidence) / 2.0
    lo = _percentile(draws, alpha)
    hi = _percentile(draws, 1.0 - alpha)
    if lo is None or hi is None:
        return None
    return [lo, hi]


def _exact_sign_test_two_sided(wins: int, losses: int) -> float | None:
    """Exact paired sign test, excluding ties, without scipy dependency."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


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
        "total_instance_seconds": record.total_instance_seconds,
        "agent_execution_seconds": record.agent_execution_seconds,
        "latency_seconds": record.latency_seconds,
        "latency_source": record.latency_source,
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
    phase = record.manifest.get("phase_timing") or {}
    if isinstance(phase, dict):
        for field in (
            "workspace_prepare_seconds",
            "gateway_warmup_queue_seconds",
            "gateway_warmup_seconds",
            "environment_prepare_seconds",
            "parent_execution_seconds",
            "post_parent_quiescence_seconds",
            "candidate_freeze_seconds",
            "verifier_seconds",
            "cleanup_seconds",
        ):
            result[field] = _number(phase.get(field))
    for event in CORE_EVENTS:
        result[f"{event}_count"] = int(events.get(event) or 0)
    return result


def summarize_arm(records: Sequence[RunRecord]) -> dict[str, Any]:
    statuses = Counter(record.status for record in records)
    verifier_statuses = Counter(
        record.verifier_status or "MISSING" for record in records
    )
    outcomes = Counter(
        str(record.manifest.get("agent_execution_outcome") or "MISSING")
        for record in records
    )
    rows = [record_to_dict(record) for record in records]
    rewards = [record.reward for record in records if record.reward is not None]
    pass_rates = [
        record.test_pass_rate
        for record in records
        if record.test_pass_rate is not None
    ]
    durations = [
        record.latency_seconds
        for record in records
        if record.latency_seconds is not None
    ]
    latency_sources = Counter(
        record.latency_source or "MISSING" for record in records
    )
    full_passes = sum(
        1 for record in records
        if record.reward is not None and record.reward >= 1.0
    )
    trajectory_count = sum(record.has_trajectory for record in records)
    events: Counter[str] = Counter()
    for record in records:
        events.update(record.trajectory.get("event_types") or {})

    def values(field: str) -> list[float]:
        return [
            value
            for row in rows
            if (value := _number(row.get(field))) is not None
        ]

    efficiency = {
        "mean_model_requests_per_task": _mean(values("model_request_count")),
        "mean_tool_calls_per_task": _mean(values("tool_call_count")),
        "mean_prompt_tokens_per_task": _mean(values("total_prompt_tokens")),
        "mean_completion_tokens_per_task": _mean(
            values("total_completion_tokens")
        ),
        "mean_sessions_per_task": _mean(values("trajectory_session_count")),
        "total_spawns": sum(_int_number(row.get("spawn_count")) for row in rows),
        "total_compactions": sum(
            _int_number(row.get("compaction_count")) for row in rows
        ),
        "total_context_overflows": sum(
            _int_number(row.get("context_overflow_count")) for row in rows
        ),
        "total_loop_detections": sum(
            _int_number(row.get("loop_detected_count")) for row in rows
        ),
    }

    return {
        "task_count": len(records),
        "evaluated_count": len(rewards),
        "infra_error_count": statuses.get("ERROR", 0),
        "status_counts": dict(sorted(statuses.items())),
        "verifier_status_counts": dict(sorted(verifier_statuses.items())),
        "agent_execution_outcome_counts": dict(sorted(outcomes.items())),
        "accuracy": {
            "mean_reward": _mean(rewards),
            "median_reward": _median(rewards),
            "mean_test_pass_rate": _mean(pass_rates),
            "full_pass_count": full_passes,
            "full_pass_rate": full_passes / len(rewards) if rewards else None,
        },
        "latency_seconds": {
            "primary_metric": "agent_execution_seconds",
            "legacy_fallback": "duration_seconds",
            "source_counts": dict(sorted(latency_sources.items())),
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
            "total_event_count": sum(
                _int_number(record.trajectory.get("event_count"))
                for record in records
            ),
            "total_model_request_count": sum(
                _int_number(row.get("model_request_count")) for row in rows
            ),
            "event_type_counts": dict(sorted(events.items())),
        },
        "efficiency": efficiency,
    }


def _paired_metric_summary(
    rows: Sequence[dict[str, Any]],
    sub_field: str,
    single_field: str,
) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        sub = _number(row.get(sub_field))
        single = _number(row.get(single_field))
        if sub is not None and single is not None:
            pairs.append((sub, single))
    deltas = [sub - single for sub, single in pairs]
    ratios = [
        single / sub
        for sub, single in pairs
        if sub > 0 and single >= 0
    ]
    positive_ratios = [value for value in ratios if value > 0]
    return {
        "paired_count": len(pairs),
        "subagent_mean": _mean([pair[0] for pair in pairs]),
        "single_agent_mean": _mean([pair[1] for pair in pairs]),
        "mean_delta_subagent_minus_single": _mean(deltas),
        "median_delta_subagent_minus_single": _median(deltas),
        "geomean_ratio_single_over_subagent": (
            math.exp(statistics.fmean(math.log(value) for value in positive_ratios))
            if positive_ratios
            else None
        ),
    }


def compare_paired(
    subagent: Sequence[RunRecord],
    single_agent: Sequence[RunRecord],
    *,
    bootstrap_samples: int,
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
        sub_flat = record_to_dict(sub)
        single_flat = record_to_dict(single)

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
            sub.latency_seconds is not None
            and single.latency_seconds is not None
        ):
            latency_delta = single.latency_seconds - sub.latency_seconds
            latency_deltas.append(latency_delta)
            if sub.latency_seconds > 0 and single.latency_seconds > 0:
                speedup = single.latency_seconds / sub.latency_seconds
                speedups.append(speedup)
            if latency_delta > 1e-9:
                latency_wins += 1
            elif latency_delta < -1e-9:
                latency_losses += 1
            else:
                latency_ties += 1

        row: dict[str, Any] = {
            "instance_id": key[0],
            "seed": key[1],
            "subagent_job_id": sub.job_id,
            "single_agent_job_id": single.job_id,
            "subagent_status": sub.status,
            "single_agent_status": single.status,
            "subagent_verifier_status": sub.verifier_status or None,
            "single_agent_verifier_status": single.verifier_status or None,
            "subagent_reward": sub.reward,
            "single_agent_reward": single.reward,
            "reward_delta_subagent_minus_single": reward_delta,
            "subagent_duration_seconds": sub.duration_seconds,
            "single_agent_duration_seconds": single.duration_seconds,
            "subagent_total_instance_seconds": sub.total_instance_seconds,
            "single_agent_total_instance_seconds": single.total_instance_seconds,
            "subagent_agent_execution_seconds": sub.agent_execution_seconds,
            "single_agent_agent_execution_seconds": single.agent_execution_seconds,
            "subagent_latency_seconds": sub.latency_seconds,
            "single_agent_latency_seconds": single.latency_seconds,
            "subagent_latency_source": sub.latency_source,
            "single_agent_latency_source": single.latency_source,
            "latency_delta_single_minus_subagent_seconds": latency_delta,
            "subagent_speedup_single_over_subagent": speedup,
            "subagent_trajectory": sub.has_trajectory,
            "single_agent_trajectory": single.has_trajectory,
            "subagent_trajectory_path": (
                str(sub.trajectory_path) if sub.trajectory_path else None
            ),
            "single_agent_trajectory_path": (
                str(single.trajectory_path) if single.trajectory_path else None
            ),
        }
        for field, _label in TRAJECTORY_COMPARE_FIELDS:
            sub_value = _number(sub_flat.get(field))
            single_value = _number(single_flat.get(field))
            row[f"subagent_{field}"] = sub_value
            row[f"single_agent_{field}"] = single_value
            row[f"{field}_delta_subagent_minus_single"] = (
                sub_value - single_value
                if sub_value is not None and single_value is not None
                else None
            )
        rows.append(row)

    both_trajectories = sum(
        bool(row["subagent_trajectory"])
        and bool(row["single_agent_trajectory"])
        for row in rows
    )
    reward_ci = _bootstrap_ci(
        reward_deltas,
        statistics.fmean,
        samples=bootstrap_samples,
        seed=20260903,
    )
    latency_ci = _bootstrap_ci(
        latency_deltas,
        statistics.fmean,
        samples=bootstrap_samples,
        seed=20260904,
    )
    speedup_ci = _bootstrap_ci(
        speedups,
        lambda values: math.exp(
            statistics.fmean(math.log(value) for value in values)
        ),
        samples=bootstrap_samples,
        seed=20260905,
    )
    geo_speedup = (
        math.exp(statistics.fmean(math.log(value) for value in speedups))
        if speedups
        else None
    )

    efficiency = {
        field: _paired_metric_summary(
            rows, f"subagent_{field}", f"single_agent_{field}"
        )
        for field, _label in TRAJECTORY_COMPARE_FIELDS
    }

    summary = {
        "paired_task_count": len(common),
        "paired_seeds": sorted({key[1] for key in common}),
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
            "median_reward_delta_subagent_minus_single": _median(reward_deltas),
            "mean_reward_delta_95pct_bootstrap_ci": reward_ci,
            "subagent_wins": accuracy_wins,
            "ties": accuracy_ties,
            "subagent_losses": accuracy_losses,
            "exact_sign_test_two_sided_p": _exact_sign_test_two_sided(
                accuracy_wins, accuracy_losses
            ),
            "paired_sign_effect": (
                (accuracy_wins - accuracy_losses)
                / (accuracy_wins + accuracy_losses)
                if accuracy_wins + accuracy_losses
                else None
            ),
        },
        "latency": {
            "paired_evaluable_count": len(latency_deltas),
            "mean_delta_single_minus_subagent_seconds": _mean(latency_deltas),
            "median_delta_single_minus_subagent_seconds": _median(latency_deltas),
            "mean_delta_95pct_bootstrap_ci_seconds": latency_ci,
            "geomean_speedup_single_over_subagent": geo_speedup,
            "median_speedup_single_over_subagent": _median(speedups),
            "geomean_speedup_95pct_bootstrap_ci": speedup_ci,
            "subagent_faster": latency_wins,
            "ties": latency_ties,
            "subagent_slower": latency_losses,
            "exact_sign_test_two_sided_p": _exact_sign_test_two_sided(
                latency_wins, latency_losses
            ),
        },
        "trajectory": {
            "both_present_count": both_trajectories,
            "both_present_rate": (
                both_trajectories / len(rows) if rows else None
            ),
            "paired_efficiency": efficiency,
        },
        "statistics": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_unit": "paired RoadmapBench task",
            "note": (
                "Intervals summarize across-task variation. With one seed they "
                "do not estimate run-to-run model stochasticity."
            ),
        },
    }
    return summary, rows


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
    efficiency = summary["efficiency"]
    return (
        f"| {name} | {summary['task_count']} | {summary['evaluated_count']} | "
        f"{summary['infra_error_count']} | {_fmt(accuracy['mean_reward'])} | "
        f"{_fmt(accuracy['full_pass_rate'])} | {_fmt(latency['median'], 1)} | "
        f"{_fmt(latency['p95'], 1)} | {_fmt(trajectory['coverage_rate'])} | "
        f"{_fmt(efficiency['mean_model_requests_per_task'], 1)} | "
        f"{_fmt(efficiency['mean_prompt_tokens_per_task'], 0)} |"
    )


def _efficiency_markdown(
    paired_summary: dict[str, Any],
) -> list[str]:
    efficiency = paired_summary["trajectory"]["paired_efficiency"]
    lines = [
        "| Metric | Paired n | Subagent mean | Single-agent mean | "
        "Δ (sub-single) | Ratio (single/sub) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field, label in TRAJECTORY_COMPARE_FIELDS:
        data = efficiency[field]
        ratio = data["geomean_ratio_single_over_subagent"]
        ratio_text = "NA" if ratio is None else f"{ratio:.3f}x"
        lines.append(
            f"| {label} | {data['paired_count']} | "
            f"{_fmt(data['subagent_mean'], 1)} | "
            f"{_fmt(data['single_agent_mean'], 1)} | "
            f"{_fmt(data['mean_delta_subagent_minus_single'], 1)} | "
            f"{ratio_text} |"
        )
    return lines


def render_markdown(
    subagent: Sequence[RunRecord],
    sub_summary: dict[str, Any],
    *,
    single_agent: Sequence[RunRecord] | None = None,
    single_summary: dict[str, Any] | None = None,
    paired_summary: dict[str, Any] | None = None,
    paired_rows: Sequence[dict[str, Any]] | None = None,
    duplicate_count: int = 0,
    figure_names: Sequence[str] = (),
) -> str:
    lines = [
        "# RoadmapBench paired accuracy / latency / trajectory report",
        "",
        "Accuracy uses the authoritative verifier `reward`. Primary latency uses "
        "`manifest.agent_execution_seconds` (initial parent dispatch through workspace "
        "quiescence). Older manifests without that field fall back explicitly to legacy "
        "`duration_seconds`; agent self-reported test claims are not treated as accuracy.",
        "",
        "## Arm-level summary",
        "",
        "| Arm | Tasks | Evaluated | Infra errors | Mean reward | Full-pass "
        "rate | Median latency (s) | P95 (s) | Trajectory coverage | Mean "
        "model requests | Mean prompt tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            "## Paired primary outcomes",
            "",
            f"- paired tasks: **{paired_summary['paired_task_count']}**",
            f"- paired seeds: **{paired_summary['paired_seeds']}**",
            f"- reward-evaluable pairs: **{accuracy['paired_evaluable_count']}**; "
            f"latency-evaluable pairs: **{latency['paired_evaluable_count']}**",
            "- mean reward delta (subagent - single-agent): "
            f"**{_fmt(accuracy['mean_reward_delta_subagent_minus_single'])}** "
            f"(95% task-bootstrap CI {_fmt_ci(accuracy['mean_reward_delta_95pct_bootstrap_ci'])})",
            "- accuracy W/T/L for subagent: "
            f"**{accuracy['subagent_wins']}/{accuracy['ties']}/"
            f"{accuracy['subagent_losses']}**; exact paired sign-test "
            f"p={_fmt(accuracy['exact_sign_test_two_sided_p'])}",
            "- mean latency delta (single-agent - subagent): "
            f"**{_fmt(latency['mean_delta_single_minus_subagent_seconds'], 1)} s** "
            f"(95% task-bootstrap CI "
            f"{_fmt_ci(latency['mean_delta_95pct_bootstrap_ci_seconds'], 1)})",
            "- geometric-mean speedup (single-agent / subagent): "
            f"**{_fmt(latency['geomean_speedup_single_over_subagent'])}x** "
            f"(95% task-bootstrap CI "
            f"{_fmt_ci(latency['geomean_speedup_95pct_bootstrap_ci'])})",
            "- latency faster/tie/slower for subagent: "
            f"**{latency['subagent_faster']}/{latency['ties']}/"
            f"{latency['subagent_slower']}**; exact paired sign-test "
            f"p={_fmt(latency['exact_sign_test_two_sided_p'])}",
            "- both trajectories present: "
            f"**{trajectory['both_present_count']}/"
            f"{paired_summary['paired_task_count']}**",
            "",
            "Interpretation: positive reward delta favors subagent; latency "
            "delta > 0 and speedup > 1 mean subagent is faster.",
            "",
            "The confidence intervals are bootstrapped over paired tasks. With "
            "one seed they characterize task-to-task variation, not repeated-run "
            "stochastic uncertainty.",
            "",
            "## Trajectory / resource efficiency",
            "",
            *_efficiency_markdown(paired_summary),
        ]

        if figure_names:
            lines += ["", "## Figures", ""]
            captions = {
                "tradeoff": (
                    "Accuracy-latency trade-off; upper-right means subagent is "
                    "both more accurate and faster."
                ),
                "accuracy": "Paired verifier reward by task.",
                "latency": "Per-task agent-trajectory latency speedup; >1 favors subagent.",
                "requests": (
                    "Per-task model-request ratio; >1 means single-agent used "
                    "more model requests."
                ),
            }
            for name in figure_names:
                key = next((key for key in captions if key in name), "figure")
                caption = captions.get(key, "Comparison figure.")
                lines += [
                    f"**{caption}**",
                    "",
                    f"![{caption}]({name})",
                    "",
                ]

        if paired_rows:
            lines += [
                "",
                "## Per-task paired results",
                "",
                "| Task | Seed | Sub reward | Single reward | Δ reward | Sub "
                "latency (s) | Single latency (s) | Speedup | Sub requests | "
                "Single requests | Trajectory |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
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
                    f"{_fmt(row['subagent_latency_seconds'], 1)} | "
                    f"{_fmt(row['single_agent_latency_seconds'], 1)} | "
                    f"{_fmt(row['subagent_speedup_single_over_subagent'])}x | "
                    f"{_fmt(row['subagent_model_request_count'], 0)} | "
                    f"{_fmt(row['single_agent_model_request_count'], 0)} | "
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


def _load_pyplot():
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001 - plotting is intentionally optional
        print(
            "WARNING: plots skipped because matplotlib is unavailable: "
            f"{exc}"
        )
        return None
    return plt


def _plot_accuracy(
    plt: Any, rows: Sequence[dict[str, Any]], path: Path, dpi: int
) -> bool:
    clean = [
        row for row in rows
        if _number(row.get("subagent_reward")) is not None
        and _number(row.get("single_agent_reward")) is not None
    ]
    if not clean:
        return False
    clean = sorted(
        clean,
        key=lambda row: _number(row.get("reward_delta_subagent_minus_single")) or 0,
    )
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.34 * len(clean) + 1.5)))
    y = list(range(len(clean)))
    sub = [float(row["subagent_reward"]) for row in clean]
    single = [float(row["single_agent_reward"]) for row in clean]
    for yi, left, right in zip(y, sub, single):
        ax.plot([left, right], [yi, yi], linewidth=0.8, alpha=0.45)
    ax.scatter(sub, y, marker="o", label="subagent", zorder=3)
    ax.scatter(single, y, marker="x", label="single-agent", zorder=3)
    ax.set_yticks(y, [row["instance_id"] for row in clean], fontsize=8)
    ax.set_xlabel("Verifier reward")
    ax.set_title("RoadmapBench paired accuracy")
    ax.set_xlim(-0.03, 1.03)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_ratio_by_task(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    value_field: str,
    title: str,
    xlabel: str,
    dpi: int,
) -> bool:
    clean = [
        row for row in rows
        if (value := _number(row.get(value_field))) is not None and value > 0
    ]
    if not clean:
        return False
    clean = sorted(clean, key=lambda row: float(row[value_field]))
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.34 * len(clean) + 1.5)))
    y = list(range(len(clean)))
    values = [float(row[value_field]) for row in clean]
    ax.scatter(values, y, marker="o")
    ax.axvline(1.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y, [row["instance_id"] for row in clean], fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_tradeoff(
    plt: Any, rows: Sequence[dict[str, Any]], path: Path, dpi: int
) -> bool:
    clean = [
        row for row in rows
        if _number(row.get("reward_delta_subagent_minus_single")) is not None
        and (speedup := _number(row.get("subagent_speedup_single_over_subagent")))
        is not None
        and speedup > 0
    ]
    if not clean:
        return False
    x = [
        math.log2(float(row["subagent_speedup_single_over_subagent"]))
        for row in clean
    ]
    y = [float(row["reward_delta_subagent_minus_single"]) for row in clean]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, marker="o")
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    for xi, yi, row in zip(x, y, clean):
        ax.annotate(
            row["instance_id"],
            (xi, yi),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
            alpha=0.75,
        )
    ax.set_xlabel("log2 latency speedup (single-agent / subagent)")
    ax.set_ylabel("Reward delta (subagent - single-agent)")
    ax.set_title("Accuracy-latency trade-off by task")
    ax.grid(alpha=0.2)
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
    """Generate paired comparison PNGs without making plotting a hard dependency."""
    plt = _load_pyplot()
    if plt is None or not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    candidates = [
        (
            output_dir / f"{basename}-accuracy-paired.png",
            lambda path: _plot_accuracy(plt, rows, path, dpi),
        ),
        (
            output_dir / f"{basename}-latency-speedup.png",
            lambda path: _plot_ratio_by_task(
                plt,
                rows,
                path,
                value_field="subagent_speedup_single_over_subagent",
                title="RoadmapBench agent-trajectory latency speedup",
                xlabel="single-agent agent latency / subagent agent latency (>1 favors subagent)",
                dpi=dpi,
            ),
        ),
        (
            output_dir / f"{basename}-accuracy-latency-tradeoff.png",
            lambda path: _plot_tradeoff(plt, rows, path, dpi),
        ),
    ]

    request_rows: list[dict[str, Any]] = []
    for row in rows:
        sub = _number(row.get("subagent_model_request_count"))
        single = _number(row.get("single_agent_model_request_count"))
        if sub is None or single is None or sub <= 0 or single <= 0:
            continue
        augmented = dict(row)
        augmented["model_request_ratio_single_over_subagent"] = single / sub
        request_rows.append(augmented)
    candidates.append(
        (
            output_dir / f"{basename}-trajectory-model-requests.png",
            lambda path: _plot_ratio_by_task(
                plt,
                request_rows,
                path,
                value_field="model_request_ratio_single_over_subagent",
                title="Trajectory model-request load",
                xlabel="single-agent requests / subagent requests (>1 means more single-agent requests)",
                dpi=dpi,
            ),
        )
    )

    for path, render in candidates:
        try:
            if render(path):
                generated.append(path)
        except Exception as exc:  # noqa: BLE001 - report data must still survive
            print(f"WARNING: plot generation failed for {path.name}: {exc}")
    return generated


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
        help="Directory for JSON/CSV/Markdown/PNG reports.",
    )
    parser.add_argument(
        "--name", help="Optional report basename; PBS ids are used by default."
    )
    parser.add_argument(
        "--strict-trajectory",
        action="store_true",
        help="Exit 2 if any selected non-ERROR task lacks trajectory.jsonl.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Paired-task bootstrap resamples for 95%% CIs (default: 10000).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate paired PNG figures.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=180,
        help="PNG figure DPI when plotting is available (default: 180).",
    )
    args = parser.parse_args(argv)

    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.plot_dpi < 72:
        parser.error("--plot-dpi must be at least 72")

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
        parser.error("provide positional PBS jobs or both comparison arms")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = bool(args.subagent)
    figure_paths: list[Path] = []
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
        paired_summary, paired_rows = compare_paired(
            subagent,
            single_agent,
            bootstrap_samples=args.bootstrap_samples,
        )
        duplicates = {
            **{f"subagent::{key}": value for key, value in sub_duplicates.items()},
            **{
                f"single-agent::{key}": value
                for key, value in single_duplicates.items()
            },
        }
        duplicate_count = sum(len(group) - 1 for group in duplicates.values())
        basename = _safe_name(args.name or _default_name(subagent, single_agent))
        selected = [*subagent, *single_agent]

        if not args.no_plots:
            figure_paths = generate_plots(
                paired_rows,
                output_dir,
                basename,
                dpi=args.plot_dpi,
            )
        figure_names = [path.name for path in figure_paths]

        report = {
            "schema_version": 2,
            "mode": "comparison",
            "subagent_jobs": _jobs(subagent),
            "single_agent_jobs": _jobs(single_agent),
            "subagent": sub_summary,
            "single_agent": single_summary,
            "paired": paired_summary,
            "tasks": {
                "subagent": [record_to_dict(record) for record in subagent],
                "single_agent": [
                    record_to_dict(record) for record in single_agent
                ],
                "paired": paired_rows,
            },
            "duplicates": _duplicates_json(duplicates),
            "figures": figure_names,
        }
        markdown = render_markdown(
            subagent,
            sub_summary,
            single_agent=single_agent,
            single_summary=single_summary,
            paired_summary=paired_summary,
            paired_rows=paired_rows,
            duplicate_count=duplicate_count,
            figure_names=figure_names,
        )
    else:
        raw_sub = collect_records(
            args.runs, arm="subagent", runs_root=runs_root
        )
        subagent, duplicates = select_best_records(raw_sub)
        sub_summary = summarize_arm(subagent)
        duplicate_count = sum(len(group) - 1 for group in duplicates.values())
        basename = _safe_name(args.name or _default_name(subagent))
        selected = list(subagent)
        paired_rows: list[dict[str, Any]] = []
        report = {
            "schema_version": 2,
            "mode": "summary",
            "jobs": _jobs(subagent),
            "summary": sub_summary,
            "tasks": [record_to_dict(record) for record in subagent],
            "duplicates": _duplicates_json(duplicates),
            "figures": [],
        }
        markdown = render_markdown(
            subagent,
            sub_summary,
            duplicate_count=duplicate_count,
        )

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
    generated.extend(figure_paths)

    missing_trajectory = [
        record
        for record in selected
        if record.status != "ERROR" and not record.has_trajectory
    ]

    print(markdown)
    print("Generated:")
    for path in generated:
        print(f"  {path}")

    if comparison and not args.no_plots and not figure_paths:
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
