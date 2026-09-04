from __future__ import annotations

import json
from pathlib import Path

from scripts import compare_roadmap_runs as compare


def _record(
    tmp_path: Path,
    *,
    arm: str = "subagent",
    instance_id: str = "task-1",
    duration_seconds: float | None = 100.0,
    agent_execution_seconds: float | None = None,
) -> compare.RunRecord:
    run_dir = tmp_path / f"{arm}-{instance_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "instance_id": instance_id,
        "seed": 42,
        "status": "DONE",
        "reward": 1.0,
        "duration_seconds": duration_seconds,
    }
    if agent_execution_seconds is not None:
        manifest["agent_execution_seconds"] = agent_execution_seconds
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return compare.RunRecord(
        arm=arm,
        job_id="1.pbs101",
        pbs_run_dir=run_dir.parent,
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory_path=None,
        trajectory={
            "present": False,
            "event_count": 0,
            "event_types": {
                "spawn": 3,
                "model_request": 11,
            },
            "timed_event_count": 0,
            "untimed_event_count": 0,
            "span_seconds": None,
            "session_keys": [],
            "model_request_latency_ms": {
                "count": 11,
                "mean": 10.0,
                "p50": 10.0,
                "p95": 10.0,
                "max": 10.0,
            },
        },
    )


def test_latency_never_falls_back_to_duration(tmp_path):
    record = _record(tmp_path, duration_seconds=123.0)
    assert record.duration_seconds == 123.0
    assert record.agent_execution_seconds is None
    assert record.latency_seconds is None
    assert record.latency_source is None


def test_paired_latency_requires_agent_execution_seconds_on_both_arms(tmp_path):
    sub = _record(tmp_path, arm="subagent", duration_seconds=90.0)
    single = _record(tmp_path, arm="single-agent", duration_seconds=120.0)

    summary, rows = compare.compare_paired(
        [sub],
        [single],
        bootstrap_samples=100,
    )

    assert len(rows) == 1
    assert summary["latency"]["paired_evaluable_count"] == 0
    assert rows[0]["subagent_latency_seconds"] is None
    assert rows[0]["single_agent_latency_seconds"] is None


def test_actual_subagent_count_comes_from_runtime_observability(tmp_path):
    record = _record(tmp_path, agent_execution_seconds=60.0)
    record.manifest["phase_timing"] = {
        "agent_execution_start_ns": 1_000_000_000,
        "agent_execution_end_ns": 61_000_000_000,
    }
    record.manifest["runtime_observability_path"] = str(
        record.run_dir / "runtime_observability.json"
    )
    (record.run_dir / "runtime_observability.json").write_text(
        json.dumps(
            {
                "child_session_count": 2,
                "child_runtime_complete_count": 2,
                "max_concurrent_children": 2,
                "child_sum_runtime_seconds": 40.0,
                "child_union_active_seconds": 25.0,
                "child_overlap_wall_seconds": 15.0,
                "child_concurrency_excess_seconds": 15.0,
                "child_runtime_by_session": {
                    "child-a": {
                        "start_timestamp_ns": 5_000_000_000,
                        "terminal_timestamp_ns": 25_000_000_000,
                        "runtime_seconds": 20.0,
                    },
                    "child-b": {
                        "start_timestamp_ns": 10_000_000_000,
                        "terminal_timestamp_ns": 30_000_000_000,
                        "runtime_seconds": 20.0,
                    },
                },
            }
        )
    )

    row = compare.record_to_dict(record)

    assert row["materialized_subagent_count"] == 2
    assert row["max_concurrent_subagents"] == 2
    assert len(row["child_intervals"]) == 2
    assert "spawn_count" not in row
    assert "model_request_count" not in row


def test_roadmap_profile_and_prompt_allow_sixteen_concurrent_subagents():
    root = Path(__file__).resolve().parents[1]
    profile = (
        root / "profiles" / "harnesses" / "openclaw_roadmapbench.yaml"
    ).read_text()
    prompt = (
        root
        / "agent_bench"
        / "benchmarks"
        / "roadmap_bench"
        / "openclaw"
        / "agents"
        / "AGENTS.md"
    ).read_text()

    assert "maxConcurrent: 16" in profile
    assert "Up to sixteen subagents may run concurrently." in prompt
