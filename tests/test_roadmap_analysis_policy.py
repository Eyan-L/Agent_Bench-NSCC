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
    benchmark_valid: bool | None = None,
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
    if benchmark_valid is not None:
        manifest["benchmark_valid"] = benchmark_valid
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


def test_invalid_benchmark_keeps_diagnostics_but_not_formal_metrics(tmp_path):
    valid = _record(
        tmp_path,
        instance_id="valid",
        agent_execution_seconds=30.0,
        benchmark_valid=True,
    )
    invalid = _record(
        tmp_path,
        instance_id="invalid",
        agent_execution_seconds=90.0,
        benchmark_valid=False,
    )
    invalid.manifest["reward"] = 0.0

    summary = compare.summarize_arm([valid, invalid])

    assert summary["task_count"] == 2
    assert summary["benchmark_valid_count"] == 1
    assert summary["benchmark_invalid_count"] == 1
    assert summary["evaluated_count"] == 1
    assert summary["accuracy"]["mean_reward"] == 1.0
    assert summary["latency_seconds"]["count"] == 1
    assert summary["latency_seconds"]["mean"] == 30.0


def test_invalid_pair_is_excluded_from_formal_paired_metrics(tmp_path):
    sub = _record(
        tmp_path,
        arm="subagent",
        agent_execution_seconds=30.0,
        benchmark_valid=False,
    )
    single = _record(
        tmp_path,
        arm="single-agent",
        agent_execution_seconds=60.0,
        benchmark_valid=True,
    )

    summary, rows = compare.compare_paired(
        [sub], [single], bootstrap_samples=100
    )

    assert len(rows) == 1
    assert rows[0]["pair_benchmark_valid"] is False
    assert summary["paired_benchmark_valid_count"] == 0
    assert summary["paired_benchmark_invalid_count"] == 1
    assert summary["accuracy"]["paired_evaluable_count"] == 0
    assert summary["latency"]["paired_evaluable_count"] == 0


def test_valid_retry_is_selected_over_scored_infrastructure_error(tmp_path):
    invalid = _record(
        tmp_path,
        arm="invalid",
        benchmark_valid=False,
    )
    invalid.manifest["status"] = "ERROR"
    valid = _record(
        tmp_path,
        arm="valid",
        benchmark_valid=True,
    )
    valid.manifest["reward"] = None

    selected, _duplicates = compare.select_best_records([invalid, valid])

    assert selected == [valid]


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
    assert "maxChildrenPerAgent: 16" in profile
    assert "Up to sixteen subagents may run concurrently." in prompt
    assert "you are encouraged" in prompt
    assert "Delegation is preferred" in prompt
    assert "MUST spawn" not in prompt


def test_roadmap_subagents_cannot_park_with_sessions_yield():
    root = Path(__file__).resolve().parents[1]
    subagent_profile = (
        root / "profiles" / "harnesses" / "openclaw_roadmapbench.yaml"
    ).read_text()
    single_agent_profile = (
        root
        / "profiles"
        / "harnesses"
        / "openclaw_roadmapbench_single_agent.yaml"
    ).read_text()

    assert "subagents:\n      tools:\n        deny:\n        - sessions_yield" in subagent_profile
    assert "subagents:\n      tools:\n        deny:\n        - sessions_yield" not in single_agent_profile
    assert "    - sessions_yield" in subagent_profile


def test_roadmap_profile_uses_only_the_benchmark_execution_deadline():
    root = Path(__file__).resolve().parents[1]
    profile = (
        root / "profiles" / "harnesses" / "openclaw_roadmapbench.yaml"
    ).read_text()

    assert "runTimeoutSeconds: 0" in profile
    assert "announceTimeoutMs: 7200000" in profile
    assert "archiveAfterMinutes: 0" in profile


def test_roadmap_profiles_enable_builtin_memory_plugin():
    root = Path(__file__).resolve().parents[1]

    for name in (
        "openclaw_roadmapbench.yaml",
        "openclaw_roadmapbench_single_agent.yaml",
    ):
        profile = (root / "profiles" / "harnesses" / name).read_text()
        assert "memory-core:\n        enabled: true" in profile


def test_roadmap_profiles_use_builtin_env_secret_provider():
    root = Path(__file__).resolve().parents[1]

    for name in (
        "openclaw_roadmapbench.yaml",
        "openclaw_roadmapbench_single_agent.yaml",
    ):
        profile = (root / "profiles" / "harnesses" / name).read_text()
        assert "source: env\n        provider: default" in profile
        assert "provider: env" not in profile


def test_comparison_bundle_copies_selected_trajectory_evidence(tmp_path):
    record = _record(tmp_path, instance_id="task/selected")
    run_dir = record.run_dir
    sessions = run_dir / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "sessions.json").write_text("{}\n")
    (sessions / "parent.jsonl").write_text('{"type":"session"}\n')

    expected = {
        "trajectory.jsonl": '{"event_type":"resume"}\n',
        "spans.jsonl": '{"name":"model"}\n',
        "openclaw_exec_bridge.jsonl": '{"event":"exec"}\n',
        "openclaw_gateway.log": "gateway ready\n",
        "candidate.patch": "diff --git a/a b/a\n",
        "verdict.json": '{"reward":1.0}\n',
    }
    for name, content in expected.items():
        (run_dir / name).write_text(content)

    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    (output_dir / "comparison.json").write_text("legacy")
    (output_dir / "comparison-old.png").write_text("legacy")
    stale = output_dir / "comparison" / "01_durable_session_records" / "stale"
    stale.mkdir(parents=True)
    (stale / "old-task.jsonl").write_text("stale")

    bundle = compare._impl._prepare_bundle_dir(output_dir, "comparison")
    copied = compare._impl._copy_selected_task_artifacts([record], bundle)

    assert {path.name for path in bundle.iterdir()} == set(
        compare._impl.BUNDLE_SUBDIRS
    )
    assert not (output_dir / "comparison.json").exists()
    assert not (output_dir / "comparison-old.png").exists()
    assert not stale.exists()
    task_dir = "task_selected__seed42"
    assert (
        bundle
        / "01_durable_session_records"
        / "subagent"
        / task_dir
        / "conversation"
        / "sessions"
        / "parent.jsonl"
    ).is_file()
    assert (
        bundle
        / "02_timeline_model_trace"
        / "subagent"
        / task_dir
        / "trajectory.jsonl"
    ).is_file()
    assert (
        bundle
        / "03_execution_evidence"
        / "subagent"
        / task_dir
        / "openclaw_exec_bridge.jsonl"
    ).is_file()
    assert (
        bundle
        / "04_final_workspace_score"
        / "subagent"
        / task_dir
        / "candidate.patch"
    ).is_file()
    assert len(copied) == 1
    assert copied[0]["instance_id"] == "task/selected"
