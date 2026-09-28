"""Render native Codex active turns without counting idle child lifetimes."""
from __future__ import annotations


def install_comparison(comparison):
    if getattr(comparison, "_codex_turn_intervals_installed", False):
        return
    original = comparison._child_intervals
    original_record = comparison.record_to_dict

    def intervals(record):
        obs, _ = comparison._runtime_observability(record)
        if obs.get("source") != "codex_native_rollouts":
            return original(record)
        rows = []
        for key, child in (obs.get("child_runtime_by_session") or {}).items():
            for index, turn in enumerate(child.get("turn_intervals") or []):
                rows.append({
                    "session_key": key, "session_id": key, "label": f"{key} / turn {index + 1}",
                    "status": turn["status"], "start_timestamp_ns": turn["start_timestamp_ns"],
                    "end_timestamp_ns": turn["end_timestamp_ns"], "terminal_timestamp_ns": turn["end_timestamp_ns"],
                    "runtime_seconds": (turn["end_timestamp_ns"] - turn["start_timestamp_ns"]) / 1e9,
                    "runtime_complete": child["runtime_complete"], "runtime_censored": child["runtime_censored"],
                    "start_timestamp_source": "codex_rollout.task_started", "start_timestamp_quality": "native_event",
                    "end_timestamp_source": "codex_rollout.task_complete_or_turn_aborted", "end_timestamp_quality": "native_event",
                })
        return sorted(rows, key=lambda row: (row["start_timestamp_ns"], row["session_key"]))

    def record_to_dict(record):
        row = original_record(record)
        obs, _ = comparison._runtime_observability(record)
        if obs.get("source") == "codex_native_rollouts":
            for name in (
                "spawned_child_count", "missing_child_session_ids",
                "parent_activity_during_child_execution", "parent_overlapping_model_calls",
                "parent_overlapping_tool_calls", "parent_overlapping_coordination_calls",
            ):
                row[name] = obs.get(name)
        return row

    comparison._child_intervals = intervals
    comparison.record_to_dict = record_to_dict
    if hasattr(comparison, "_impl"):
        comparison._impl.record_to_dict = record_to_dict
    comparison._codex_turn_intervals_installed = True


def install_analysis(analysis):
    install_comparison(analysis.base)
