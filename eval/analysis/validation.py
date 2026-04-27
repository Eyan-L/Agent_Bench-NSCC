"""
Data quality validation checks.

Runs automatically on every aggregate invocation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from .load import load_run


def validate_data_quality(long_df: pd.DataFrame, runs_dir: Path) -> str:
    """
    Run data quality checks and produce a markdown report.

    Checks:
    1. Sum of step-type durations ≈ trajectory wall time (within 1%)
    2. AR trajectory wall time ≈ LLM_INFER duration
    3. OpenClaw non-LLM time fraction > 0
    4. Telemetry coverage ≥ 95% per trajectory

    Args:
        long_df: Long-format metrics DataFrame
        runs_dir: Root directory containing run data

    Returns:
        Markdown-formatted report string
    """
    report_lines = ["# Data Quality Report", ""]

    # Get unique run_ids
    run_ids = long_df["run_id"].unique()

    check_results = {
        "step_duration_sum": [],
        "ar_llm_time": [],
        "openclaw_non_llm": [],
        "telemetry_coverage": [],
    }

    for run_id in run_ids:
        run_data = long_df[long_df["run_id"] == run_id]
        config_id = run_data["config_id"].iloc[0] if not run_data.empty else None

        try:
            from .load import load_run

            steps, telemetry, verdict = load_run(run_id, runs_dir)

            # Check 1: Step duration sum ≈ trajectory wall time
            total_step_time = steps["duration_ms"].sum()
            wall_time = (steps["end_ts"].max() - steps["start_ts"].min()) / 1e6  # ms

            if wall_time > 0:
                duration_ratio = total_step_time / wall_time
                check_results["step_duration_sum"].append(
                    {
                        "run_id": run_id,
                        "ratio": duration_ratio,
                        "pass": 0.99 <= duration_ratio <= 1.01,
                    }
                )

            # Check 2: AR trajectory wall time ≈ LLM time
            if config_id and "ar" in config_id.lower():
                llm_time = steps[steps["step_type"] == "LLM_INFER"]["duration_ms"].sum()
                if wall_time > 0:
                    llm_ratio = llm_time / wall_time
                    check_results["ar_llm_time"].append(
                        {
                            "run_id": run_id,
                            "ratio": llm_ratio,
                            "pass": llm_ratio > 0.90,  # Allow 10% overhead
                        }
                    )

            # Check 3: OpenClaw non-LLM time > 0
            if config_id and "openclaw" in config_id.lower():
                llm_time = steps[steps["step_type"] == "LLM_INFER"]["duration_ms"].sum()
                non_llm_frac = (wall_time - llm_time) / wall_time if wall_time > 0 else 0
                check_results["openclaw_non_llm"].append(
                    {
                        "run_id": run_id,
                        "non_llm_frac": non_llm_frac,
                        "pass": non_llm_frac > 0,
                    }
                )

            # Check 4: Telemetry coverage
            if not telemetry.empty and "timestamp_ns" in telemetry.columns:
                telem_span = telemetry["timestamp_ns"].max() - telemetry["timestamp_ns"].min()
                traj_span = steps["end_ts"].max() - steps["start_ts"].min()
                coverage = telem_span / traj_span if traj_span > 0 else 0
                check_results["telemetry_coverage"].append(
                    {
                        "run_id": run_id,
                        "coverage": coverage,
                        "pass": coverage >= 0.95,
                    }
                )

        except Exception as e:
            report_lines.append(f"⚠️  Error validating {run_id}: {e}")
            continue

    # Summarize results
    report_lines.append("## Summary")
    report_lines.append("")

    for check_name, results in check_results.items():
        if not results:
            continue

        passed = sum(1 for r in results if r["pass"])
        total = len(results)
        pass_rate = passed / total if total > 0 else 0

        status = "✅" if pass_rate >= 0.95 else "⚠️"
        report_lines.append(
            f"{status} **{check_name}**: {passed}/{total} runs passed ({pass_rate:.1%})"
        )

    # Detail failures
    report_lines.append("")
    report_lines.append("## Failed Checks")
    report_lines.append("")

    for check_name, results in check_results.items():
        failures = [r for r in results if not r["pass"]]
        if failures:
            report_lines.append(f"### {check_name}")
            report_lines.append("")
            for f in failures[:10]:  # Show first 10 failures
                report_lines.append(f"- `{f['run_id']}`: {f}")
            if len(failures) > 10:
                report_lines.append(f"- ... and {len(failures) - 10} more")
            report_lines.append("")

    return "\n".join(report_lines)
