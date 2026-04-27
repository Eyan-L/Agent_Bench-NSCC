"""
Retry and fragmentation metrics.

Metrics quantify wasted work from retries and execution fragmentation.
Some are inferred (depend on burst-window parameter T).
"""

import pandas as pd
import numpy as np


def compute_retry_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute retry and fragmentation metrics for a single trajectory.

    Definition:
    - failed_attempt_cost_fraction: duration of non-final attempts / total duration (measured)
    - wasted_resource_fraction: (GPU+CPU time of non-final attempts) / total (measured, approx)
    - llm_burst_count: number of consecutive LLM_INFER runs separated by > T (inferred, T=1s)
    - llm_burst_duration_mean: mean duration of LLM bursts (inferred, T=1s)
    - inter_step_idle_gap_mean: mean idle time between consecutive steps (measured)
    - inter_step_idle_gap_p95: 95th percentile idle gap (measured)

    Args:
        steps: Step trace for one trajectory

    Returns:
        DataFrame with columns: metric_name, metric_value, kind, params
    """
    if steps.empty:
        return _empty_metrics_df()

    metrics = []

    # Failed attempt cost (measured, requires attempt_id and attempt outcome)
    if "attempt_id" in steps.columns and "wasted_step" in steps.columns:
        total_time = steps["duration_ms"].sum()
        wasted_time = steps[steps["wasted_step"]]["duration_ms"].sum()
        failed_frac = wasted_time / total_time if total_time > 0 else 0.0
        metrics.append(
            {
                "metric_name": "failed_attempt_cost_fraction",
                "metric_value": failed_frac,
                "kind": "measured",
                "params": {},
            }
        )
    else:
        metrics.append(
            {
                "metric_name": "failed_attempt_cost_fraction",
                "metric_value": np.nan,
                "kind": "measured",
                "params": {"reason": "attempt_tracking_unavailable"},
            }
        )

    # Wasted resource fraction (measured, approximation)
    # Simplified: assume resource usage proportional to time
    # Real version needs GPU-time and CPU-time alignment
    if "attempt_id" in steps.columns and "wasted_step" in steps.columns:
        metrics.append(
            {
                "metric_name": "wasted_resource_fraction",
                "metric_value": failed_frac,  # Same as time fraction in this approx
                "kind": "measured",
                "params": {"note": "approximation_time_proportional"},
            }
        )
    else:
        metrics.append(
            {
                "metric_name": "wasted_resource_fraction",
                "metric_value": np.nan,
                "kind": "measured",
                "params": {"reason": "attempt_tracking_unavailable"},
            }
        )

    # LLM burst count and duration (inferred, T=1s)
    burst_window_s = 1.0
    llm_steps = steps[steps["step_type"] == "LLM_INFER"].copy()

    if not llm_steps.empty:
        llm_steps = llm_steps.sort_values("start_ts")
        gaps = (
            llm_steps["start_ts"].diff() / 1e9
        ).fillna(0)  # Gap in seconds from previous LLM step

        # Burst starts when gap > threshold
        burst_starts = gaps > burst_window_s
        burst_ids = burst_starts.cumsum()

        llm_steps["burst_id"] = burst_ids

        burst_count = llm_steps["burst_id"].nunique()
        burst_durations = []
        for bid, group in llm_steps.groupby("burst_id"):
            burst_start = group["start_ts"].min()
            burst_end = group["end_ts"].max()
            burst_durations.append((burst_end - burst_start) / 1e9)

        burst_duration_mean = np.mean(burst_durations) if burst_durations else 0.0

        metrics.append(
            {
                "metric_name": "llm_burst_count",
                "metric_value": float(burst_count),
                "kind": "inferred",
                "params": {"burst_window_seconds": burst_window_s},
            }
        )
        metrics.append(
            {
                "metric_name": "llm_burst_duration_mean",
                "metric_value": burst_duration_mean,
                "kind": "inferred",
                "params": {"burst_window_seconds": burst_window_s},
            }
        )
    else:
        metrics.append(
            {
                "metric_name": "llm_burst_count",
                "metric_value": 0.0,
                "kind": "inferred",
                "params": {"burst_window_seconds": burst_window_s},
            }
        )
        metrics.append(
            {
                "metric_name": "llm_burst_duration_mean",
                "metric_value": 0.0,
                "kind": "inferred",
                "params": {"burst_window_seconds": burst_window_s},
            }
        )

    # Inter-step idle gaps (measured)
    steps_sorted = steps.sort_values("start_ts")
    idle_gaps = (
        steps_sorted["start_ts"] - steps_sorted["end_ts"].shift()
    ) / 1e6  # ms

    # Filter out negative gaps (overlapping steps) and NaN (first step)
    idle_gaps = idle_gaps[idle_gaps > 0].dropna()

    if not idle_gaps.empty:
        metrics.append(
            {
                "metric_name": "inter_step_idle_gap_mean",
                "metric_value": idle_gaps.mean(),
                "kind": "measured",
                "params": {},
            }
        )
        metrics.append(
            {
                "metric_name": "inter_step_idle_gap_p95",
                "metric_value": idle_gaps.quantile(0.95),
                "kind": "measured",
                "params": {},
            }
        )
    else:
        for stat in ["mean", "p95"]:
            metrics.append(
                {
                    "metric_name": f"inter_step_idle_gap_{stat}",
                    "metric_value": 0.0,
                    "kind": "measured",
                    "params": {},
                }
            )

    return pd.DataFrame(metrics)


def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["metric_name", "metric_value", "kind", "params"])
