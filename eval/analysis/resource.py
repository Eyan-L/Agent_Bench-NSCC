"""
RQ2 resource utilization metrics.

Metrics computed from aligned telemetry time-series.
Some metrics are *inferred* (depend on thresholds or assumptions).
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_resource_metrics(
    steps: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute resource metrics for a single trajectory.

    Definition:
    - gpu_utilization_mean: mean GPU utilization over trajectory wall time (measured)
    - gpu_utilization_p95: 95th percentile GPU utilization (measured)
    - gpu_idle_fraction: fraction of time GPU util < threshold (inferred, threshold=10%)
    - sm_occupancy_mean: mean SM occupancy if available, else NaN (measured)
    - hbm_bandwidth_mean: mean HBM bandwidth if available, else NaN (measured)
    - mfu: Model FLOPs Utilization if available, else NaN (measured)
    - mbu: Memory Bandwidth Utilization if available, else NaN (measured)
    - cpu_utilization_mean: mean CPU utilization (measured)
    - cpu_utilization_p95: 95th percentile CPU utilization (measured)
    - subprocess_cpu_time: total CPU time in subprocesses (measured)
    - rss_peak_MB: peak resident set size (measured)
    - rss_mean_MB: time-averaged RSS (measured)
    - pcio_total_GB: total PCIe/disk I/O if available (measured)

    Args:
        steps: Step trace for one trajectory
        telemetry: Telemetry time-series aligned to trajectory

    Returns:
        DataFrame with columns: metric_name, metric_value, kind, params
    """
    if steps.empty:
        return _empty_metrics_df()

    # Trajectory time bounds
    traj_start = steps["start_ts"].min()
    traj_end = steps["end_ts"].max()

    # Filter telemetry to trajectory window
    telem = telemetry[
        (telemetry["timestamp_ns"] >= traj_start)
        & (telemetry["timestamp_ns"] <= traj_end)
    ].copy()

    metrics = []

    # GPU utilization (measured)
    if "gpu_utilization_%" in telem.columns and not telem.empty:
        gpu_util = telem["gpu_utilization_%"]
        metrics.append(
            {
                "metric_name": "gpu_utilization_mean",
                "metric_value": gpu_util.mean(),
                "kind": "measured",
                "params": {},
            }
        )
        metrics.append(
            {
                "metric_name": "gpu_utilization_p95",
                "metric_value": gpu_util.quantile(0.95),
                "kind": "measured",
                "params": {},
            }
        )

        # GPU idle fraction (inferred, threshold-based)
        idle_threshold = 10.0  # %
        idle_frac = (gpu_util < idle_threshold).mean()
        metrics.append(
            {
                "metric_name": "gpu_idle_fraction",
                "metric_value": idle_frac,
                "kind": "inferred",
                "params": {"idle_threshold_percent": idle_threshold},
            }
        )
    else:
        # GPU metrics unavailable
        for name in ["gpu_utilization_mean", "gpu_utilization_p95", "gpu_idle_fraction"]:
            metrics.append(
                {
                    "metric_name": name,
                    "metric_value": np.nan,
                    "kind": "measured" if "idle" not in name else "inferred",
                    "params": {"reason": "counter_unavailable"},
                }
            )

    # SM occupancy, MFU, MBU, HBM bandwidth (measured, but often unavailable)
    for col, metric_name in [
        ("sm_occupancy_%", "sm_occupancy_mean"),
        ("mfu_%", "mfu_mean"),
        ("mbu_%", "mbu_mean"),
        ("hbm_bandwidth_GB/s", "hbm_bandwidth_mean"),
    ]:
        if col in telem.columns and not telem.empty:
            metrics.append(
                {
                    "metric_name": metric_name,
                    "metric_value": telem[col].mean(),
                    "kind": "measured",
                    "params": {},
                }
            )
        else:
            metrics.append(
                {
                    "metric_name": metric_name,
                    "metric_value": np.nan,
                    "kind": "measured",
                    "params": {"reason": "counter_unavailable"},
                }
            )

    # CPU utilization (measured)
    if "cpu_utilization_%" in telem.columns and not telem.empty:
        cpu_util = telem["cpu_utilization_%"]
        metrics.append(
            {
                "metric_name": "cpu_utilization_mean",
                "metric_value": cpu_util.mean(),
                "kind": "measured",
                "params": {},
            }
        )
        metrics.append(
            {
                "metric_name": "cpu_utilization_p95",
                "metric_value": cpu_util.quantile(0.95),
                "kind": "measured",
                "params": {},
            }
        )
    else:
        for name in ["cpu_utilization_mean", "cpu_utilization_p95"]:
            metrics.append(
                {
                    "metric_name": name,
                    "metric_value": np.nan,
                    "kind": "measured",
                    "params": {"reason": "counter_unavailable"},
                }
            )

    # Subprocess CPU time (measured, from steps if available)
    if "subprocess_cpu_time_s" in telem.columns and not telem.empty:
        subprocess_cpu = telem["subprocess_cpu_time_s"].max()  # Cumulative
        metrics.append(
            {
                "metric_name": "subprocess_cpu_time",
                "metric_value": subprocess_cpu,
                "kind": "measured",
                "params": {},
            }
        )
    else:
        metrics.append(
            {
                "metric_name": "subprocess_cpu_time",
                "metric_value": np.nan,
                "kind": "measured",
                "params": {"reason": "counter_unavailable"},
            }
        )

    # Memory (measured)
    if "rss_MB" in telem.columns and not telem.empty:
        rss = telem["rss_MB"]
        metrics.append(
            {
                "metric_name": "rss_peak_MB",
                "metric_value": rss.max(),
                "kind": "measured",
                "params": {},
            }
        )
        metrics.append(
            {
                "metric_name": "rss_mean_MB",
                "metric_value": rss.mean(),
                "kind": "measured",
                "params": {},
            }
        )
    else:
        for name in ["rss_peak_MB", "rss_mean_MB"]:
            metrics.append(
                {
                    "metric_name": name,
                    "metric_value": np.nan,
                    "kind": "measured",
                    "params": {"reason": "counter_unavailable"},
                }
            )

    # PCIe / network I/O (measured, often unavailable)
    if "pcio_read_MB" in telem.columns and "pcio_write_MB" in telem.columns:
        pcio_total = (
            telem["pcio_read_MB"].sum() + telem["pcio_write_MB"].sum()
        ) / 1024.0  # GB
        metrics.append(
            {
                "metric_name": "pcio_total_GB",
                "metric_value": pcio_total,
                "kind": "measured",
                "params": {},
            }
        )
    else:
        metrics.append(
            {
                "metric_name": "pcio_total_GB",
                "metric_value": np.nan,
                "kind": "measured",
                "params": {"reason": "counter_unavailable"},
            }
        )

    return pd.DataFrame(metrics)


def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["metric_name", "metric_value", "kind", "params"])
