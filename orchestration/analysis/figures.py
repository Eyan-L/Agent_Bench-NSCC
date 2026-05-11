"""
Generate diagnostic figures for sanity-checking raw data.

Minimal set:
- Step-type stacked bar per condition
- GPU utilization timeline for one sample trajectory per condition
- TTFT-vs-e2e scatter per condition
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .load import load_run


def plot_step_composition(long_df: pd.DataFrame, out_dir: Path):
    """
    Stacked bar chart: step-type time fraction per condition.

    Args:
        long_df: Long-format metrics DataFrame
        out_dir: Output directory for figures
    """
    # Filter time_fraction metrics
    time_frac = long_df[long_df["metric_name"].str.startswith("time_fraction_")]

    if time_frac.empty:
        return

    # Extract step_type from metric_name
    time_frac = time_frac.copy()
    time_frac["step_type"] = time_frac["metric_name"].str.replace(
        "time_fraction_", ""
    )

    # Aggregate by config_id and step_type
    pivot = time_frac.pivot_table(
        index="config_id",
        columns="step_type",
        values="metric_value",
        aggfunc="mean",
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_ylabel("Time Fraction")
    ax.set_xlabel("Condition")
    ax.set_title("Step-Type Composition by Condition")
    ax.legend(title="Step Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = out_dir / "figures" / "step_composition.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved step composition figure to {out_path}")
    plt.close(fig)


def plot_gpu_timeline(long_df: pd.DataFrame, runs_dir: Path, out_dir: Path):
    """
    GPU utilization timeline for one sample trajectory per condition.

    Args:
        long_df: Long-format metrics DataFrame
        runs_dir: Root directory with run data
        out_dir: Output directory for figures
    """
    # Sample one run per config_id
    sampled_runs = long_df.groupby("config_id").first().reset_index()

    fig, axes = plt.subplots(
        len(sampled_runs), 1, figsize=(12, 3 * len(sampled_runs)), sharex=False
    )

    if len(sampled_runs) == 1:
        axes = [axes]

    for idx, row in sampled_runs.iterrows():
        run_id = row["run_id"]
        config_id = row["config_id"]

        try:
            steps, telemetry, verdict = load_run(run_id, runs_dir)

            if telemetry.empty or "gpu_utilization_%" not in telemetry.columns:
                continue

            # Plot GPU utilization over time
            time_s = (telemetry["timestamp_ns"] - telemetry["timestamp_ns"].min()) / 1e9
            gpu_util = telemetry["gpu_utilization_%"]

            axes[idx].plot(time_s, gpu_util, linewidth=0.5)
            axes[idx].set_ylabel("GPU Util (%)")
            axes[idx].set_title(f"{config_id} ({run_id[:8]})")
            axes[idx].grid(alpha=0.3)

        except Exception as e:
            print(f"Error plotting GPU timeline for {run_id}: {e}")
            continue

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    out_path = out_dir / "figures" / "gpu_timeline.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved GPU timeline to {out_path}")
    plt.close(fig)


def plot_ttft_vs_e2e(long_df: pd.DataFrame, out_dir: Path):
    """
    Scatter plot: TTFT vs end-to-end task latency per condition.

    This is the visual version of the ranking-alignment test (RQ3).

    Args:
        long_df: Long-format metrics DataFrame
        out_dir: Output directory for figures
    """
    # Extract TTFT and task_latency
    ttft = long_df[long_df["metric_name"] == "ttft_p50"][
        ["run_id", "config_id", "metric_value"]
    ].rename(columns={"metric_value": "ttft_p50"})

    task_lat = long_df[long_df["metric_name"] == "task_latency"][
        ["run_id", "metric_value"]
    ].rename(columns={"metric_value": "task_latency"})

    # Join
    joined = ttft.merge(task_lat, on="run_id")

    if joined.empty:
        return

    # Plot per config_id
    fig, ax = plt.subplots(figsize=(8, 6))

    for config_id, group in joined.groupby("config_id"):
        ax.scatter(
            group["ttft_p50"],
            group["task_latency"],
            label=config_id,
            alpha=0.6,
            s=50,
        )

    ax.set_xlabel("TTFT p50 (ms)")
    ax.set_ylabel("Task Latency (s)")
    ax.set_title("TTFT vs End-to-End Latency")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "figures" / "ttft_vs_e2e.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved TTFT-vs-e2e scatter to {out_path}")
    plt.close(fig)


def generate_all_figures(long_df: pd.DataFrame, runs_dir: Path, out_dir: Path):
    """
    Generate all diagnostic figures.

    Args:
        long_df: Long-format metrics DataFrame
        runs_dir: Root directory with run data
        out_dir: Output directory for analysis
    """
    print("Generating diagnostic figures...")
    plot_step_composition(long_df, out_dir)
    plot_gpu_timeline(long_df, runs_dir, out_dir)
    plot_ttft_vs_e2e(long_df, out_dir)
    print("Figures complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate diagnostic figures")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Analysis directory with long.parquet",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Root directory with run data",
    )

    args = parser.parse_args()

    long_df = pd.read_parquet(args.analysis_dir / "long.parquet")
    generate_all_figures(long_df, args.runs_dir, args.analysis_dir)
