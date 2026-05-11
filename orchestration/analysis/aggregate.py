"""
Per-condition aggregation with bootstrap confidence intervals.

Entrypoint: python -m orchestration.analysis.aggregate
"""

import argparse
from pathlib import Path
from typing import List
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

from .load import load_run, list_runs
from .composition import compute_composition_metrics
from .resource import compute_resource_metrics
from .evaluation import compute_evaluation_metrics, ranking_alignment_test
from .retry import compute_retry_metrics
from .validation import validate_data_quality


def compute_all_metrics(
    run_id: str,
    runs_dir: Path,
) -> pd.DataFrame:
    """
    Compute all metrics for a single run.

    Returns:
        DataFrame with columns: run_id, metric_name, metric_value, kind, params
    """
    steps, telemetry, verdict = load_run(run_id, runs_dir)

    # Compute metric families
    comp = compute_composition_metrics(steps)
    res = compute_resource_metrics(steps, telemetry)
    eval_m = compute_evaluation_metrics(steps, verdict)
    retry_m = compute_retry_metrics(steps)

    # Combine
    all_metrics = pd.concat([comp, res, eval_m, retry_m], ignore_index=True)

    # Add run metadata
    all_metrics["run_id"] = run_id
    all_metrics["instance_id"] = verdict.get("instance_id")
    all_metrics["config_id"] = verdict.get("config_id")

    return all_metrics


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> tuple:
    """
    Compute bootstrap confidence interval.

    Args:
        values: Array of values
        n_boot: Number of bootstrap samples
        ci: Confidence level (default 0.95)

    Returns:
        (lower, upper) bounds of CI
    """
    if len(values) < 2:
        return (np.nan, np.nan)

    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)

    return (lower, upper)


def aggregate_per_condition(
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate metrics per condition with bootstrap CIs.

    Args:
        long_df: Long-format DataFrame with all runs

    Returns:
        DataFrame with columns:
            - config_id
            - metric_name
            - mean
            - median
            - p95
            - p99
            - stddev
            - ci_lower
            - ci_upper
            - n_runs
    """
    agg_results = []

    for (config_id, metric_name), group in long_df.groupby(["config_id", "metric_name"]):
        values = group["metric_value"].dropna().values

        if len(values) == 0:
            continue

        ci_lower, ci_upper = bootstrap_ci(values)

        agg_results.append(
            {
                "config_id": config_id,
                "metric_name": metric_name,
                "mean": np.mean(values),
                "median": np.median(values),
                "p95": np.percentile(values, 95) if len(values) > 1 else values[0],
                "p99": np.percentile(values, 99) if len(values) > 1 else values[0],
                "stddev": np.std(values) if len(values) > 1 else 0.0,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_runs": len(values),
            }
        )

    return pd.DataFrame(agg_results)


def main():
    parser = argparse.ArgumentParser(description="Aggregate trace metrics")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing run data (data/runs/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for analysis results (data/analysis/)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI (default: 1000)",
    )

    args = parser.parse_args()

    runs_dir = args.runs_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # List all runs
    print("Listing runs...")
    runs = list_runs(runs_dir)
    print(f"Found {len(runs)} runs")

    # Compute metrics for all runs
    print("Computing metrics per run...")
    all_metrics = []

    for _, row in tqdm(runs.iterrows(), total=len(runs)):
        run_id = row["run_id"]
        try:
            metrics = compute_all_metrics(run_id, runs_dir)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {run_id}: {e}")
            continue

    # Combine into long format
    long_df = pd.concat(all_metrics, ignore_index=True)

    # Save long format
    long_path = out_dir / "long.parquet"
    long_df.to_parquet(long_path, index=False)
    print(f"Saved long-format metrics to {long_path}")

    # Aggregate per condition
    print("Aggregating per condition...")
    agg_df = aggregate_per_condition(long_df)

    agg_path = out_dir / "aggregated.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated metrics to {agg_path}")

    # Run data quality validation
    print("Running data quality checks...")
    quality_report = validate_data_quality(long_df, runs_dir)

    quality_path = out_dir / "data_quality.md"
    with open(quality_path, "w") as f:
        f.write(quality_report)
    print(f"Saved data quality report to {quality_path}")

    print("Aggregation complete.")


if __name__ == "__main__":
    main()
