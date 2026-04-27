"""
Generate wide CSV tables per metric family.

Tables are easier to inspect than long-format Parquet.
"""

import pandas as pd
from pathlib import Path


def generate_tables(long_df: pd.DataFrame, out_dir: Path):
    """
    Generate wide CSV tables per metric family.

    Args:
        long_df: Long-format metrics DataFrame
        out_dir: Output directory for tables
    """
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Define metric families
    families = {
        "composition": [
            "step_count",
            "time_fraction",
            "tool_call_ratio",
            "non_llm_time_fraction",
            "wait_stall_fraction",
            "attempts_per_trajectory",
            "input_output_token_ratio",
        ],
        "resource": [
            "gpu_utilization",
            "gpu_idle_fraction",
            "sm_occupancy",
            "hbm_bandwidth",
            "mfu",
            "mbu",
            "cpu_utilization",
            "subprocess_cpu_time",
            "rss_peak_MB",
            "rss_mean_MB",
            "pcio_total_GB",
        ],
        "evaluation": [
            "ttft",
            "tpot",
            "task_latency",
            "success",
        ],
        "retry": [
            "failed_attempt_cost_fraction",
            "wasted_resource_fraction",
            "llm_burst_count",
            "llm_burst_duration_mean",
            "inter_step_idle_gap",
        ],
    }

    for family_name, prefixes in families.items():
        # Filter metrics matching family prefixes
        family_metrics = long_df[
            long_df["metric_name"].apply(
                lambda x: any(x.startswith(p) for p in prefixes)
            )
        ]

        if family_metrics.empty:
            continue

        # Pivot to wide format: rows = config_id, cols = metric_name
        wide = family_metrics.pivot_table(
            index=["config_id"],
            columns="metric_name",
            values="metric_value",
            aggfunc="mean",
        )

        # Save
        out_path = tables_dir / f"{family_name}_wide.csv"
        wide.to_csv(out_path)
        print(f"Saved {family_name} table to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate wide CSV tables")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Analysis directory with long.parquet",
    )

    args = parser.parse_args()

    long_df = pd.read_parquet(args.analysis_dir / "long.parquet")
    generate_tables(long_df, args.analysis_dir)
