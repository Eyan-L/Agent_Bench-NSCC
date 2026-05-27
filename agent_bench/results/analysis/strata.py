"""
Post-hoc stratification from trace-derived features.

Stratify tasks by observed trajectory properties:
- search-heavy: >50% TOOL_READ time
- edit-test-loop: ≥3 attempts
- retry-heavy: ≥5 attempts
- llm-dominated: >70% LLM_INFER time
"""

import pandas as pd
from pathlib import Path


def stratify_runs(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign strata labels to each run based on composition metrics.

    Args:
        long_df: Long-format metrics DataFrame

    Returns:
        DataFrame with columns: run_id, strata (list of strata labels)
    """
    strata_results = []

    for run_id, group in long_df.groupby("run_id"):
        strata = []

        # Search-heavy: >50% TOOL_READ time
        tool_read_frac = group[group["metric_name"] == "time_fraction_TOOL_READ"][
            "metric_value"
        ].values
        if len(tool_read_frac) > 0 and tool_read_frac[0] > 0.5:
            strata.append("search-heavy")

        # Edit-test-loop: ≥3 attempts
        attempts = group[group["metric_name"] == "attempts_per_trajectory"][
            "metric_value"
        ].values
        if len(attempts) > 0 and attempts[0] >= 3:
            strata.append("edit-test-loop")

        # Retry-heavy: ≥5 attempts
        if len(attempts) > 0 and attempts[0] >= 5:
            strata.append("retry-heavy")

        # LLM-dominated: >70% LLM_INFER time
        llm_frac = group[group["metric_name"] == "time_fraction_LLM_INFER"][
            "metric_value"
        ].values
        if len(llm_frac) > 0 and llm_frac[0] > 0.7:
            strata.append("llm-dominated")

        strata_results.append(
            {
                "run_id": run_id,
                "strata": ",".join(strata) if strata else "none",
            }
        )

    return pd.DataFrame(strata_results)


def write_strata_doc(strata_df: pd.DataFrame, out_path: Path):
    """
    Write strata definitions and distribution to markdown.

    Args:
        strata_df: DataFrame with run_id and strata columns
        out_path: Output path for strata.md
    """
    lines = ["# Stratification Definitions", ""]

    lines.append("## Strata Criteria (Trace-Derived)")
    lines.append("")
    lines.append("- **search-heavy**: >50% of trajectory time in TOOL_READ steps")
    lines.append("- **edit-test-loop**: ≥3 attempts per trajectory")
    lines.append("- **retry-heavy**: ≥5 attempts per trajectory")
    lines.append("- **llm-dominated**: >70% of trajectory time in LLM_INFER steps")
    lines.append("")

    lines.append("## Distribution")
    lines.append("")

    strata_counts = strata_df["strata"].value_counts()
    lines.append("| Strata | Count |")
    lines.append("|--------|-------|")
    for strata, count in strata_counts.items():
        lines.append(f"| {strata} | {count} |")

    lines.append("")
    lines.append(f"Total runs: {len(strata_df)}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stratify runs")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Analysis directory with long.parquet",
    )

    args = parser.parse_args()

    long_df = pd.read_parquet(args.analysis_dir / "long.parquet")
    strata_df = stratify_runs(long_df)

    out_path = args.analysis_dir / "strata.md"
    write_strata_doc(strata_df, out_path)
    print(f"Saved strata definitions to {out_path}")

    # Also save strata assignments
    strata_df.to_csv(args.analysis_dir / "strata.csv", index=False)
    print(f"Saved strata assignments to {args.analysis_dir / 'strata.csv'}")
