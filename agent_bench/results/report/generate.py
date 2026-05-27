"""
Report generator entrypoint.

Usage:
    python -m agent_bench.results.report.generate --analysis-dir data/analysis --out-dir report
"""

import argparse
from pathlib import Path
import json
import pandas as pd

from .sections import (
    generate_abstract,
    generate_introduction,
    generate_background,
    generate_methodology,
    generate_rq1_results,
    generate_rq2_results,
    generate_rq3_results,
    generate_discussion,
    generate_limitations,
    generate_conclusion,
)
from .claims import ClaimRegistry
from .pushback import generate_pushback


def load_analysis_data(analysis_dir: Path) -> dict:
    """
    Load all analysis artifacts.

    Args:
        analysis_dir: Path to analysis directory with long.parquet, tables/, figures/

    Returns:
        Dictionary with:
            - long_df: Long-format Parquet
            - tables: Dict of metric family -> DataFrame
            - figures: Dict of figure name -> Path
            - strata: Stratification report text (if exists)
            - data_quality: Data quality report text (if exists)
    """
    data = {}

    # Load long format
    long_path = analysis_dir / "long.parquet"
    if not long_path.exists():
        raise FileNotFoundError(f"Long-format data not found: {long_path}")
    data["long_df"] = pd.read_parquet(long_path)

    # Load aggregated data
    agg_path = analysis_dir / "aggregated.csv"
    if agg_path.exists():
        data["aggregated_df"] = pd.read_csv(agg_path)
    else:
        data["aggregated_df"] = None

    # Load wide tables
    tables_dir = analysis_dir / "tables"
    data["tables"] = {}
    if tables_dir.exists():
        for table_file in tables_dir.glob("*_wide.csv"):
            family_name = table_file.stem.replace("_wide", "")
            data["tables"][family_name] = pd.read_csv(table_file, index_col=0)

    # Collect figure paths
    figures_dir = analysis_dir / "figures"
    data["figures"] = {}
    if figures_dir.exists():
        for fig_file in figures_dir.glob("*.png"):
            data["figures"][fig_file.stem] = fig_file

    # Load optional reports
    strata_path = analysis_dir / "strata.md"
    if strata_path.exists():
        data["strata"] = strata_path.read_text()
    else:
        data["strata"] = None

    quality_path = analysis_dir / "data_quality.md"
    if quality_path.exists():
        data["data_quality"] = quality_path.read_text()
    else:
        data["data_quality"] = None

    return data


def generate_report(analysis_dir: Path, out_dir: Path, plan_path: Path = None):
    """
    Generate full research report.

    Args:
        analysis_dir: Analysis directory with metrics and figures
        out_dir: Output directory for report files
        plan_path: Optional path to plan.md for methodology reference
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading analysis data...")
    data = load_analysis_data(analysis_dir)

    # Initialize claim registry
    claims = ClaimRegistry()

    # Load plan if provided
    plan_text = None
    if plan_path and plan_path.exists():
        plan_text = plan_path.read_text()

    print("Generating sections...")

    sections = []

    sections.append(("# Abstract\n\n", generate_abstract(data, claims)))
    sections.append(("# 1. Introduction\n\n", generate_introduction(data, claims)))
    sections.append(("# 2. Background\n\n", generate_background(data, claims)))
    sections.append(
        ("# 3. Methodology\n\n", generate_methodology(data, claims, plan_text))
    )
    sections.append(
        ("# 4. Results: RQ1 — Execution Composition\n\n", generate_rq1_results(data, claims))
    )
    sections.append(
        ("# 5. Results: RQ2 — Resource Bottlenecks\n\n", generate_rq2_results(data, claims))
    )
    sections.append(
        ("# 6. Results: RQ3 — Evaluation Metric Sufficiency\n\n", generate_rq3_results(data, claims))
    )
    sections.append(("# 7. Discussion\n\n", generate_discussion(data, claims)))
    sections.append(
        ("# 8. Limitations and Threats to Validity\n\n", generate_limitations(data, claims))
    )
    sections.append(("# 9. Conclusion\n\n", generate_conclusion(data, claims)))

    # Assemble report
    report_md = "\n".join([header + body for header, body in sections])

    # Write report
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"Wrote report to {report_path}")

    # Write claims.json
    claims_path = out_dir / "claims.json"
    with open(claims_path, "w") as f:
        json.dump(claims.to_dict(), f, indent=2)
    print(f"Wrote claims registry to {claims_path}")

    # Generate pushback.md
    print("Generating pushback analysis...")
    pushback_text = generate_pushback(data, claims, report_md)
    pushback_path = out_dir / "pushback.md"
    with open(pushback_path, "w") as f:
        f.write(pushback_text)
    print(f"Wrote pushback to {pushback_path}")

    # Check for unsupported claims
    unsupported = claims.get_unsupported()
    if unsupported:
        todo_path = out_dir / "TODO_CLAIMS.md"
        with open(todo_path, "w") as f:
            f.write("# Unsupported Claims\n\n")
            f.write(
                "The following claims were flagged as unsupported by available evidence:\n\n"
            )
            for claim in unsupported:
                f.write(f"- **{claim['claim_text']}**\n")
                f.write(f"  - Reason: {claim['reason']}\n\n")
        print(f"Wrote unsupported claims to {todo_path}")

    print("Report generation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate research report from analyzed metrics"
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Analysis directory with long.parquet, tables/, figures/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for report files",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        help="Optional path to plan.md for methodology reference",
    )

    args = parser.parse_args()

    generate_report(args.analysis_dir, args.out_dir, args.plan)


if __name__ == "__main__":
    main()
