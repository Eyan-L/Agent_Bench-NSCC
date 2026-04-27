"""
Load step traces, telemetry, and verdicts keyed by run_id.

Functions return DataFrames with standardized schemas.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import pyarrow.parquet as pq


def load_run(
    run_id: str,
    runs_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load step trace, telemetry, and verdict for a single run.

    Args:
        run_id: Unique run identifier
        runs_dir: Root directory containing run data (data/runs/)

    Returns:
        (steps, telemetry, verdict) where:
        - steps: DataFrame with columns matching raw trace schema
        - telemetry: DataFrame with timestamp_ns and resource columns
        - verdict: Dict with instance_id, status, test_pass_rate, etc.

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If schema validation fails
    """
    run_dir = runs_dir / run_id

    # Load step trace
    steps_path = run_dir / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Step trace not found: {steps_path}")

    steps = []
    with open(steps_path, "r") as f:
        for line in f:
            steps.append(json.loads(line))

    steps_df = pd.DataFrame(steps)
    _validate_steps_schema(steps_df, run_id)

    # Load telemetry (Parquet or CSV fallback)
    telemetry_parquet = run_dir / "telemetry.parquet"
    telemetry_csv = run_dir / "telemetry.csv"

    if telemetry_parquet.exists():
        telemetry_df = pd.read_parquet(telemetry_parquet)
    elif telemetry_csv.exists():
        telemetry_df = pd.read_csv(telemetry_csv)
    else:
        # Missing telemetry is allowed but flagged
        telemetry_df = pd.DataFrame({"timestamp_ns": [], "missing": [True]})

    if not telemetry_df.empty:
        _validate_telemetry_schema(telemetry_df, run_id)

    # Load verdict
    verdict_path = run_dir / "verdict.json"
    if not verdict_path.exists():
        raise FileNotFoundError(f"Verdict not found: {verdict_path}")

    with open(verdict_path, "r") as f:
        verdict = json.load(f)

    return steps_df, telemetry_df, verdict


def _validate_steps_schema(df: pd.DataFrame, run_id: str):
    """Validate step trace schema."""
    required = [
        "trajectory_id",
        "step_id",
        "step_type",
        "start_ts",
        "end_ts",
        "duration_ms",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Step trace for {run_id} missing required columns: {missing}"
        )

    # Validate step types
    valid_types = {"LLM_INFER", "TOOL_READ", "TOOL_WRITE", "TOOL_EXEC", "WAIT_STALL"}
    invalid = set(df["step_type"].unique()) - valid_types
    if invalid:
        raise ValueError(
            f"Step trace for {run_id} contains invalid step_type values: {invalid}"
        )


def _validate_telemetry_schema(df: pd.DataFrame, run_id: str):
    """Validate telemetry schema."""
    if "timestamp_ns" not in df.columns:
        raise ValueError(
            f"Telemetry for {run_id} missing required column: timestamp_ns"
        )


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    """Load manifest.json for a run (optional metadata)."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r") as f:
        return json.load(f)


def list_runs(runs_dir: Path) -> pd.DataFrame:
    """
    List all runs with their metadata.

    Returns:
        DataFrame with columns: run_id, instance_id, config_id, model_name, repo_id
    """
    runs = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        manifest = load_manifest(run_dir)

        # Try to extract metadata from verdict if manifest missing
        verdict_path = run_dir / "verdict.json"
        if verdict_path.exists():
            with open(verdict_path, "r") as f:
                verdict = json.load(f)
                runs.append(
                    {
                        "run_id": run_id,
                        "instance_id": verdict.get("instance_id"),
                        "config_id": manifest.get("config_id") or verdict.get("config_id"),
                        "model_name": manifest.get("model_name"),
                        "repo_id": manifest.get("repo_id"),
                    }
                )

    return pd.DataFrame(runs)
