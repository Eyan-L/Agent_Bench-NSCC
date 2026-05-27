"""
RQ1 composition metrics.

Metrics quantify step-type breakdown, tool usage, and token distribution.
All metrics are *measured* (directly derived from logged quantities).
"""

import pandas as pd
import numpy as np


def compute_composition_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composition metrics for a single trajectory.

    Definition:
    - step_count_by_type: count of steps per step_type
    - time_fraction_by_type: fraction of total duration per step_type
    - tool_call_ratio: (TOOL_READ + TOOL_WRITE + TOOL_EXEC) / total_steps
    - non_llm_time_fraction: (total_time - LLM_time) / total_time
    - wait_stall_fraction: WAIT_STALL_time / total_time
    - attempts_per_trajectory: number of distinct attempt_id values
    - input_output_token_ratio: sum(prompt_tokens) / sum(completion_tokens) over LLM steps

    Args:
        steps: DataFrame with step trace for one trajectory

    Returns:
        DataFrame with columns:
        - metric_name: str
        - metric_value: float
        - kind: "measured"
        - params: dict (empty for measured metrics)
    """
    if steps.empty:
        return _empty_metrics_df()

    total_time = steps["duration_ms"].sum()
    total_steps = len(steps)

    metrics = []

    # Step count by type
    for step_type, group in steps.groupby("step_type"):
        metrics.append(
            {
                "metric_name": f"step_count_{step_type}",
                "metric_value": len(group),
                "kind": "measured",
                "params": {},
            }
        )

    # Time fraction by type
    for step_type, group in steps.groupby("step_type"):
        time_frac = group["duration_ms"].sum() / total_time if total_time > 0 else 0.0
        metrics.append(
            {
                "metric_name": f"time_fraction_{step_type}",
                "metric_value": time_frac,
                "kind": "measured",
                "params": {},
            }
        )

    # Tool call ratio
    tool_steps = steps[
        steps["step_type"].isin(["TOOL_READ", "TOOL_WRITE", "TOOL_EXEC"])
    ]
    tool_ratio = len(tool_steps) / total_steps if total_steps > 0 else 0.0
    metrics.append(
        {
            "metric_name": "tool_call_ratio",
            "metric_value": tool_ratio,
            "kind": "measured",
            "params": {},
        }
    )

    # Non-LLM time fraction
    llm_time = steps[steps["step_type"] == "LLM_INFER"]["duration_ms"].sum()
    non_llm_frac = (total_time - llm_time) / total_time if total_time > 0 else 0.0
    metrics.append(
        {
            "metric_name": "non_llm_time_fraction",
            "metric_value": non_llm_frac,
            "kind": "measured",
            "params": {},
        }
    )

    # Wait stall fraction
    stall_time = steps[steps["step_type"] == "WAIT_STALL"]["duration_ms"].sum()
    stall_frac = stall_time / total_time if total_time > 0 else 0.0
    metrics.append(
        {
            "metric_name": "wait_stall_fraction",
            "metric_value": stall_frac,
            "kind": "measured",
            "params": {},
        }
    )

    # Attempts per trajectory
    if "attempt_id" in steps.columns:
        attempts = steps["attempt_id"].nunique()
    else:
        attempts = 1  # Default if not tracked
    metrics.append(
        {
            "metric_name": "attempts_per_trajectory",
            "metric_value": float(attempts),
            "kind": "measured",
            "params": {},
        }
    )

    # Input/output token ratio
    llm_steps = steps[steps["step_type"] == "LLM_INFER"]
    if not llm_steps.empty and "prompt_tokens" in llm_steps.columns:
        total_prompt = llm_steps["prompt_tokens"].sum()
        total_completion = llm_steps["completion_tokens"].sum()
        token_ratio = (
            total_prompt / total_completion if total_completion > 0 else np.nan
        )
    else:
        token_ratio = np.nan

    metrics.append(
        {
            "metric_name": "input_output_token_ratio",
            "metric_value": token_ratio,
            "kind": "measured",
            "params": {},
        }
    )

    return pd.DataFrame(metrics)


def _empty_metrics_df() -> pd.DataFrame:
    """Return empty metrics DataFrame with correct schema."""
    return pd.DataFrame(
        columns=["metric_name", "metric_value", "kind", "params"]
    )
