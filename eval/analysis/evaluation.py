"""
RQ3 evaluation metrics.

Metrics for token-level latency (TTFT, TPOT) and task-level latency/success.
Includes ranking-alignment test (core RQ3 probe).
"""

import pandas as pd
import numpy as np
# scipy is only needed by ranking_alignment_test; importing it lazily lets
# compute_evaluation_metrics run on lean envs (e.g. minimal plotting boxes)
# that haven't installed scipy.


def compute_evaluation_metrics(
    steps: pd.DataFrame,
    verdict: dict,
) -> pd.DataFrame:
    """
    Compute evaluation metrics for a single trajectory.

    Definition:
    - ttft_mean, ttft_p50, ttft_p95, ttft_p99: Time to first token per LLM span (measured)
    - tpot_mean, tpot_p50, tpot_p95, tpot_p99: Time per output token per LLM span (measured)
    - task_latency: End-to-end trajectory wall time (measured)
    - success: Binary 0/1 for task success (measured)
    - success_under_300s, success_under_600s, success_under_1800s: Success within latency budget (measured)

    Args:
        steps: Step trace for one trajectory
        verdict: Verdict dict with status, test_pass_rate, etc.

    Returns:
        DataFrame with columns: metric_name, metric_value, kind, params
    """
    if steps.empty:
        return _empty_metrics_df()

    metrics = []

    # Extract LLM spans
    llm_steps = steps[steps["step_type"] == "LLM_INFER"].copy()

    # TTFT and TPOT (measured from LLM spans)
    # Note: TTFT/TPOT require instrumentation that logs these per-span
    # If not available in trace, compute approximations from duration and tokens
    if not llm_steps.empty:
        # TTFT: assume first chunk latency if logged, else approximate from duration
        if "ttft_ms" in llm_steps.columns:
            ttft = llm_steps["ttft_ms"]
        else:
            # Approximation: duration / (1 + completion_tokens) * 1
            # This is a rough heuristic; real TTFT needs instrumentation
            ttft = llm_steps["duration_ms"] / (
                1 + llm_steps["completion_tokens"].fillna(1)
            )

        for stat, func in [
            ("mean", np.mean),
            ("p50", lambda x: np.percentile(x, 50)),
            ("p95", lambda x: np.percentile(x, 95)),
            ("p99", lambda x: np.percentile(x, 99)),
        ]:
            metrics.append(
                {
                    "metric_name": f"ttft_{stat}",
                    "metric_value": func(ttft),
                    "kind": "measured",
                    "params": {},
                }
            )

        # TPOT: duration / completion_tokens
        if "completion_tokens" in llm_steps.columns:
            tpot = llm_steps["duration_ms"] / llm_steps["completion_tokens"].replace(
                0, np.nan
            )
            for stat, func in [
                ("mean", np.mean),
                ("p50", lambda x: np.nanpercentile(x, 50)),
                ("p95", lambda x: np.nanpercentile(x, 95)),
                ("p99", lambda x: np.nanpercentile(x, 99)),
            ]:
                metrics.append(
                    {
                        "metric_name": f"tpot_{stat}",
                        "metric_value": func(tpot),
                        "kind": "measured",
                        "params": {},
                    }
                )
        else:
            for stat in ["mean", "p50", "p95", "p99"]:
                metrics.append(
                    {
                        "metric_name": f"tpot_{stat}",
                        "metric_value": np.nan,
                        "kind": "measured",
                        "params": {"reason": "completion_tokens_unavailable"},
                    }
                )
    else:
        # No LLM steps
        for prefix in ["ttft", "tpot"]:
            for stat in ["mean", "p50", "p95", "p99"]:
                metrics.append(
                    {
                        "metric_name": f"{prefix}_{stat}",
                        "metric_value": np.nan,
                        "kind": "measured",
                        "params": {"reason": "no_llm_steps"},
                    }
                )

    # Task latency (measured)
    task_latency_s = (steps["end_ts"].max() - steps["start_ts"].min()) / 1e9
    metrics.append(
        {
            "metric_name": "task_latency",
            "metric_value": task_latency_s,
            "kind": "measured",
            "params": {},
        }
    )

    # Success (measured)
    success = 1.0 if verdict.get("status") == "solved" else 0.0
    metrics.append(
        {
            "metric_name": "success",
            "metric_value": success,
            "kind": "measured",
            "params": {},
        }
    )

    # Success under latency budgets (measured)
    for budget_s, name in [(300, "300s"), (600, "600s"), (1800, "1800s")]:
        success_under = 1.0 if success == 1.0 and task_latency_s <= budget_s else 0.0
        metrics.append(
            {
                "metric_name": f"success_under_{name}",
                "metric_value": success_under,
                "kind": "measured",
                "params": {"budget_seconds": budget_s},
            }
        )

    return pd.DataFrame(metrics)


def ranking_alignment_test(
    metrics_per_condition: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether ranking by TTFT/TPOT aligns with ranking by task latency.

    This is the core RQ3 probe. If token-level metrics don't predict task-level
    performance, they are insufficient for agentic evaluation.

    Args:
        metrics_per_condition: DataFrame with columns:
            - config_id
            - instance_id
            - ttft_p50
            - tpot_p50
            - task_latency

    Returns:
        DataFrame with columns:
            - comparison: str (ttft_vs_latency, tpot_vs_latency)
            - spearman_rho: float
            - p_value: float
            - aligned: bool (rho > 0.7 and p < 0.05)
    """
    from scipy.stats import spearmanr  # local import — only needed here
    results = []

    for metric_name, task_metric in [
        ("ttft_p50", "task_latency"),
        ("tpot_p50", "task_latency"),
    ]:
        # Per-config correlation
        for config_id, group in metrics_per_condition.groupby("config_id"):
            if len(group) < 3:
                continue  # Need at least 3 points for correlation

            x = group[metric_name].dropna()
            y = group[task_metric].dropna()

            # Align by index
            common_idx = x.index.intersection(y.index)
            if len(common_idx) < 3:
                continue

            x = x.loc[common_idx]
            y = y.loc[common_idx]

            rho, p_value = spearmanr(x, y)
            aligned = rho > 0.7 and p_value < 0.05

            results.append(
                {
                    "config_id": config_id,
                    "comparison": f"{metric_name}_vs_{task_metric}",
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "aligned": aligned,
                }
            )

    return pd.DataFrame(results)


def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["metric_name", "metric_value", "kind", "params"])
