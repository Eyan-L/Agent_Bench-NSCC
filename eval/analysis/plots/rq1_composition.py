"""RQ1: How does a trajectory-based workflow change execution composition
compared with traditional autoregressive (AR) serving?

Six figures, one per call to a `plot_*` helper. They are independent so the
caller can disable any of them. All emit PNG into --out-dir; --also-pdf
additionally emits a vector copy.

Run:
    python -m eval.analysis.plots.rq1_composition \
        --runs-dir data/runs --out-dir data/figures/rq1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from eval.analysis.plots._util import (
    CONFIG_DISPLAY_LABEL,
    build_long_metrics,
    display_label,
    iter_runs,
    order_configs,
    pivot_metric,
    save_fig,
    setup_style,
    warn_if_low_n,
)

logger = logging.getLogger(__name__)


# Ordered for stacked bars: chunky LLM at the bottom, tools above, wait/stall
# on top so it's easy to compare wait fractions across configs by eye.
STEP_TYPES_ORDERED = ["LLM_INFER", "TOOL_READ", "TOOL_WRITE", "TOOL_EXEC", "WAIT_STALL"]
STEP_COLOR = {
    "LLM_INFER":  "#4C72B0",
    "TOOL_READ":  "#55A868",
    "TOOL_WRITE": "#8172B2",
    "TOOL_EXEC":  "#CCB974",
    "WAIT_STALL": "#C44E52",
}


# ---------------------------------------------------------------------------
# Per-step aggregations (pulled from raw steps.jsonl, not the long metrics
# DataFrame) — gives us flexibility to do step-count + time-fraction views.
# ---------------------------------------------------------------------------

def collect_step_table(runs_dir: Path) -> pd.DataFrame:
    """One row per step with run-level columns attached. Columns:
    run_id, config_id, workload_class, step_type, duration_ms,
    prompt_tokens, completion_tokens.
    """
    rows = []
    for run_dir, steps, _telem, _verdict, manifest in iter_runs(runs_dir):
        if steps.empty:
            continue
        from eval.analysis.plots._util import workload_class_for
        cfg = (
            manifest.get("condition_id")
            or manifest.get("config_id")
            or (steps["config_id"].dropna().iloc[0] if "config_id" in steps.columns else None)
        )
        wc = workload_class_for(cfg, manifest)
        run_id = (
            manifest.get("run_id")
            or steps.get("trajectory_id", pd.Series(dtype=object)).dropna().iloc[0]
            if not steps.empty else None
        )
        df = steps[["step_type", "duration_ms"]].copy()
        for col in ("prompt_tokens", "completion_tokens"):
            df[col] = steps[col] if col in steps.columns else np.nan
        df["run_id"] = run_id
        df["config_id"] = cfg
        df["workload_class"] = wc
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=["run_id", "config_id", "workload_class", "step_type",
                     "duration_ms", "prompt_tokens", "completion_tokens"]
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# F1.1 — stacked bar: time fraction by step type per config_id
# ---------------------------------------------------------------------------

def plot_time_fraction_stacked(steps_long: pd.DataFrame, out_dir: Path,
                                also_pdf: bool = False) -> Optional[Path]:
    if steps_long.empty:
        return None
    import matplotlib.pyplot as plt

    # Sum duration per (config_id, step_type), then normalize per config.
    by_cfg = (
        steps_long.groupby(["config_id", "step_type"])["duration_ms"]
        .sum().unstack(fill_value=0.0)
    )
    by_cfg = by_cfg.reindex(columns=STEP_TYPES_ORDERED, fill_value=0.0)
    totals = by_cfg.sum(axis=1).replace(0, np.nan)
    fracs = by_cfg.divide(totals, axis=0).fillna(0.0)
    fracs = fracs.reindex(order_configs(list(fracs.index)))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bottoms = np.zeros(len(fracs))
    x = np.arange(len(fracs))
    for st in STEP_TYPES_ORDERED:
        vals = fracs[st].values
        ax.bar(x, vals, bottom=bottoms, label=st,
               color=STEP_COLOR[st], edgecolor="white", linewidth=0.4)
        bottoms = bottoms + vals
    ax.set_xticks(x)
    ax.set_xticklabels([display_label(c) for c in fracs.index], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Wall-time fraction")
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    return save_fig(fig, out_dir, "rq1_time_fraction_by_step_type", also_pdf)


# ---------------------------------------------------------------------------
# F1.2 — stacked bar: step-count fraction by step type per config_id
# (companion to F1.1; same data shape but counts instead of duration)
# ---------------------------------------------------------------------------

def plot_step_count_stacked(steps_long: pd.DataFrame, out_dir: Path,
                             also_pdf: bool = False) -> Optional[Path]:
    if steps_long.empty:
        return None
    import matplotlib.pyplot as plt

    by_cfg = (
        steps_long.groupby(["config_id", "step_type"]).size().unstack(fill_value=0)
    )
    by_cfg = by_cfg.reindex(columns=STEP_TYPES_ORDERED, fill_value=0)
    totals = by_cfg.sum(axis=1).replace(0, np.nan)
    fracs = by_cfg.divide(totals, axis=0).fillna(0.0)
    fracs = fracs.reindex(order_configs(list(fracs.index)))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bottoms = np.zeros(len(fracs))
    x = np.arange(len(fracs))
    for st in STEP_TYPES_ORDERED:
        vals = fracs[st].values
        ax.bar(x, vals, bottom=bottoms, label=st,
               color=STEP_COLOR[st], edgecolor="white", linewidth=0.4)
        bottoms = bottoms + vals
    ax.set_xticks(x)
    ax.set_xticklabels([display_label(c) for c in fracs.index], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Step-count fraction")
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    return save_fig(fig, out_dir, "rq1_step_count_by_step_type", also_pdf)


# ---------------------------------------------------------------------------
# F1.3 — three-grouped bar: tool-call ratio, non-LLM time fraction,
# wait-stall fraction (one bar group per config_id)
# ---------------------------------------------------------------------------

def plot_tool_and_stall_fractions(long: pd.DataFrame, out_dir: Path,
                                  also_pdf: bool = False) -> Optional[Path]:
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    metrics = ["tool_call_ratio", "non_llm_time_fraction", "wait_stall_fraction"]
    titles = ["Tool-call ratio", "Non-LLM time fraction", "Wait/stall fraction"]
    sub = long[long["metric_name"].isin(metrics)]
    if sub.empty:
        return None
    cfgs = order_configs(sub["config_id"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    n_groups = len(cfgs)
    width = 0.8 / max(len(metrics), 1)
    x = np.arange(n_groups)
    for i, m in enumerate(metrics):
        means = []
        errs = []
        for cfg in cfgs:
            v = sub[(sub["config_id"] == cfg) & (sub["metric_name"] == m)]["metric_value"].dropna()
            if len(v) == 0:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(v.mean())
                errs.append(v.std() if len(v) > 1 else 0.0)
        ax.bar(x + i * width - 0.4 + width / 2, means, width=width,
               yerr=errs, capsize=2, label=titles[i])
    ax.set_xticks(x)
    ax.set_xticklabels([display_label(c) for c in cfgs], rotation=15, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    return save_fig(fig, out_dir, "rq1_tool_and_stall_fractions", also_pdf)


# ---------------------------------------------------------------------------
# F1.4 — input/output token ratio per config_id (boxplot)
# ---------------------------------------------------------------------------

def plot_token_ratio(long: pd.DataFrame, out_dir: Path,
                     also_pdf: bool = False) -> Optional[Path]:
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    sub = pivot_metric(long, "input_output_token_ratio").dropna(subset=["input_output_token_ratio"])
    if sub.empty:
        return None
    cfgs = order_configs(sub["config_id"].dropna().unique().tolist())
    data = [sub[sub["config_id"] == c]["input_output_token_ratio"].values for c in cfgs]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bp = ax.boxplot(data, tick_labels=[display_label(c) for c in cfgs],
                    showmeans=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.4))
    # Overlay individual points so single-run configs still show up.
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.scatter([i] * len(vals), vals, color="black", alpha=0.6, s=15, zorder=3)
    ax.set_ylabel("Prompt / completion tokens")
    ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return save_fig(fig, out_dir, "rq1_token_ratio", also_pdf)


# ---------------------------------------------------------------------------
# F1.5 — per-workload-class composition (only differs once we have web_search
# / doc_analysis runs; today it just reproduces F1.1 split by workload_class).
# ---------------------------------------------------------------------------

def plot_composition_by_workload_class(steps_long: pd.DataFrame, out_dir: Path,
                                       also_pdf: bool = False) -> Optional[Path]:
    if steps_long.empty:
        return None
    classes = sorted(steps_long["workload_class"].dropna().unique().tolist())
    if len(classes) < 2:
        # Only one workload class present (today: code_agent only). Skip the
        # facet; F1.1 already covers this case.
        return None
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(classes), figsize=(3.6 * len(classes), 3.5), sharey=True,
    )
    if len(classes) == 1:
        axes = [axes]

    for ax, wc in zip(axes, classes):
        block = steps_long[steps_long["workload_class"] == wc]
        by = (
            block.groupby(["config_id", "step_type"])["duration_ms"]
            .sum().unstack(fill_value=0.0)
            .reindex(columns=STEP_TYPES_ORDERED, fill_value=0.0)
        )
        totals = by.sum(axis=1).replace(0, np.nan)
        fracs = by.divide(totals, axis=0).fillna(0.0)
        fracs = fracs.reindex(order_configs(list(fracs.index)))
        bottoms = np.zeros(len(fracs))
        x = np.arange(len(fracs))
        for st in STEP_TYPES_ORDERED:
            v = fracs[st].values
            ax.bar(x, v, bottom=bottoms, label=st, color=STEP_COLOR[st],
                   edgecolor="white", linewidth=0.4)
            bottoms += v
        ax.set_xticks(x)
        ax.set_xticklabels([display_label(c) for c in fracs.index], rotation=15, ha="right")
        # workload class label is shown via the legend / x-axis ticks
        from eval.analysis.plots._util import display_workload
        ax.set_xlabel(display_workload(wc), fontsize=11)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Fraction of total wall time")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return save_fig(fig, out_dir, "rq1_composition_by_workload_class", also_pdf)


# ---------------------------------------------------------------------------
# F1.6 — AR vs Agent rollup: collapse all AR conditions, all Agent conditions
# into one bar each, so the headline contrast is one chart.
# ---------------------------------------------------------------------------

def plot_family_rollup(steps_long: pd.DataFrame, out_dir: Path,
                       also_pdf: bool = False) -> Optional[Path]:
    if steps_long.empty:
        return None
    import matplotlib.pyplot as plt
    from eval.analysis.plots._util import family_for

    df = steps_long.copy()
    df["family"] = df["config_id"].map(family_for)
    df = df[df["family"].isin(["AR", "Agent"])]
    if df.empty:
        return None

    by = (
        df.groupby(["family", "step_type"])["duration_ms"]
        .sum().unstack(fill_value=0.0)
        .reindex(columns=STEP_TYPES_ORDERED, fill_value=0.0)
    )
    totals = by.sum(axis=1).replace(0, np.nan)
    fracs = by.divide(totals, axis=0).fillna(0.0)
    fracs = fracs.reindex(["AR", "Agent"]).dropna(how="all")

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    bottoms = np.zeros(len(fracs))
    x = np.arange(len(fracs))
    for st in STEP_TYPES_ORDERED:
        v = fracs[st].values
        ax.bar(x, v, bottom=bottoms, label=st, color=STEP_COLOR[st],
               edgecolor="white", linewidth=0.4)
        bottoms += v
    ax.set_xticks(x)
    ax.set_xticklabels(list(fracs.index))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Wall-time fraction")
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    return save_fig(fig, out_dir, "rq1_ar_vs_agent_rollup", also_pdf)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RQ1 figures: AR vs trajectory execution composition"
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--also-pdf", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    setup_style()

    long = build_long_metrics(args.runs_dir)
    warn_if_low_n(long)
    steps_long = collect_step_table(args.runs_dir)

    produced = []
    for fn, kwargs in [
        (plot_time_fraction_stacked,        dict(steps_long=steps_long)),
        (plot_step_count_stacked,           dict(steps_long=steps_long)),
        (plot_tool_and_stall_fractions,     dict(long=long)),
        (plot_token_ratio,                  dict(long=long)),
        (plot_composition_by_workload_class, dict(steps_long=steps_long)),
        (plot_family_rollup,                dict(steps_long=steps_long)),
    ]:
        try:
            p = fn(out_dir=args.out_dir, also_pdf=args.also_pdf, **kwargs)
            if p:
                produced.append(p)
        except Exception as e:  # noqa: BLE001
            logger.exception("plot %s failed: %s", fn.__name__, e)

    print(f"wrote {len(produced)} figure(s) to {args.out_dir}:")
    for p in produced:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
