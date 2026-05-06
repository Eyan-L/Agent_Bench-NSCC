"""RQ3: Are token-centric metrics still sufficient for evaluating agentic
workflow?

Five figures: TTFT/TPOT distributions per config, end-to-end task latency,
TTFT-vs-task-latency and TPOT-vs-task-latency scatter with Spearman ρ
annotations (the core RQ3 probe), and completion-rate bars.

Run:
    python -m eval.analysis.plots.rq3_metrics \
        --runs-dir data/runs --out-dir data/figures/rq3
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
    build_long_metrics,
    display_label,
    is_passed,
    iter_runs,
    order_configs,
    pivot_metric,
    save_fig,
    setup_style,
    warn_if_low_n,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F3.1 — TTFT distribution per config (boxplot)
# ---------------------------------------------------------------------------

def plot_ttft_distribution(long: pd.DataFrame, out_dir: Path,
                           also_pdf: bool = False) -> Optional[Path]:
    return _plot_metric_box(
        long, "ttft_mean",
        title="RQ3 — TTFT (mean) per condition",
        ylabel="TTFT_mean (ms)",
        name="rq3_ttft_mean",
        out_dir=out_dir, also_pdf=also_pdf,
        log_y=True,
    )


# ---------------------------------------------------------------------------
# F3.2 — TPOT distribution per config (boxplot)
# ---------------------------------------------------------------------------

def plot_tpot_distribution(long: pd.DataFrame, out_dir: Path,
                           also_pdf: bool = False) -> Optional[Path]:
    return _plot_metric_box(
        long, "tpot_mean",
        title="RQ3 — TPOT (mean) per condition",
        ylabel="TPOT_mean (ms / output token)",
        name="rq3_tpot_mean",
        out_dir=out_dir, also_pdf=also_pdf,
        log_y=True,
    )


def _plot_metric_box(long: pd.DataFrame, metric: str, title: str, ylabel: str,
                     name: str, out_dir: Path, also_pdf: bool,
                     log_y: bool = False) -> Optional[Path]:
    if long.empty:
        return None
    sub = pivot_metric(long, metric).dropna(subset=[metric])
    if sub.empty:
        return None
    import matplotlib.pyplot as plt

    cfgs = order_configs(sub["config_id"].dropna().unique().tolist())
    data = [sub[sub["config_id"] == c][metric].values for c in cfgs]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot(data, tick_labels=[display_label(c) for c in cfgs],
               showmeans=True, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.4))
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.scatter([i] * len(vals), vals, color="black", alpha=0.6, s=15, zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return save_fig(fig, out_dir, name, also_pdf)


# ---------------------------------------------------------------------------
# F3.3 — Task latency distribution per config (boxplot)
# ---------------------------------------------------------------------------

def plot_task_latency_box(long: pd.DataFrame, out_dir: Path,
                          also_pdf: bool = False) -> Optional[Path]:
    return _plot_metric_box(
        long, "task_latency",
        title="RQ3 — End-to-end task latency",
        ylabel="Task latency (s)",
        name="rq3_task_latency",
        out_dir=out_dir, also_pdf=also_pdf,
    )


# ---------------------------------------------------------------------------
# F3.4 / F3.5 — token-metric vs task-latency scatter, with Spearman ρ
# ---------------------------------------------------------------------------

def plot_alignment_scatter(long: pd.DataFrame, token_metric: str,
                           out_dir: Path, also_pdf: bool = False) -> Optional[Path]:
    """Scatter of token-level metric vs task latency. One point per run.
    Spearman ρ printed per condition in the legend; an overall ρ across all
    runs is added as a panel title hint when N >= 3.
    """
    if long.empty:
        return None
    tk = pivot_metric(long, token_metric).rename(columns={token_metric: "tk"})
    tl = pivot_metric(long, "task_latency").rename(columns={"task_latency": "tl"})
    if tk.empty or tl.empty:
        return None
    df = tk.merge(tl[["run_id", "tl"]], on="run_id", how="inner").dropna(subset=["tk", "tl"])
    if df.empty:
        return None

    import matplotlib.pyplot as plt
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None

    cfgs = order_configs(df["config_id"].dropna().unique().tolist())
    palette = ["#4C72B0", "#55A868", "#8172B2", "#CCB974", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    legend_lines = []
    for i, cfg in enumerate(cfgs):
        block = df[df["config_id"] == cfg]
        ax.scatter(block["tk"], block["tl"], s=55, alpha=0.75,
                   color=palette[i % len(palette)],
                   label=None, edgecolor="white")
        if len(block) >= 3 and spearmanr is not None:
            rho, p = spearmanr(block["tk"], block["tl"])
            legend_lines.append(f"{display_label(cfg)}: ρ={rho:+.2f}  p={p:.3f}  n={len(block)}")
        else:
            legend_lines.append(f"{display_label(cfg)}: n={len(block)} (ρ N/A)")

    if len(df) >= 3 and spearmanr is not None:
        rho_all, p_all = spearmanr(df["tk"], df["tl"])
        title_suffix = f"  (overall ρ={rho_all:+.2f}, p={p_all:.3f}, n={len(df)})"
    else:
        title_suffix = ""

    label = "TTFT_mean" if token_metric == "ttft_mean" else "TPOT_mean"
    ax.set_xlabel(f"{label}")
    ax.set_ylabel("End-to-end task latency (s)")
    ax.set_title(f"RQ3 — {label} vs task latency{title_suffix}")
    if (df["tk"] > 0).all():
        ax.set_xscale("log")
    # Manually craft legend so each line has its config color box.
    handles = [
        plt.Line2D([0], [0], marker="o", color=palette[i % len(palette)],
                   linestyle="", markersize=7, markeredgecolor="white")
        for i in range(len(cfgs))
    ]
    ax.legend(handles, legend_lines, frameon=False, fontsize=8,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))
    name = f"rq3_{token_metric}_vs_task_latency"
    return save_fig(fig, out_dir, name, also_pdf)


# ---------------------------------------------------------------------------
# F3.6 — Completion rate per config (overall + within latency budgets)
# ---------------------------------------------------------------------------

def plot_completion_rate(runs_dir: Path, out_dir: Path,
                         also_pdf: bool = False) -> Optional[Path]:
    """Overall pass rate plus three latency-budget-conditioned pass rates,
    grouped by config. Reads verdict + total wall time directly so it
    doesn't depend on evaluation.py's success metric (which has a string
    mismatch we don't want to silently propagate)."""
    rows = []
    for run_dir, steps, _telem, verdict, manifest in iter_runs(runs_dir):
        if steps.empty:
            continue
        cfg = (
            manifest.get("condition_id")
            or manifest.get("config_id")
            or (steps["config_id"].dropna().iloc[0] if "config_id" in steps.columns else None)
        )
        passed = is_passed(verdict)
        # Prefer manifest wall duration; fall back to step span.
        if manifest.get("duration_seconds") is not None:
            latency = float(manifest["duration_seconds"])
        else:
            latency = float((steps["end_ts"].max() - steps["start_ts"].min()) / 1e9)
        rows.append({
            "run_id": run_dir.name, "config_id": cfg,
            "passed": passed, "latency_s": latency,
        })
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df[df["passed"].notna()]
    if df.empty:
        return None
    cfgs = order_configs(df["config_id"].dropna().unique().tolist())

    import matplotlib.pyplot as plt

    budgets = [(None, "Overall"), (300, "≤300s"), (600, "≤600s"), (1800, "≤1800s")]
    rates = {label: [] for _, label in budgets}
    n_per = []
    for cfg in cfgs:
        block = df[df["config_id"] == cfg]
        n_per.append(len(block))
        for budget, label in budgets:
            if budget is None:
                rates[label].append(block["passed"].mean() if len(block) else np.nan)
            else:
                bonly = block[block["latency_s"] <= budget]
                rates[label].append(bonly["passed"].mean() if len(bonly) else np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / len(budgets)
    x = np.arange(len(cfgs))
    palette = ["#4C72B0", "#55A868", "#8172B2", "#CCB974"]
    for i, (_, label) in enumerate(budgets):
        ax.bar(x + i * width - 0.4 + width / 2, rates[label], width=width,
               color=palette[i % len(palette)], edgecolor="white", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{display_label(c)}\n(n={n})" for c, n in zip(cfgs, n_per)],
        rotation=15, ha="right",
    )
    ax.set_ylabel("Completion rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("RQ3 — Completion rate (overall and within latency budgets)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return save_fig(fig, out_dir, "rq3_completion_rate", also_pdf)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RQ3 figures: token-centric vs task-centric evaluation"
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

    produced = []
    for fn, kwargs in [
        (plot_ttft_distribution,    dict(long=long)),
        (plot_tpot_distribution,    dict(long=long)),
        (plot_task_latency_box,     dict(long=long)),
        (plot_alignment_scatter,    dict(long=long, token_metric="ttft_mean")),
        (plot_alignment_scatter,    dict(long=long, token_metric="tpot_mean")),
        (plot_completion_rate,      dict(runs_dir=args.runs_dir)),
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
