"""RQ3: Are token-centric metrics still sufficient for evaluating agentic
workflow?

Five figures: TTFT/TPOT distributions per config, end-to-end task latency,
TTFT-vs-task-latency and TPOT-vs-task-latency scatter with Spearman ρ
annotations (the core RQ3 probe), and completion-rate bars.

Run:
    python -m orchestration.analysis.plots.rq3_metrics \
        --runs-dir data/experiments/<stamp>/runs --out-dir data/figures/rq3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from orchestration.analysis.plots._util import (
    build_long_metrics,
    display_label,
    display_workload,
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
        ylabel="TTFT (ms)",
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
        ylabel="TPOT (ms/tok)",
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

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.boxplot(data, tick_labels=[display_label(c) for c in cfgs],
               showmeans=True, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.4))
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.scatter([i] * len(vals), vals, color="black", alpha=0.6, s=15, zorder=3)
    ax.set_ylabel(ylabel)
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

    Encoding:
      - color  → workload class (code_agent, web_search, doc_analysis)
      - marker → family (AR = square, Agent = circle)
    Legend has two compact entries — one for color (workload), one for
    marker (family) — instead of one row per (cfg, ρ, p, n).
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

    # Workload-class palette (stable across plots).
    workload_palette = {
        "code_agent":   "#4C72B0",
        "web_search":   "#55A868",
        "doc_analysis": "#C44E52",
    }
    # AR vs Agent → marker shape.
    family_marker = {"AR": "s", "Agent": "o"}  # s=square, o=circle

    workloads = sorted(df["workload_class"].dropna().unique().tolist())
    families = sorted(df["family"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for wl in workloads:
        for fam in families:
            block = df[(df["workload_class"] == wl) & (df["family"] == fam)]
            if block.empty:
                continue
            ax.scatter(
                block["tk"], block["tl"],
                s=55, alpha=0.8,
                color=workload_palette.get(wl, "#888888"),
                marker=family_marker.get(fam, "o"),
                edgecolor="white", linewidth=0.6,
            )

    label = "TTFT" if token_metric == "ttft_mean" else "TPOT"
    ax.set_xlabel(f"{label} (ms)")
    ax.set_ylabel("Task latency (s)")
    if (df["tk"] > 0).all():
        ax.set_xscale("log")

    # Two compact legend groups: one for color, one for marker shape.
    handles = []
    for wl in workloads:
        handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="",
            color=workload_palette.get(wl, "#888888"),
            markersize=7, markeredgecolor="white",
            label=display_workload(wl),
        ))
    for fam in families:
        handles.append(plt.Line2D(
            [0], [0], marker=family_marker.get(fam, "o"), linestyle="",
            color="black", markersize=7, markeredgecolor="white",
            label=fam,
        ))
    ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.85,
              fontsize=10)
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

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
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
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    return save_fig(fig, out_dir, "rq3_completion_rate", also_pdf)


# ---------------------------------------------------------------------------
# F3.7 — Serving-regime inversion: TTFT/TPOT vs task latency split by
# {lat-opt-disagg, thru-opt-agg}. Demonstrates that token-centric metrics
# are not strictly positively correlated with end-to-end task latency
# under agentic workflows: AR (single-shot) inherits LLM cost so lat-opt
# wins on both axes, but Agent (multi-turn) gains from batched throughput
# so thru-opt yields lower task latency despite worse TTFT/TPOT.
# ---------------------------------------------------------------------------

def _collect_regime_table(runs_dir: Path) -> pd.DataFrame:
    """Per-run rows with TTFT/TPOT (mean) + task latency + regime tag.

    TTFT and TPOT are extracted from steps.jsonl as duration / (1 + ct)
    and duration / ct respectively — same approximation evaluation.py
    falls back to. Reading steps directly avoids depending on the
    long-format pipeline knowing about regimes.
    """
    rows = []
    for run_dir, steps, _telem, _verdict, manifest in iter_runs(runs_dir):
        if steps.empty:
            continue
        regime = (
            manifest.get("serving_regime")
            or manifest.get("serving_config_id")
        )
        if not regime:
            continue
        cfg = (
            manifest.get("condition_id")
            or manifest.get("config_id")
            or (steps["config_id"].dropna().iloc[0] if "config_id" in steps.columns else None)
        )
        wc = manifest.get("workload_class")
        if not cfg or not wc:
            continue

        llm = steps[steps["step_type"] == "LLM_INFER"].copy()
        if llm.empty:
            continue
        ct = llm["completion_tokens"].fillna(0)
        dur_ms = llm["duration_ms"].fillna(0)
        ttfts = dur_ms / (1.0 + ct)
        tpots = dur_ms / ct.replace(0, np.nan)

        wall_s = (
            float(manifest["duration_seconds"])
            if manifest.get("duration_seconds") is not None
            else float((steps["end_ts"].max() - steps["start_ts"].min()) / 1e9)
        )
        rows.append({
            "run_id":   run_dir.name,
            "config_id": cfg,
            "family":   "AR" if cfg.startswith("ar") else "Agent",
            "workload_class": wc,
            "regime":   regime,
            "regime_label": manifest.get("serving_regime_label") or regime,
            "ttft_ms":  float(ttfts.mean()),
            "tpot_ms":  float(tpots.dropna().mean()) if tpots.notna().any() else np.nan,
            "task_latency_s": wall_s,
        })
    return pd.DataFrame(rows)


def plot_regime_inversion(runs_dir: Path, out_dir: Path,
                          also_pdf: bool = False) -> Optional[Path]:
    """Two-panel scatter: TTFT (left), TPOT (right) on x; task latency on y.
    Each (family × workload_class) cell appears as two points connected by
    an arrow lat-opt → thru-opt. Down-and-right arrows = inversion.
    """
    df = _collect_regime_table(runs_dir)
    if df.empty:
        return None
    import matplotlib.pyplot as plt

    # Per-cell mean across runs.
    grp = (df
           .groupby(["family", "workload_class", "regime", "regime_label"], as_index=False)
           [["ttft_ms", "tpot_ms", "task_latency_s"]]
           .mean())

    family_marker = {"AR": "s", "Agent": "o"}
    workload_color = {
        "code_agent":   "#4C72B0",
        "doc_analysis": "#C44E52",
        "web_search":   "#55A868",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    panel_specs = [("ttft_ms", "TTFT (ms)"), ("tpot_ms", "TPOT (ms/tok)")]

    families  = sorted(grp["family"].unique().tolist())
    workloads = sorted(grp["workload_class"].unique().tolist())

    for ax, (xcol, xlab) in zip(axes, panel_specs):
        for fam in families:
            for wl in workloads:
                cell = grp[(grp["family"] == fam) & (grp["workload_class"] == wl)]
                if cell.empty:
                    continue
                # Order: lat-opt first (lower x expected), thru-opt second.
                cell = cell.sort_values("ttft_ms")
                xs = cell[xcol].values
                ys = cell["task_latency_s"].values
                col = workload_color.get(wl, "#888888")
                marker = family_marker.get(fam, "o")
                # Connecting arrow lat-opt -> thru-opt
                if len(cell) == 2:
                    ax.annotate(
                        "",
                        xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle="->", color=col, alpha=0.55,
                                        linewidth=1.2),
                    )
                ax.scatter(xs, ys, color=col, marker=marker, s=55,
                           alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Task latency (s)")
        if (grp[xcol] > 0).all():
            ax.set_xscale("log")

    # Build a compact legend on the second panel: workload colors + family markers.
    handles = []
    for wl in workloads:
        handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="",
            color=workload_color.get(wl, "#888888"),
            markersize=7, markeredgecolor="white",
            label=display_workload(wl),
        ))
    for fam in families:
        handles.append(plt.Line2D(
            [0], [0], marker=family_marker.get(fam, "o"), linestyle="",
            color="black", markersize=7, markeredgecolor="white",
            label=fam,
        ))
    # Arrow legend entry indicates direction of regime change.
    handles.append(plt.Line2D(
        [0], [0], marker=r"$\rightarrow$", linestyle="",
        color="#444444", markersize=11, label="lat-opt → thru-opt",
    ))
    axes[1].legend(handles=handles, loc="best", frameon=True, framealpha=0.85,
                   fontsize=9)
    fig.tight_layout()
    return save_fig(fig, out_dir, "rq3_regime_inversion", also_pdf)


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
        (plot_regime_inversion,     dict(runs_dir=args.runs_dir)),
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
