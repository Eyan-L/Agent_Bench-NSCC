"""RQ2: Do structural differences change the resource bottlenecks?

Six figures covering GPU/CPU utilization, memory, retries-as-wasted-work,
end-to-end latency, and a CPU-vs-GPU scatter that exposes whether agentic
workloads shift away from pure GPU-centric inference.

Run:
    python -m eval.analysis.plots.rq2_resources \
        --runs-dir data/runs --out-dir data/figures/rq2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eval.analysis.plots._util import (
    build_long_metrics,
    display_label,
    iter_runs,
    order_configs,
    pivot_metric,
    save_fig,
    setup_style,
    warn_if_low_n,
    workload_class_for,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample telemetry table for the timeline figure (avoids re-reading util
# files inside each plot helper).
# ---------------------------------------------------------------------------

def collect_telemetry_table(runs_dir: Path) -> pd.DataFrame:
    """Per-sample telemetry rows with run/config metadata attached. Used by
    plot_gpu_timeline. Returns columns:
        run_id, config_id, workload_class, t_rel_s,
        gpu_utilization_%, cpu_utilization_%, gpu_mem_used_MB, gpu_power_W
    where t_rel_s is seconds from trajectory start.
    """
    rows = []
    for run_dir, steps, telem, _verdict, manifest in iter_runs(runs_dir):
        if telem.empty or steps.empty:
            continue
        cfg = (
            manifest.get("condition_id")
            or manifest.get("config_id")
            or (steps["config_id"].dropna().iloc[0] if "config_id" in steps.columns else None)
        )
        wc = workload_class_for(cfg, manifest)
        run_id = (
            manifest.get("run_id")
            or steps.get("trajectory_id", pd.Series(dtype=object)).dropna().iloc[0]
        )
        traj_start_ns = steps["start_ts"].min()
        t = telem.copy()
        t["t_rel_s"] = (t["timestamp_ns"] - traj_start_ns) / 1e9
        t["run_id"] = run_id
        t["config_id"] = cfg
        t["workload_class"] = wc
        rows.append(t[[
            "run_id", "config_id", "workload_class", "t_rel_s",
            "gpu_utilization_%", "cpu_utilization_%",
            "gpu_mem_used_MB", "gpu_power_W",
        ]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# F2.1 — utilization timelines (one panel per config; one line per run)
# ---------------------------------------------------------------------------

def plot_utilization_timelines(telem_long: pd.DataFrame, out_dir: Path,
                                also_pdf: bool = False) -> Optional[Path]:
    if telem_long.empty:
        return None
    import matplotlib.pyplot as plt

    cfgs = order_configs(telem_long["config_id"].dropna().unique().tolist())
    n_panels = len(cfgs)
    fig, axes = plt.subplots(2, n_panels, figsize=(4.5 * n_panels, 5.5),
                              sharex="col", sharey="row", squeeze=False)
    for j, cfg in enumerate(cfgs):
        block = telem_long[telem_long["config_id"] == cfg]
        for run_id, run in block.groupby("run_id"):
            axes[0, j].plot(run["t_rel_s"], run["gpu_utilization_%"],
                             alpha=0.6, linewidth=1)
            axes[1, j].plot(run["t_rel_s"], run["cpu_utilization_%"],
                             alpha=0.6, linewidth=1)
        axes[0, j].set_title(display_label(cfg))
        axes[1, j].set_xlabel("Time since trajectory start (s)")
    axes[0, 0].set_ylabel("GPU util (%)")
    axes[1, 0].set_ylabel("CPU util (%)")
    for ax in axes.flat:
        ax.set_ylim(0, 105)
    fig.suptitle("RQ2 — GPU/CPU utilization timelines", y=1.02)
    return save_fig(fig, out_dir, "rq2_utilization_timelines", also_pdf)


# ---------------------------------------------------------------------------
# F2.2 — GPU utilization mean/p95 per config (grouped bar)
# ---------------------------------------------------------------------------

def plot_gpu_summary(long: pd.DataFrame, out_dir: Path,
                      also_pdf: bool = False) -> Optional[Path]:
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    metrics = ["gpu_utilization_mean", "gpu_utilization_p95", "gpu_idle_fraction"]
    sub = long[long["metric_name"].isin(metrics)]
    if sub.empty:
        return None
    cfgs = order_configs(sub["config_id"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, m, ylab in zip(
        axes, metrics,
        ["Mean GPU util (%)", "p95 GPU util (%)", "GPU idle fraction (util<10%)"],
    ):
        means, errs = [], []
        for cfg in cfgs:
            v = sub[(sub["config_id"] == cfg) & (sub["metric_name"] == m)]["metric_value"].dropna()
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std() if len(v) > 1 else 0.0)
        ax.bar(np.arange(len(cfgs)), means, yerr=errs, capsize=2,
               color="#4C72B0", edgecolor="white")
        ax.set_xticks(np.arange(len(cfgs)))
        ax.set_xticklabels([display_label(c) for c in cfgs], rotation=15, ha="right")
        ax.set_ylabel(ylab)
        if "fraction" in m:
            ax.set_ylim(0, 1.05)
    fig.suptitle("RQ2 — GPU utilization summary", y=1.03)
    fig.tight_layout()
    return save_fig(fig, out_dir, "rq2_gpu_summary", also_pdf)


# ---------------------------------------------------------------------------
# F2.3 — CPU utilization summary (mean/p95) + memory peak (RSS proxy)
# ---------------------------------------------------------------------------

def plot_cpu_and_memory(long: pd.DataFrame, out_dir: Path,
                        also_pdf: bool = False) -> Optional[Path]:
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    metrics = ["cpu_utilization_mean", "cpu_utilization_p95", "rss_peak_MB"]
    sub = long[long["metric_name"].isin(metrics)]
    if sub.empty:
        return None
    cfgs = order_configs(sub["config_id"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    ylabs = ["Mean CPU util (%)", "p95 CPU util (%)", "Peak host RSS (MB)"]
    for ax, m, ylab in zip(axes, metrics, ylabs):
        means, errs = [], []
        for cfg in cfgs:
            v = sub[(sub["config_id"] == cfg) & (sub["metric_name"] == m)]["metric_value"].dropna()
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std() if len(v) > 1 else 0.0)
        ax.bar(np.arange(len(cfgs)), means, yerr=errs, capsize=2,
               color="#55A868", edgecolor="white")
        ax.set_xticks(np.arange(len(cfgs)))
        ax.set_xticklabels([display_label(c) for c in cfgs], rotation=15, ha="right")
        ax.set_ylabel(ylab)
    fig.suptitle("RQ2 — CPU & memory summary", y=1.03)
    fig.tight_layout()
    return save_fig(fig, out_dir, "rq2_cpu_and_memory", also_pdf)


# ---------------------------------------------------------------------------
# F2.4 — Retries / wasted work per trajectory (count + duration fraction)
# ---------------------------------------------------------------------------

def plot_wasted_work(runs_dir: Path, out_dir: Path,
                     also_pdf: bool = False) -> Optional[Path]:
    """Wasted work = duration of all attempts beyond the last one. Computed
    inline from steps because eval.analysis.retry isn't always present.
    """
    import matplotlib.pyplot as plt

    rows = []
    for run_dir, steps, _telem, _verdict, manifest in iter_runs(runs_dir):
        if steps.empty or "attempt_id" not in steps.columns:
            continue
        cfg = (
            manifest.get("condition_id")
            or manifest.get("config_id")
            or (steps["config_id"].dropna().iloc[0] if "config_id" in steps.columns else None)
        )
        attempts = steps["attempt_id"].dropna().astype(int)
        if attempts.empty:
            continue
        last = attempts.max()
        total_ms = steps["duration_ms"].sum()
        wasted_ms = steps[steps["attempt_id"] != last]["duration_ms"].sum()
        rows.append({
            "run_id": run_dir.name,
            "config_id": cfg,
            "n_attempts": int(attempts.nunique()),
            "wasted_fraction": (wasted_ms / total_ms) if total_ms > 0 else 0.0,
            "wasted_ms": float(wasted_ms),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    cfgs = order_configs(df["config_id"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Attempts per trajectory.
    means_a, errs_a = [], []
    for cfg in cfgs:
        v = df[df["config_id"] == cfg]["n_attempts"]
        means_a.append(v.mean() if len(v) else np.nan)
        errs_a.append(v.std() if len(v) > 1 else 0.0)
    axes[0].bar(np.arange(len(cfgs)), means_a, yerr=errs_a, capsize=2,
                color="#CCB974", edgecolor="white")
    axes[0].set_xticks(np.arange(len(cfgs)))
    axes[0].set_xticklabels([display_label(c) for c in cfgs], rotation=15, ha="right")
    axes[0].set_ylabel("Mean attempts per trajectory")

    # Wasted-work fraction.
    means_w, errs_w = [], []
    for cfg in cfgs:
        v = df[df["config_id"] == cfg]["wasted_fraction"]
        means_w.append(v.mean() if len(v) else np.nan)
        errs_w.append(v.std() if len(v) > 1 else 0.0)
    axes[1].bar(np.arange(len(cfgs)), means_w, yerr=errs_w, capsize=2,
                color="#C44E52", edgecolor="white")
    axes[1].set_xticks(np.arange(len(cfgs)))
    axes[1].set_xticklabels([display_label(c) for c in cfgs], rotation=15, ha="right")
    axes[1].set_ylabel("Wasted-work fraction (failed attempts / total)")
    axes[1].set_ylim(0, max(0.05, max((m for m in means_w if not np.isnan(m)), default=0.0) * 1.3))

    fig.suptitle("RQ2 — Retries and wasted work", y=1.03)
    fig.tight_layout()
    return save_fig(fig, out_dir, "rq2_wasted_work", also_pdf)


# ---------------------------------------------------------------------------
# F2.5 — End-to-end task latency per config (boxplot)
# ---------------------------------------------------------------------------

def plot_task_latency(long: pd.DataFrame, out_dir: Path,
                      also_pdf: bool = False) -> Optional[Path]:
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    sub = pivot_metric(long, "task_latency").dropna(subset=["task_latency"])
    if sub.empty:
        return None
    cfgs = order_configs(sub["config_id"].dropna().unique().tolist())
    data = [sub[sub["config_id"] == c]["task_latency"].values for c in cfgs]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot(data, tick_labels=[display_label(c) for c in cfgs],
               showmeans=True, patch_artist=True,
               boxprops=dict(facecolor="#8172B2", alpha=0.4))
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.scatter([i] * len(vals), vals, color="black", alpha=0.6, s=15, zorder=3)
    ax.set_ylabel("End-to-end task latency (s)")
    ax.set_title("RQ2 — Task latency distribution per condition")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return save_fig(fig, out_dir, "rq2_task_latency", also_pdf)


# ---------------------------------------------------------------------------
# F2.6 — CPU vs GPU scatter (one point per run, colored by config_id)
# Reveals whether agentic shifts off the GPU-only diagonal.
# ---------------------------------------------------------------------------

def plot_cpu_vs_gpu_scatter(long: pd.DataFrame, out_dir: Path,
                            also_pdf: bool = False) -> Optional[Path]:
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    gpu = pivot_metric(long, "gpu_utilization_mean").rename(columns={"gpu_utilization_mean": "gpu_util"})
    cpu = pivot_metric(long, "cpu_utilization_mean").rename(columns={"cpu_utilization_mean": "cpu_util"})
    if gpu.empty or cpu.empty:
        return None
    df = gpu.merge(cpu[["run_id", "cpu_util"]], on="run_id", how="inner").dropna(subset=["gpu_util", "cpu_util"])
    if df.empty:
        return None

    cfgs = order_configs(df["config_id"].dropna().unique().tolist())
    palette = ["#4C72B0", "#55A868", "#8172B2", "#CCB974", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, cfg in enumerate(cfgs):
        block = df[df["config_id"] == cfg]
        ax.scatter(block["gpu_util"], block["cpu_util"],
                   s=55, alpha=0.75,
                   color=palette[i % len(palette)],
                   label=display_label(cfg), edgecolor="white")
    ax.set_xlabel("Mean GPU utilization (%)")
    ax.set_ylabel("Mean CPU utilization (%)")
    ax.set_xlim(0, max(105, df["gpu_util"].max() * 1.05))
    ax.set_ylim(0, max(105, df["cpu_util"].max() * 1.05))
    ax.set_title("RQ2 — Per-run CPU vs GPU utilization")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return save_fig(fig, out_dir, "rq2_cpu_vs_gpu_scatter", also_pdf)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RQ2 figures: resource bottlenecks of AR vs agent"
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
    telem_long = collect_telemetry_table(args.runs_dir)

    produced = []
    for fn, kwargs in [
        (plot_utilization_timelines, dict(telem_long=telem_long)),
        (plot_gpu_summary,           dict(long=long)),
        (plot_cpu_and_memory,        dict(long=long)),
        (plot_wasted_work,           dict(runs_dir=args.runs_dir)),
        (plot_task_latency,          dict(long=long)),
        (plot_cpu_vs_gpu_scatter,    dict(long=long)),
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
