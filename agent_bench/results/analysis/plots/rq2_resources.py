"""RQ2: Do structural differences change the resource bottlenecks?

Six figures covering GPU/CPU utilization, memory, retries-as-wasted-work,
end-to-end latency, and a CPU-vs-GPU scatter that exposes whether agentic
workloads shift away from pure GPU-centric inference.

Run:
    python -m agent_bench.results.analysis.plots.rq2_resources \
        --runs-dir data/experiments/<stamp>/runs --out-dir data/figures/rq2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent_bench.results.analysis.plots._util import (
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
        cols = [
            "run_id", "config_id", "workload_class", "t_rel_s",
            "gpu_utilization_%", "cpu_utilization_%",
            "gpu_mem_used_MB", "gpu_power_W",
        ]
        # cpu_cores_busy is the new (per-run x ncpu) signal — present when
        # the sampler had per-pid info; harmless to include either way.
        if "cpu_cores_busy" in t.columns:
            cols.append("cpu_cores_busy")
        rows.append(t[cols])
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
    fig, axes = plt.subplots(2, n_panels, figsize=(3.4 * n_panels, 4.5),
                              sharex="col", sharey="row", squeeze=False)
    use_cores = "cpu_cores_busy" in telem_long.columns
    cpu_col = "cpu_cores_busy" if use_cores else "cpu_utilization_%"
    for j, cfg in enumerate(cfgs):
        block = telem_long[telem_long["config_id"] == cfg]
        for run_id, run in block.groupby("run_id"):
            axes[0, j].plot(run["t_rel_s"], run["gpu_utilization_%"],
                             alpha=0.6, linewidth=1)
            axes[1, j].plot(run["t_rel_s"], run[cpu_col],
                             alpha=0.6, linewidth=1)
        axes[1, j].set_xlabel("Time since trajectory start (s)")
    axes[0, 0].set_ylabel("GPU util. (%)")
    axes[1, 0].set_ylabel("CPU cores busy" if use_cores else "CPU util. (%)")
    for ax in axes[0]:
        ax.set_ylim(0, 105)
    if not use_cores:
        for ax in axes[1]:
            ax.set_ylim(0, 105)
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

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
    for ax, m, ylab in zip(
        axes, metrics,
        ["GPU util. (mean, %)", "GPU util. (p95, %)", "GPU idle fraction"],
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
    fig.tight_layout()
    return save_fig(fig, out_dir, "rq2_gpu_summary", also_pdf)


# ---------------------------------------------------------------------------
# F2.3 — CPU utilization summary (mean/p95) + memory peak (RSS proxy)
# ---------------------------------------------------------------------------

def plot_cpu_and_memory(long: pd.DataFrame, runs_dir: Path, out_dir: Path,
                        also_pdf: bool = False) -> Optional[Path]:
    """CPU panels are in cores busy (host_pct x ncpu / 100), computed
    per-run from telem so we don't depend on the long-metric pipeline
    knowing ncpu. RSS panel still comes from `long`."""
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    # Per-run CPU cores (mean/p95) from telemetry.
    cpu_rows = []
    for run_dir, _steps, telem, _verdict, manifest in iter_runs(runs_dir):
        if telem.empty or "cpu_cores_busy" not in telem.columns:
            continue
        cores = telem["cpu_cores_busy"].dropna()
        if cores.empty:
            continue
        cfg = (
            manifest.get("condition_id")
            or manifest.get("config_id")
            or (
                _steps["config_id"].dropna().iloc[0]
                if "config_id" in _steps.columns and not _steps.empty
                else None
            )
        )
        cpu_rows.append({
            "run_id":   run_dir.name,
            "config_id": cfg,
            "cores_mean": float(cores.mean()),
            "cores_p95":  float(np.nanpercentile(cores.values, 95)),
        })
    cpu_df = pd.DataFrame(cpu_rows)

    rss = long[long["metric_name"] == "rss_peak_MB"]
    cfgs_pool = pd.concat(
        [cpu_df["config_id"] if not cpu_df.empty else pd.Series(dtype=object),
         rss["config_id"] if not rss.empty else pd.Series(dtype=object)]
    )
    cfgs = order_configs(cfgs_pool.dropna().unique().tolist())
    if not cfgs:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
    panels = [
        ("CPU cores (mean)", lambda c: cpu_df.loc[cpu_df["config_id"] == c, "cores_mean"]),
        ("CPU cores (p95)",  lambda c: cpu_df.loc[cpu_df["config_id"] == c, "cores_p95"]),
        ("Peak RSS (MB)",    lambda c: rss.loc[rss["config_id"] == c, "metric_value"]),
    ]
    for ax, (ylab, fetch) in zip(axes, panels):
        means, errs = [], []
        for cfg in cfgs:
            v = pd.Series(fetch(cfg)).dropna() if cpu_df is not None or not rss.empty else pd.Series(dtype=float)
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std() if len(v) > 1 else 0.0)
        ax.bar(np.arange(len(cfgs)), means, yerr=errs, capsize=2,
               color="#55A868", edgecolor="white")
        ax.set_xticks(np.arange(len(cfgs)))
        ax.set_xticklabels([display_label(c) for c in cfgs], rotation=15, ha="right")
        ax.set_ylabel(ylab)
    fig.tight_layout()
    return save_fig(fig, out_dir, "rq2_cpu_and_memory", also_pdf)


# ---------------------------------------------------------------------------
# F2.4 — Retries / wasted work per trajectory (count + duration fraction)
# ---------------------------------------------------------------------------

def plot_wasted_work(runs_dir: Path, out_dir: Path,
                     also_pdf: bool = False) -> Optional[Path]:
    """Wasted work = duration of all attempts beyond the last one. Computed
    inline from steps because agent_bench.results.analysis.retry isn't always present.
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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

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
    axes[0].set_ylabel("Attempts (mean)")

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
    axes[1].set_ylabel("Wasted-work fraction")
    axes[1].set_ylim(0, max(0.05, max((m for m in means_w if not np.isnan(m)), default=0.0) * 1.3))

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

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.boxplot(data, tick_labels=[display_label(c) for c in cfgs],
               showmeans=True, patch_artist=True,
               boxprops=dict(facecolor="#8172B2", alpha=0.4))
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.scatter([i] * len(vals), vals, color="black", alpha=0.6, s=15, zorder=3)
    ax.set_ylabel("Task latency (s)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return save_fig(fig, out_dir, "rq2_task_latency", also_pdf)


# ---------------------------------------------------------------------------
# F2.6 — CPU vs GPU scatter (one point per run, colored by config_id)
# Reveals whether agentic shifts off the GPU-only diagonal.
# ---------------------------------------------------------------------------

def plot_cpu_vs_gpu_scatter(long: pd.DataFrame, runs_dir: Path, out_dir: Path,
                            also_pdf: bool = False) -> Optional[Path]:
    """CPU axis is mean cores busy (host_pct x ncpu / 100). GPU axis stays
    in % since GPUs are too few to make 'cards busy' meaningful."""
    if long.empty:
        return None
    import matplotlib.pyplot as plt

    gpu = pivot_metric(long, "gpu_utilization_mean").rename(columns={"gpu_utilization_mean": "gpu_util"})
    if gpu.empty:
        return None

    # Per-run mean cores from telemetry.
    cpu_rows = []
    for run_dir, _steps, telem, _verdict, _manifest in iter_runs(runs_dir):
        if telem.empty or "cpu_cores_busy" not in telem.columns:
            continue
        v = telem["cpu_cores_busy"].dropna()
        if v.empty:
            continue
        cpu_rows.append({"run_id": run_dir.name, "cpu_cores": float(v.mean())})
    if not cpu_rows:
        return None
    cpu_df = pd.DataFrame(cpu_rows)
    df = gpu.merge(cpu_df, on="run_id", how="inner").dropna(subset=["gpu_util", "cpu_cores"])
    if df.empty:
        return None

    cfgs = order_configs(df["config_id"].dropna().unique().tolist())
    palette = ["#4C72B0", "#55A868", "#8172B2", "#CCB974", "#C44E52"]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for i, cfg in enumerate(cfgs):
        block = df[df["config_id"] == cfg]
        ax.scatter(block["gpu_util"], block["cpu_cores"],
                   s=55, alpha=0.75,
                   color=palette[i % len(palette)],
                   label=display_label(cfg), edgecolor="white")
    ax.set_xlabel("GPU util. (mean, %)")
    ax.set_ylabel("CPU cores busy (mean)")
    ax.set_xlim(0, max(105, df["gpu_util"].max() * 1.05))
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    return save_fig(fig, out_dir, "rq2_cpu_vs_gpu_scatter", also_pdf)


# ---------------------------------------------------------------------------
# F2.7 — CPU utilization split by component (LLM engine / client / tools)
# Per-condition grouped bar; one bar group per condition, four bars per
# condition (engine / client / tools / other). Values are mean across the
# trajectory, in "% of one core x ncpu" units (same scale as host-wide).
# Requires the sampler to have been launched with --llm-pid/--client-pid/
# --gateway-pid; runs without those args contribute NaN and are excluded
# from this figure (the existing rq2_cpu_and_memory plot still works).
# ---------------------------------------------------------------------------

def plot_cpu_by_component(runs_dir: Path, out_dir: Path,
                          also_pdf: bool = False) -> Optional[Path]:
    """Read each run's util.jsonl directly (not via build_long_metrics, which
    summarizes host-wide). Per run, mean per-group CPU%; per condition, mean
    of per-run means."""
    import matplotlib.pyplot as plt

    rows = []
    for run_dir, steps, telem, _verdict, manifest in iter_runs(runs_dir):
        if telem.empty:
            continue
        cfg = (
            manifest.get("condition_id")
            or manifest.get("config_id")
            or (steps["config_id"].dropna().iloc[0] if "config_id" in steps.columns else None)
        )
        # Skip rows where the per-group split wasn't recorded (sampler ran
        # without --*-pid args).
        if telem["cpu_pct_llm"].dropna().empty and \
           telem["cpu_pct_client"].dropna().empty and \
           telem["cpu_pct_tools"].dropna().empty:
            continue
        rows.append({
            "run_id":  run_dir.name,
            "config_id": cfg,
            # Cores busy = jiffies-share x ncpu, ÷ 100 = cores. Use the
            # cpu_cores_* columns (already in core units).
            "llm":    float(telem["cpu_cores_llm"].mean(skipna=True) or 0.0),
            "client": float(telem["cpu_cores_client"].mean(skipna=True) or 0.0),
            "tools":  float(telem["cpu_cores_tools"].mean(skipna=True) or 0.0),
            "other":  float(telem["cpu_cores_other"].mean(skipna=True) or 0.0),
        })
    if not rows:
        logger.info("no runs with per-component CPU samples; skipping rq2_cpu_by_component")
        return None
    df = pd.DataFrame(rows)
    cfgs = order_configs(df["config_id"].dropna().unique().tolist())
    components = ["llm", "client", "tools", "other"]
    comp_labels = ["LLM engine", "Client", "Tools", "Other"]
    palette = ["#4C72B0", "#55A868", "#C44E52", "#999999"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    width = 0.8 / len(components)
    x = np.arange(len(cfgs))
    for i, (c, lab, col) in enumerate(zip(components, comp_labels, palette)):
        means, errs = [], []
        for cfg in cfgs:
            v = df[df["config_id"] == cfg][c]
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std() if len(v) > 1 else 0.0)
        ax.bar(x + i * width - 0.4 + width / 2, means, width=width,
               yerr=errs, capsize=2, label=lab, color=col, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([display_label(c) for c in cfgs], rotation=15, ha="right")
    ax.set_ylabel("CPU cores busy")
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    fig.tight_layout()
    return save_fig(fig, out_dir, "rq2_cpu_by_component", also_pdf)


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
        (plot_cpu_and_memory,        dict(long=long, runs_dir=args.runs_dir)),
        (plot_wasted_work,           dict(runs_dir=args.runs_dir)),
        (plot_task_latency,          dict(long=long)),
        (plot_cpu_vs_gpu_scatter,    dict(long=long, runs_dir=args.runs_dir)),
        (plot_cpu_by_component,      dict(runs_dir=args.runs_dir)),
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
