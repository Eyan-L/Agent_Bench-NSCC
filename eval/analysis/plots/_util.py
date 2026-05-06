"""Shared helpers for the RQ1/RQ2/RQ3 plotting scripts.

Bridges three things:
  1. Per-run output files under data/runs/<sweep>/<run_id>/ (steps.jsonl,
     util.jsonl, verdict.json, manifest.json).
  2. The metric computers in eval/analysis/{composition,resource,evaluation}.py
     which already produce the right metrics from a (steps_df, telemetry_df,
     verdict) triple.
  3. The per-run util.jsonl schema (sidecar sampler, nested cpu/gpu) →
     wide DataFrame schema that resource.py expects (`gpu_utilization_%`,
     `cpu_utilization_%`, `rss_MB`, ...).

This module is pure data — no plotting. Plotting lives in rq{1,2,3}_*.py.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# eval.analysis.* import must work whether you run this as a module or a
# script. Adjust sys.path inside `main()` of the per-RQ scripts.
from eval.analysis.composition import compute_composition_metrics
from eval.analysis.evaluation import compute_evaluation_metrics
from eval.analysis.load import load_manifest
from eval.analysis.resource import compute_resource_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# config_id → display labels and groupings
# ---------------------------------------------------------------------------

# Order matters for plotting (left → right reads as "less agentic → more").
CONFIG_DISPLAY_ORDER: List[str] = [
    "ar_issue_only",
    "ar_static_context",
    "openclaw_agent",
    "tb_react",
]

CONFIG_DISPLAY_LABEL: Dict[str, str] = {
    "ar_issue_only":      "AR (issue only)",
    "ar_static_context":  "AR (static ctx)",
    "openclaw_agent":     "Agent (OpenClaw)",
    "tb_react":           "Agent (TB ReAct)",
}

# AR baselines vs agentic workflow — the core RQ1 split.
AR_FAMILIES = {"ar_issue_only", "ar_static_context"}
AGENT_FAMILIES = {"openclaw_agent", "tb_react"}


def family_for(config_id: Optional[str]) -> str:
    if config_id in AR_FAMILIES:
        return "AR"
    if config_id in AGENT_FAMILIES:
        return "Agent"
    return "Unknown"


def workload_class_for(config_id: Optional[str], manifest: Optional[Dict] = None) -> str:
    """Map a run to one of {code_agent, web_search, doc_analysis}.

    Order of resolution:
      1. Manifest's `workload_class` if explicitly set (forward-compatible).
      2. config_id heuristic — currently every condition in this repo is a
         code agent, so the default is `code_agent`.
    """
    if manifest:
        wc = manifest.get("workload_class")
        if wc:
            return wc
    # Future: detect web_search / doc_analysis from config_id naming.
    if config_id in (AR_FAMILIES | AGENT_FAMILIES):
        return "code_agent"
    return "code_agent"


def display_label(config_id: str) -> str:
    return CONFIG_DISPLAY_LABEL.get(config_id, config_id)


def order_configs(configs: List[str]) -> List[str]:
    """Return `configs` sorted by CONFIG_DISPLAY_ORDER (unknown configs last)."""
    rank = {c: i for i, c in enumerate(CONFIG_DISPLAY_ORDER)}
    return sorted(configs, key=lambda c: (rank.get(c, len(rank)), c))


# ---------------------------------------------------------------------------
# Per-run loaders
# ---------------------------------------------------------------------------

def load_steps(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "steps.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_verdict(run_dir: Path) -> Dict:
    p = run_dir / "verdict.json"
    if not p.exists():
        return {}
    with p.open() as fh:
        return json.load(fh)


def load_util_telemetry(run_dir: Path) -> pd.DataFrame:
    """Convert sidecar util.jsonl into the wide DataFrame schema that
    eval.analysis.resource.compute_resource_metrics expects.

    util.jsonl per-sample shape (after the leading `_meta: true` header):
        {ts_ns, iso, cpu: {pct, mem_pct, mem_used_mb, ...},
         gpu: [{util_gpu_pct, mem_used_mb, power_w, ...}, ...]}

    Output columns:
        timestamp_ns           — int (= ts_ns)
        cpu_utilization_%      — float (cpu.pct)
        gpu_utilization_%      — float (mean of gpu[].util_gpu_pct)
        gpu_mem_used_MB        — float (sum across GPUs)
        gpu_mem_total_MB       — float (sum across GPUs)
        gpu_power_W            — float (sum)
        cpu_mem_used_MB        — float (cpu.mem_used_mb)
        rss_MB                 — float (= cpu_mem_used_MB; we only have host-wide,
                                        not per-process, so this is a proxy)
    """
    p = run_dir / "util.jsonl"
    if not p.exists():
        return pd.DataFrame()

    rows = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            cpu = rec.get("cpu") or {}
            gpus = rec.get("gpu") or []
            gpu_utils = [g.get("util_gpu_pct", np.nan) for g in gpus]
            gpu_mem = sum((g.get("mem_used_mb") or 0) for g in gpus)
            gpu_mem_total = sum((g.get("mem_total_mb") or 0) for g in gpus)
            gpu_power = sum((g.get("power_w") or 0) for g in gpus)
            rows.append({
                "timestamp_ns":      rec.get("ts_ns"),
                "cpu_utilization_%": cpu.get("pct"),
                "gpu_utilization_%": (
                    float(np.nanmean(gpu_utils)) if gpu_utils else np.nan
                ),
                "gpu_mem_used_MB":   gpu_mem if gpus else np.nan,
                "gpu_mem_total_MB":  gpu_mem_total if gpus else np.nan,
                "gpu_power_W":       gpu_power if gpus else np.nan,
                "cpu_mem_used_MB":   cpu.get("mem_used_mb"),
                "rss_MB":            cpu.get("mem_used_mb"),  # proxy; see docstring
                "n_gpus":            len(gpus),
            })
    return pd.DataFrame(rows)


def list_run_dirs(runs_dir: Path) -> List[Path]:
    """Walk under runs_dir and return any directory containing a steps.jsonl
    — works for both flat layouts (data/runs/<run_id>/) and nested-by-sweep
    layouts (data/runs/phase1/<run_id>/)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    out = []
    for p in runs_dir.rglob("steps.jsonl"):
        out.append(p.parent)
    # Stable sort for deterministic plots.
    return sorted(out, key=lambda p: p.name)


def iter_runs(runs_dir: Path) -> Iterator[Tuple[Path, pd.DataFrame, pd.DataFrame, Dict, Dict]]:
    """Yield (run_dir, steps_df, telemetry_df, verdict, manifest) for every
    run with at least a steps.jsonl. Telemetry / verdict / manifest may be
    empty if those files are missing — callers must handle this.

    Bad runs (steps.jsonl missing or malformed) are skipped with a warning.
    """
    for run_dir in list_run_dirs(runs_dir):
        try:
            steps = load_steps(run_dir)
            if steps.empty:
                logger.warning("skip %s: empty steps.jsonl", run_dir.name)
                continue
            telem = load_util_telemetry(run_dir)
            verdict = load_verdict(run_dir)
            manifest = load_manifest(run_dir)
            yield run_dir, steps, telem, verdict, manifest
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", run_dir.name, e)


# ---------------------------------------------------------------------------
# Per-run, then long-format metrics
# ---------------------------------------------------------------------------

# Status values mean different things across harnesses. Treat any of these
# as "the agent solved it"; everything else (FAIL, TIMEOUT, ERROR, DEFERRED,
# RUNNING, ...) is failure for completion-rate purposes. This is intentionally
# broader than evaluation.py's "solved" check, which doesn't match the actual
# data.
PASS_STATUSES = {"PASS", "solved"}


def is_passed(verdict: Dict) -> Optional[float]:
    """1.0 if passed, 0.0 if failed terminal, None if non-terminal/unknown."""
    if not verdict:
        return None
    status = verdict.get("status")
    if status is None:
        return None
    if status in PASS_STATUSES:
        return 1.0
    if status in {"FAIL", "TIMEOUT", "ERROR", "DEFERRED", "failed"}:
        return 0.0
    return None


def per_run_metrics(
    steps: pd.DataFrame,
    telem: pd.DataFrame,
    verdict: Dict,
    manifest: Dict,
) -> pd.DataFrame:
    """All metrics for one run, in long format. Adds run-level columns:
    run_id, instance_id, config_id, workload_class, family, status.
    """
    parts = [
        compute_composition_metrics(steps),
        compute_resource_metrics(steps, telem),
        compute_evaluation_metrics(steps, verdict),
    ]
    long = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    # Resolve identifiers from manifest first, fall back to verdict / steps.
    config_id = (
        manifest.get("condition_id")
        or manifest.get("config_id")
        or verdict.get("config_id")
        or (steps["config_id"].dropna().iloc[0] if "config_id" in steps.columns and not steps.empty else None)
    )
    instance_id = (
        manifest.get("instance_id")
        or verdict.get("instance_id")
        or (steps["instance_id"].dropna().iloc[0] if "instance_id" in steps.columns and not steps.empty else None)
    )
    run_id = (
        manifest.get("run_id")
        or (steps["trajectory_id"].dropna().iloc[0] if "trajectory_id" in steps.columns and not steps.empty else None)
    )

    long["run_id"] = run_id
    long["instance_id"] = instance_id
    long["config_id"] = config_id
    long["workload_class"] = workload_class_for(config_id, manifest)
    long["family"] = family_for(config_id)
    long["status"] = verdict.get("status")
    return long


def build_long_metrics(runs_dir: Path) -> pd.DataFrame:
    """Collect per-run long-format metrics for every run under runs_dir."""
    parts = []
    for run_dir, steps, telem, verdict, manifest in iter_runs(runs_dir):
        try:
            parts.append(per_run_metrics(steps, telem, verdict, manifest))
        except Exception as e:  # noqa: BLE001
            logger.warning("metrics for %s failed: %s", run_dir.name, e)
    if not parts:
        return pd.DataFrame(
            columns=["run_id", "instance_id", "config_id", "workload_class",
                     "family", "status", "metric_name", "metric_value", "kind"]
        )
    return pd.concat(parts, ignore_index=True)


def pivot_metric(long_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Wide DataFrame, one row per run, columns include the named metric and
    grouping fields. Convenience for boxplots / scatter."""
    if long_df.empty:
        return pd.DataFrame()
    sub = long_df[long_df["metric_name"] == metric_name].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub[[
        "run_id", "instance_id", "config_id", "workload_class", "family",
        "status", "metric_value",
    ]].rename(columns={"metric_value": metric_name})


# ---------------------------------------------------------------------------
# Plotting infrastructure
# ---------------------------------------------------------------------------

def setup_style():
    """Apply consistent matplotlib + seaborn settings. Idempotent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (side effect: backend lock)
    try:
        import seaborn as sns
        sns.set_theme(context="paper", style="whitegrid", palette="tab10")
    except ImportError:
        pass
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi":    120,
        "savefig.dpi":   200,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def save_fig(fig, out_dir: Path, name: str, also_pdf: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return png_path


def warn_if_low_n(long_df: pd.DataFrame, min_n: int = 3) -> None:
    """Emit a warning per config_id when run count is below `min_n`. Plots
    still render, but error bars / CIs may be uninformative."""
    if long_df.empty:
        warnings.warn("no runs found — figures will be empty", stacklevel=2)
        return
    n_per = long_df.drop_duplicates(["run_id", "config_id"]).groupby("config_id").size()
    for cfg, n in n_per.items():
        if n < min_n:
            warnings.warn(
                f"only {n} run(s) for config_id={cfg!r}; "
                f"distributions / CIs will be unreliable",
                stacklevel=2,
            )
