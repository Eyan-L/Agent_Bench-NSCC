#!/usr/bin/env python3
"""TFLOPS-per-GB-of-KV over time from sglang.log.

Rationale: the existing TFLOPS plot shows raw compute throughput. This plot
divides by the KV-cache memory actually in use at that moment, so the y-axis
becomes "how much compute we're doing per GB of KV footprint" — a crude
compute/memory intensity indicator. High values ⇒ good arithmetic intensity
relative to cache residency; low values ⇒ cache is full and we're spending
most of the memory budget to hold context that isn't being decoded against.

Inputs parsed from sglang.log:

  1. KV Cache allocation (one line per TP rank, emitted once at boot):

         [TS TP0] KV Cache is allocated. #tokens: 249728, K size: 9.53 GB, V size: 9.53 GB

     The total KV budget = sum over TP ranks of (K size + V size) GB.

  2. Per-batch lines emitted by the patched sglang (see `sglang_patch/`):

         [TS TPx] Prefill batch, ... token usage: 0.42, ..., TFLOPS: 18.7,
                  forward_ms: 103.4, start_ts: 1712000000.123

     Both prefill and decode batches carry `token usage` (fraction in [0,1])
     and TFLOPS/start_ts once the patch is applied.

Per point:

    used_gb   = token_usage * total_kv_gb
    intensity = TFLOPS / used_gb

Points with zero used_gb (early warmup, nothing cached yet) are dropped to
keep the axis finite.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


KV_RE = re.compile(
    r"\[.*?TP(?P<tp>\d+)\]\s*KV Cache is allocated\.\s*"
    r"#tokens:\s*\d+,\s*K size:\s*(?P<k>[0-9.]+)\s*GB,\s*"
    r"V size:\s*(?P<v>[0-9.]+)\s*GB"
)

# Combined batch regex: prefill/decode tag, plus all three added-by-patch
# fields (TFLOPS / forward_ms / start_ts) AND the pre-existing token usage.
# Using `.*?` between named groups keeps it order-tolerant as long as the
# fields appear on the same line (they do, in both prefill and decode logs).
BATCH_KIND_RE = re.compile(r"(?P<kind>Prefill batch|Decode batch)")
USAGE_RE = re.compile(r"token usage:\s*(?P<usage>[0-9]*\.?[0-9]+)")
TFLOPS_RE = re.compile(r"TFLOPS:\s*(?P<tflops>[0-9]*\.?[0-9]+)")
FORWARD_RE = re.compile(r"forward_ms:\s*(?P<forward_ms>[0-9]*\.?[0-9]+)")
START_TS_RE = re.compile(r"start_ts:\s*(?P<start_ts>[0-9]+\.?[0-9]*)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log_path", help="Path to sglang.log.")
    p.add_argument(
        "--output-plot",
        default="sglang_tflops_per_gb_over_time.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--bin-sec",
        type=float,
        default=1.0,
        help=(
            "Time-bin width (seconds) for the aggregate mean line. "
            "Within a bin the intensity is averaged, weighted by forward_ms. "
            "Set 0 to disable the aggregate line."
        ),
    )
    p.add_argument(
        "--min-used-gb",
        type=float,
        default=0.1,
        help=(
            "Drop points whose used_gb (= token_usage * total_kv_gb) is below "
            "this threshold. Helps cut out warmup noise and avoid blow-up "
            "near zero memory."
        ),
    )
    p.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Drop points with elapsed seconds (from first point) < START_SEC.",
    )
    p.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Drop points with elapsed seconds (from first point) > END_SEC.",
    )
    p.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional y-axis upper bound (TFLOPS/GB).",
    )
    p.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use log scale for y (enable when intensity spans orders of magnitude).",
    )
    return p.parse_args()


def extract_kv_total_gb(log_path: Path) -> tuple[float, dict[str, tuple[float, float]]]:
    """Return (total_kv_gb, per_tp_sizes). per_tp_sizes maps "TPx" to (K, V)."""
    per_tp: dict[str, tuple[float, float]] = {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in KV_RE.finditer(text):
        tp = f"TP{m.group('tp')}"
        # First-write-wins: a single boot emits exactly one line per rank;
        # if the log happens to contain multiple boots, the first wins.
        if tp not in per_tp:
            per_tp[tp] = (float(m.group("k")), float(m.group("v")))
    total = sum(k + v for k, v in per_tp.values())
    return total, per_tp


def extract_points(log_path: Path) -> list[tuple[str, float, float, float, float]]:
    """Return list of (kind, start_ts_epoch, tflops, forward_ms, usage).

    Any line that is a batch-log line AND has all of {token usage, TFLOPS,
    forward_ms, start_ts} contributes one point. Lines missing any of these
    (e.g. unpatched sglang, or metadata-only lines) are skipped silently.
    """
    points: list[tuple[str, float, float, float, float]] = []
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            mk = BATCH_KIND_RE.search(line)
            if not mk:
                continue
            mu = USAGE_RE.search(line)
            mt = TFLOPS_RE.search(line)
            mf = FORWARD_RE.search(line)
            ms = START_TS_RE.search(line)
            if not (mu and mt and mf and ms):
                continue
            points.append(
                (
                    mk.group("kind"),
                    float(ms.group("start_ts")),
                    float(mt.group("tflops")),
                    float(mf.group("forward_ms")),
                    float(mu.group("usage")),
                )
            )
    return points


def filter_points(
    points: list[tuple[str, float, float, float, float]],
    total_kv_gb: float,
    min_used_gb: float,
    start_sec: float | None,
    end_sec: float | None,
) -> list[tuple[str, float, float, float, float]]:
    """Drop early-warmup-near-zero-memory points and apply time window."""
    if not points:
        return points
    points_sorted = sorted(points, key=lambda x: x[1])
    t0 = points_sorted[0][1]
    lo = start_sec if start_sec is not None else float("-inf")
    hi = end_sec if end_sec is not None else float("inf")
    if lo > hi:
        raise ValueError(f"--start-sec ({lo}) must be <= --end-sec ({hi})")
    out = []
    for kind, ts, tflops, fwd, usage in points_sorted:
        used_gb = usage * total_kv_gb
        if used_gb < min_used_gb:
            continue
        if not (lo <= (ts - t0) <= hi):
            continue
        out.append((kind, ts, tflops, fwd, usage))
    return out


def aggregate_bins(
    points: list[tuple[str, float, float, float, float]],
    total_kv_gb: float,
    bin_sec: float,
) -> tuple[list[float], list[float]]:
    """Per-bin mean intensity (TFLOPS/GB), weighted by forward_ms.

    Within a bin: Σ(tflops_i / used_gb_i · forward_ms_i) / Σ(forward_ms_i).
    That gives the time-weighted average intensity rather than a simple mean
    of point values, which would over-weight short batches.
    """
    if not points or bin_sec <= 0:
        return [], []
    points_sorted = sorted(points, key=lambda x: x[1])
    t0 = points_sorted[0][1]
    last_t = points_sorted[-1][1]
    n_bins = max(1, int((last_t - t0) // bin_sec) + 1)
    num = [0.0] * n_bins   # numerator: Σ(intensity · forward_ms)
    den = [0.0] * n_bins   # denominator: Σ(forward_ms)
    for kind, ts, tflops, fwd_ms, usage in points_sorted:
        used_gb = usage * total_kv_gb
        if used_gb <= 0:
            continue
        intensity = tflops / used_gb
        idx = min(int((ts - t0) // bin_sec), n_bins - 1)
        num[idx] += intensity * fwd_ms
        den[idx] += fwd_ms
    centers: list[float] = []
    rates: list[float] = []
    for i in range(n_bins):
        if den[i] <= 0:
            continue
        centers.append((i + 0.5) * bin_sec)
        rates.append(num[i] / den[i])
    return centers, rates


def compute_average_intensity(
    points: list[tuple[str, float, float, float, float]],
    total_kv_gb: float,
) -> tuple[float, float]:
    """forward_ms-weighted mean intensity across the analyzed window."""
    if not points:
        return 0.0, 0.0
    num = 0.0
    den = 0.0
    for _kind, _ts, tflops, fwd_ms, usage in points:
        used_gb = usage * total_kv_gb
        if used_gb <= 0:
            continue
        num += (tflops / used_gb) * fwd_ms
        den += fwd_ms
    if den <= 0:
        return 0.0, 0.0
    points_sorted = sorted(points, key=lambda x: x[1])
    wall = points_sorted[-1][1] - points_sorted[0][1]
    return num / den, wall


def plot(
    points: list[tuple[str, float, float, float, float]],
    total_kv_gb: float,
    bin_sec: float,
    output_plot: Path,
    ymax: float | None,
    log_y: bool,
) -> tuple[float, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points.sort(key=lambda x: x[1])
    t0 = points[0][1]

    prefill = [((p[1] - t0), p[2] / (p[4] * total_kv_gb)) for p in points if p[0] == "Prefill batch"]
    decode = [((p[1] - t0), p[2] / (p[4] * total_kv_gb)) for p in points if p[0] == "Decode batch"]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    if prefill:
        xs, ys = zip(*prefill)
        ax.scatter(
            xs, ys, s=8, alpha=0.55, color="#1f77b4",
            label=f"Prefill (n={len(prefill)})",
        )
    if decode:
        xs, ys = zip(*decode)
        ax.scatter(
            xs, ys, s=8, alpha=0.45, color="#ff7f0e",
            label=f"Decode (n={len(decode)})",
        )

    if bin_sec > 0:
        centers, rates = aggregate_bins(points, total_kv_gb, bin_sec)
        if centers:
            ax.plot(
                centers, rates,
                color="#2ca02c", linewidth=1.6, alpha=0.9,
                label=f"Aggregate TFLOPS/GB ({bin_sec:g}s bins, fwd-weighted)",
            )

    avg_intensity, wall = compute_average_intensity(points, total_kv_gb)
    if wall > 0 and avg_intensity > 0:
        ax.axhline(
            avg_intensity,
            color="#d62728", linestyle="--", linewidth=1.4, alpha=0.85,
            label=f"Mean = {avg_intensity:.2f} TFLOPS/GB (fwd-weighted)",
        )

    title = f"SGLang TFLOPS / KV-GB Over Time  —  total KV budget: {total_kv_gb:.1f} GB"
    if wall > 0:
        title += f"  (window {wall:.1f}s, mean {avg_intensity:.2f} TFLOPS/GB)"
    ax.set_title(title)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("TFLOPS / GB (KV in use)")
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0, top=ymax)
    ax.legend(loc="best", framealpha=0.85)

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot)
    plt.close(fig)
    return avg_intensity, wall


def run() -> int:
    args = parse_args()
    log_path = Path(args.log_path).expanduser().resolve()
    output_plot = Path(args.output_plot).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    total_kv_gb, per_tp = extract_kv_total_gb(log_path)
    if total_kv_gb <= 0:
        raise ValueError(
            "No 'KV Cache is allocated' line found (or K+V summed to zero). "
            "Cannot compute TFLOPS/GB without knowing the total KV budget."
        )

    points = extract_points(log_path)
    if not points:
        raise ValueError(
            "No matching batch lines with (token usage, TFLOPS, forward_ms, "
            "start_ts) found. Is the patched sglang from sglang_patch/ active?"
        )

    points = filter_points(points, total_kv_gb, args.min_used_gb, args.start_sec, args.end_sec)
    if not points:
        raise ValueError(
            f"No points remain after filtering (min_used_gb={args.min_used_gb}, "
            f"start_sec={args.start_sec}, end_sec={args.end_sec})."
        )

    avg_intensity, wall = plot(
        points, total_kv_gb, args.bin_sec, output_plot, args.ymax, args.log_y
    )

    n_prefill = sum(1 for p in points if p[0] == "Prefill batch")
    n_decode = sum(1 for p in points if p[0] == "Decode batch")
    tp_summary = ", ".join(
        f"{tp}: K={k:.2f}+V={v:.2f}GB" for tp, (k, v) in sorted(per_tp.items())
    )

    print(f"log_path={log_path}")
    print(f"kv_ranks=[{tp_summary}]")
    print(f"total_kv_gb={total_kv_gb:.2f}")
    print(f"points={len(points)} (prefill={n_prefill}, decode={n_decode})")
    print(f"window_sec={wall:.2f}")
    print(f"avg_tflops_per_gb={avg_intensity:.3f} (forward_ms-weighted)")
    print(f"output_plot={output_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
