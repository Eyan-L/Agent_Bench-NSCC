#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

BATCH_RE = re.compile(
    r"(?P<kind>Prefill batch|Decode batch)\b.*?"
    r"TFLOPS:\s*(?P<tflops>[0-9]*\.?[0-9]+).*?"
    r"forward_ms:\s*(?P<forward_ms>[0-9]*\.?[0-9]+).*?"
    r"start_ts:\s*(?P<start_ts>[0-9]+\.?[0-9]*)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parse achieved TFLOPS per batch from sglang.log and plot TFLOPS vs time. "
            "Prefill and Decode batches are distinguished. An optional binned mean "
            "line shows aggregate compute throughput over time."
        )
    )
    p.add_argument("log_path", help="Path to sglang.log.")
    p.add_argument(
        "--output-plot",
        default="sglang_tflops_over_time.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--bin-sec",
        type=float,
        default=1.0,
        help=(
            "Time-bin width (seconds) for the aggregate FLOPs/s line. "
            "Computed as sum(TFLOPS * forward_ms / 1000) per bin / bin_sec. "
            "Set 0 to disable the aggregate line."
        ),
    )
    p.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Drop points with elapsed seconds (from first batch) < START_SEC.",
    )
    p.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Drop points with elapsed seconds (from first batch) > END_SEC.",
    )
    p.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional y-axis upper bound (TFLOPS).",
    )
    return p.parse_args()


def extract_points(log_path: Path) -> list[tuple[str, float, float, float]]:
    """Return list of (kind, start_ts_epoch, tflops, forward_ms)."""
    points: list[tuple[str, float, float, float]] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in BATCH_RE.finditer(text):
        points.append(
            (
                m.group("kind"),
                float(m.group("start_ts")),
                float(m.group("tflops")),
                float(m.group("forward_ms")),
            )
        )
    return points


def filter_by_time(
    points: list[tuple[str, float, float, float]],
    start_sec: float | None,
    end_sec: float | None,
) -> list[tuple[str, float, float, float]]:
    if not points or (start_sec is None and end_sec is None):
        return points
    lo = start_sec if start_sec is not None else float("-inf")
    hi = end_sec if end_sec is not None else float("inf")
    if lo > hi:
        raise ValueError(f"--start-sec ({lo}) must be <= --end-sec ({hi})")
    points_sorted = sorted(points, key=lambda x: x[1])
    t0 = points_sorted[0][1]
    return [p for p in points_sorted if lo <= (p[1] - t0) <= hi]


def aggregate_bins(
    points: list[tuple[str, float, float, float]],
    bin_sec: float,
) -> tuple[list[float], list[float]]:
    """Bin total compute into fixed-width windows.

    Returns (bin_center_elapsed_sec, achieved_tflops_in_bin).
    """
    if not points or bin_sec <= 0:
        return [], []
    points_sorted = sorted(points, key=lambda x: x[1])
    t0 = points_sorted[0][1]
    last_t = points_sorted[-1][1]
    n_bins = max(1, int((last_t - t0) // bin_sec) + 1)
    work = [0.0] * n_bins
    for _kind, ts, tflops, fwd_ms in points_sorted:
        idx = min(int((ts - t0) // bin_sec), n_bins - 1)
        work[idx] += tflops * (fwd_ms / 1000.0)
    centers = [(i + 0.5) * bin_sec for i in range(n_bins)]
    rates = [w / bin_sec for w in work]
    return centers, rates


def compute_average_tflops(
    points: list[tuple[str, float, float, float]],
) -> tuple[float, float, float]:
    """Wall-clock-weighted average over the analyzed window.

    Returns (avg_tflops, total_compute_tflops_s, wall_clock_sec).
    avg = Σ(TFLOPS_i · forward_s_i) / wall_clock, where wall_clock is
    last_start_ts - first_start_ts (in seconds), spanning the filtered window.
    """
    if not points:
        return 0.0, 0.0, 0.0
    points_sorted = sorted(points, key=lambda x: x[1])
    wall = points_sorted[-1][1] - points_sorted[0][1]
    if wall <= 0:
        return 0.0, 0.0, 0.0
    total = sum(tflops * (fwd_ms / 1000.0) for _k, _t, tflops, fwd_ms in points_sorted)
    return total / wall, total, wall


def plot(
    points: list[tuple[str, float, float, float]],
    bin_sec: float,
    output_plot: Path,
    ymax: float | None,
) -> tuple[float, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points.sort(key=lambda x: x[1])
    t0 = points[0][1]

    prefill = [(p[1] - t0, p[2]) for p in points if p[0] == "Prefill batch"]
    decode = [(p[1] - t0, p[2]) for p in points if p[0] == "Decode batch"]

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
        centers, rates = aggregate_bins(points, bin_sec)
        if centers:
            ax.plot(
                centers, rates,
                color="#2ca02c", linewidth=1.6, alpha=0.9,
                label=f"Aggregate TFLOPs/s ({bin_sec:g}s bins)",
            )

    avg_tflops, _total, wall = compute_average_tflops(points)
    if wall > 0:
        ax.axhline(
            avg_tflops,
            color="#d62728", linestyle="--", linewidth=1.4, alpha=0.85,
            label=f"Mean = {avg_tflops:.1f} TFLOPS (wall-clock weighted)",
        )

    title = "SGLang Achieved TFLOPS Over Time"
    if wall > 0:
        title += f"  —  window {wall:.1f}s, mean {avg_tflops:.1f} TFLOPS"
    ax.set_title(title)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("TFLOPS")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=ymax)
    ax.legend(loc="best", framealpha=0.85)

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot)
    plt.close(fig)
    return avg_tflops, wall


def run() -> int:
    args = parse_args()
    log_path = Path(args.log_path).expanduser().resolve()
    output_plot = Path(args.output_plot).expanduser().resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    points = extract_points(log_path)
    if not points:
        raise ValueError(
            "No matching lines found. Expected 'Prefill batch' or 'Decode batch' "
            "lines containing TFLOPS, forward_ms, and start_ts fields."
        )

    points = filter_by_time(points, args.start_sec, args.end_sec)
    if not points:
        raise ValueError(
            f"No points remain after time filter "
            f"(start={args.start_sec}, end={args.end_sec})"
        )

    avg_tflops, wall = plot(points, args.bin_sec, output_plot, args.ymax)

    n_prefill = sum(1 for p in points if p[0] == "Prefill batch")
    n_decode = sum(1 for p in points if p[0] == "Decode batch")
    print(f"log_path={log_path}")
    print(f"points={len(points)} (prefill={n_prefill}, decode={n_decode})")
    print(f"window_sec={wall:.2f}")
    print(f"avg_tflops={avg_tflops:.2f} (wall-clock weighted)")
    print(f"output_plot={output_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
