#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


LINE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\s+[^\]]*)?\]\s+Prefill batch,.*?token usage:\s*(?P<usage>[0-9]*\.?[0-9]+)"
)
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 'Prefill batch ... token usage' lines from sglang.log "
            "and plot token usage over time."
        )
    )
    parser.add_argument(
        "log_path",
        help="Path to sglang.log.",
    )
    parser.add_argument(
        "--output-plot",
        default="sglang_token_usage_over_time.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Drop points with elapsed_seconds (from first point) < START_SEC.",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Drop points with elapsed_seconds (from first point) > END_SEC.",
    )
    return parser.parse_args()


def filter_by_time(
    points: list[tuple[datetime, float]],
    start_sec: float | None,
    end_sec: float | None,
) -> list[tuple[datetime, float]]:
    if start_sec is None and end_sec is None:
        return points
    if not points:
        return points
    lo = start_sec if start_sec is not None else float("-inf")
    hi = end_sec if end_sec is not None else float("inf")
    if lo > hi:
        raise ValueError(f"--start-sec ({lo}) must be <= --end-sec ({hi})")
    points_sorted = sorted(points, key=lambda x: x[0])
    t0 = points_sorted[0][0]
    return [p for p in points_sorted if lo <= (p[0] - t0).total_seconds() <= hi]


def extract_points(log_path: Path) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.search(raw_line)
        if not m:
            continue
        ts = datetime.strptime(m.group("ts"), TS_FORMAT)
        usage = float(m.group("usage"))
        points.append((ts, usage))
    return points


def plot_points(points: list[tuple[datetime, float]], output_plot: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    points.sort(key=lambda x: x[0])
    times = [p[0] for p in points]
    usages = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    ax.plot(times, usages, linewidth=1.8, marker="o", markersize=2.5, color="#1f77b4")
    ax.set_title("SGLang Token Usage Over Time (Prefill Batch)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Token Usage")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot)
    plt.close(fig)


def run() -> int:
    args = parse_args()
    log_path = Path(args.log_path).expanduser().resolve()
    output_plot = Path(args.output_plot).expanduser().resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    points = extract_points(log_path)
    if not points:
        raise ValueError(
            "No matching lines found. Expected lines like: "
            "'[time] Prefill batch, ... token usage: <value>, ...'"
        )

    points = filter_by_time(points, args.start_sec, args.end_sec)
    if not points:
        raise ValueError(
            f"No points remain after time filter (start={args.start_sec}, end={args.end_sec})"
        )

    plot_points(points, output_plot)
    print(f"log_path={log_path}")
    print(f"points={len(points)}")
    print(f"output_plot={output_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
