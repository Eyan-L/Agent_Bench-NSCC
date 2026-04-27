#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DCGMI_METRICS = ["SMACT", "SMOCC", "DRAMA", "PCITX", "PCIRX"]
NVIDIA_SMI_METRICS = ["sm", "mem"]
NVIDIA_SMI_TS_FORMAT = "%Y%m%d %H:%M:%S"


@dataclass
class MetricPoint:
    elapsed_seconds: float
    values: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot GPU metrics over time from sm.log. Supports the current dcgmi "
            "dmon output used by scripts/run_agents.sh and also older "
            "nvidia-smi dmon logs."
        )
    )
    parser.add_argument(
        "sm_log_path",
        help="Path to sm.log, e.g. experiment/2026-04-01_14-42-16/sm.log",
    )
    parser.add_argument(
        "--sample-interval-sec",
        type=float,
        default=1.0,
        help=(
            "Sampling interval in seconds. Used for dcgmi logs that do not carry "
            "per-row timestamps. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Only plot one GPU id, e.g. --gpu-id 1. Default: plot all GPUs.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output file prefix. Default: <sm_log_dir>/sm_metrics. The script writes "
            "<prefix>_utilization.png and <prefix>_pcie.png."
        ),
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Drop samples with elapsed_seconds < START_SEC.",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Drop samples with elapsed_seconds > END_SEC.",
    )
    return parser.parse_args()


def _filter_series_by_time(
    series_by_gpu: dict[int, list[MetricPoint]],
    start_sec: float | None,
    end_sec: float | None,
) -> dict[int, list[MetricPoint]]:
    if start_sec is None and end_sec is None:
        return series_by_gpu
    lo = start_sec if start_sec is not None else float("-inf")
    hi = end_sec if end_sec is not None else float("inf")
    if lo > hi:
        raise ValueError(f"--start-sec ({lo}) must be <= --end-sec ({hi})")
    return {
        gpu_id: [p for p in points if lo <= p.elapsed_seconds <= hi]
        for gpu_id, points in series_by_gpu.items()
    }


def _to_float(token: str) -> float | None:
    token = token.strip()
    if not token or token == "N/A":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _is_dcgmi_log(lines: list[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith("#Entity")
    return False


def _parse_gpu_id(token: str) -> int | None:
    if token.startswith("GPU"):
        parts = token.split()
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
    if token.isdigit():
        return int(token)
    return None


def parse_dcgmi_log(lines: list[str], sample_interval_sec: float) -> dict[int, list[MetricPoint]]:
    series_by_gpu: dict[int, list[MetricPoint]] = defaultdict(list)
    sample_index_by_gpu: dict[int, int] = defaultdict(int)

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "ID":
            continue

        parts = stripped.split()
        if len(parts) < 7 or parts[0] != "GPU":
            continue

        gpu_id = _parse_gpu_id(" ".join(parts[:2]))
        if gpu_id is None:
            continue

        values: dict[str, float] = {}
        for metric_name, token in zip(DCGMI_METRICS, parts[2:7], strict=False):
            value = _to_float(token)
            if value is None:
                continue
            if metric_name in {"PCITX", "PCIRX"}:
                value /= 1024 * 1024
            values[metric_name] = value

        point = MetricPoint(
            elapsed_seconds=sample_index_by_gpu[gpu_id] * sample_interval_sec,
            values=values,
        )
        series_by_gpu[gpu_id].append(point)
        sample_index_by_gpu[gpu_id] += 1

    return dict(series_by_gpu)


def parse_nvidia_smi_log(lines: list[str]) -> dict[int, list[MetricPoint]]:
    series_by_gpu: dict[int, list[MetricPoint]] = defaultdict(list)
    start_ts: datetime | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("Date")
            or stripped.startswith("Time")
        ):
            continue

        parts = stripped.split()
        if len(parts) < 9:
            continue

        try:
            current_ts = datetime.strptime(f"{parts[0]} {parts[1]}", NVIDIA_SMI_TS_FORMAT)
            gpu_id = int(parts[2])
        except ValueError:
            continue

        if start_ts is None:
            start_ts = current_ts

        values: dict[str, float] = {}
        for metric_name, token in zip(NVIDIA_SMI_METRICS, parts[3:5], strict=False):
            value = _to_float(token)
            if value is not None:
                values[metric_name.upper()] = value / 100.0

        series_by_gpu[gpu_id].append(
            MetricPoint(
                elapsed_seconds=(current_ts - start_ts).total_seconds(),
                values=values,
            )
        )

    return dict(series_by_gpu)


def parse_sm_log(sm_log_path: Path, sample_interval_sec: float) -> tuple[str, dict[int, list[MetricPoint]]]:
    lines = sm_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if _is_dcgmi_log(lines):
        return "dcgmi", parse_dcgmi_log(lines, sample_interval_sec)
    return "nvidia-smi", parse_nvidia_smi_log(lines)


def _filter_series_by_gpu(
    series_by_gpu: dict[int, list[MetricPoint]],
    gpu_id: int | None,
) -> dict[int, list[MetricPoint]]:
    if gpu_id is None:
        return series_by_gpu
    if gpu_id not in series_by_gpu:
        raise ValueError(f"GPU {gpu_id} not found in sm.log. Available GPUs: {sorted(series_by_gpu)}")
    return {gpu_id: series_by_gpu[gpu_id]}


def _plot_metric_group(
    series_by_gpu: dict[int, list[MetricPoint]],
    metric_names: list[str],
    colors: dict[str, str],
    title_prefix: str,
    ylabel: str,
    output_plot: Path,
    ylim_top: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu_ids = sorted(series_by_gpu)
    fig, axes = plt.subplots(
        nrows=len(gpu_ids),
        ncols=1,
        figsize=(13, 4.5 * max(len(gpu_ids), 1)),
        dpi=140,
        squeeze=False,
    )

    for row_idx, gpu_id in enumerate(gpu_ids):
        ax = axes[row_idx][0]
        points = series_by_gpu[gpu_id]
        xs = [point.elapsed_seconds for point in points]
        lines = []

        for metric_name in metric_names:
            ys = [
                point.values.get(metric_name, math.nan)
                for point in points
            ]
            if all(math.isnan(y) for y in ys):
                continue
            (line,) = ax.plot(
                xs,
                ys,
                linewidth=1.8,
                label=metric_name,
                color=colors[metric_name],
            )
            lines.append(line)
            valid_ys = [y for y in ys if not math.isnan(y)]
            if valid_ys:
                mean_y = sum(valid_ys) / len(valid_ys)
                ax.axhline(
                    mean_y,
                    linewidth=1.4,
                    linestyle="--",
                    color=colors[metric_name],
                    alpha=0.8,
                    label=f"{metric_name} mean",
                )

        ax.set_title(f"{title_prefix}: GPU {gpu_id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        if ylim_top is not None:
            ax.set_ylim(top=ylim_top)

        ax.grid(True, alpha=0.3)

        if lines:
            ax.legend(
                lines,
                [line.get_label() for line in lines],
                loc="upper left",
                ncol=min(len(lines), 5),
            )

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot)
    plt.close(fig)


def run() -> int:
    args = parse_args()
    sm_log_path = Path(args.sm_log_path).expanduser().resolve()
    if not sm_log_path.exists():
        raise FileNotFoundError(f"sm.log not found: {sm_log_path}")

    if args.sample_interval_sec <= 0:
        raise ValueError("--sample-interval-sec must be > 0")

    output_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else sm_log_path.with_name("sm_metrics")
    )

    log_kind, series_by_gpu = parse_sm_log(sm_log_path, args.sample_interval_sec)
    series_by_gpu = _filter_series_by_gpu(series_by_gpu, args.gpu_id)
    series_by_gpu = _filter_series_by_time(series_by_gpu, args.start_sec, args.end_sec)
    series_by_gpu = {gid: pts for gid, pts in series_by_gpu.items() if pts}
    if not series_by_gpu:
        raise ValueError(f"No GPU metric samples parsed from {sm_log_path}")

    if log_kind == "dcgmi":
        _plot_metric_group(
            series_by_gpu=series_by_gpu,
            metric_names=["SMACT", "SMOCC", "DRAMA"],
            colors={
                "SMACT": "#1f77b4",
                "SMOCC": "#ff7f0e",
                "DRAMA": "#2ca02c",
            },
            title_prefix="GPU Utilization Metrics Over Time",
            ylabel="Utilization / Activity",
            output_plot=output_prefix.with_name(f"{output_prefix.name}_utilization.png"),
            ylim_top=1.05,
        )
        _plot_metric_group(
            series_by_gpu=series_by_gpu,
            metric_names=["PCITX", "PCIRX"],
            colors={
                "PCITX": "#d62728",
                "PCIRX": "#9467bd",
            },
            title_prefix="PCIe Bandwidth Over Time",
            ylabel="Bandwidth (MB/s)",
            output_plot=output_prefix.with_name(f"{output_prefix.name}_pcie.png"),
        )
    else:
        _plot_metric_group(
            series_by_gpu=series_by_gpu,
            metric_names=["SM", "MEM"],
            colors={
                "SM": "#1f77b4",
                "MEM": "#2ca02c",
            },
            title_prefix="GPU Utilization Metrics Over Time",
            ylabel="Utilization",
            output_plot=output_prefix.with_name(f"{output_prefix.name}_utilization.png"),
            ylim_top=1.05,
        )

    print(f"sm_log_path={sm_log_path}")
    print(f"log_kind={log_kind}")
    print(f"gpu_count={len(series_by_gpu)}")
    print(f"output_prefix={output_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
