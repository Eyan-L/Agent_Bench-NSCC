#!/usr/bin/env python3
"""Sample CPU + GPU utilization at a fixed interval and append JSONL.

Stdlib-only on purpose — runs anywhere python3 + (optionally) nvidia-smi
exist. Designed to be spawned as a sidecar around an OC run:

    python3 scripts/util_sampler.py --output util.jsonl --interval 1.0 &
    PID=$!
    # ... run benchmark ...
    kill -TERM $PID

Or with --duration N for a fixed window. SIGTERM/SIGINT trigger a clean
flush + exit.

CPU is read from /proc/stat (Linux). On platforms without /proc/stat the
"cpu" field is null. GPU is read from `nvidia-smi --query-gpu`; if the
binary is missing the "gpu" field is an empty list.

Output format — one JSON object per sample, one line:

    {
      "ts_ns": <int>,
      "iso":   "2026-05-01T12:34:56.789012+00:00",
      "cpu":   {
        "pct":          <float | null>,   # aggregate %, post-first-sample
        "per_cpu_pct":  [<float>, ...],
        "loadavg":      [<l1>, <l5>, <l15>],
        "mem_total_mb": <int>,
        "mem_used_mb":  <int>,
        "mem_pct":      <float>
      },
      "gpu":  [
        {"index": 0, "name": "...", "util_gpu_pct": <float>,
         "util_mem_pct": <float>, "mem_used_mb": <int>,
         "mem_total_mb": <int>, "power_w": <float | null>,
         "temp_c": <float | null>},
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CPU sampling — /proc/stat delta between consecutive reads.
# ---------------------------------------------------------------------------

def _read_proc_stat() -> Optional[List[Tuple[int, int]]]:
    """Return [(idle_total, total)] for index 0 = aggregate, then per-core.

    None if /proc/stat is unavailable.
    """
    try:
        with open("/proc/stat", "r") as fh:
            lines = fh.readlines()
    except (OSError, FileNotFoundError):
        return None

    out: List[Tuple[int, int]] = []
    for line in lines:
        if not line.startswith("cpu"):
            break
        parts = line.split()
        # cpu user nice system idle iowait irq softirq steal guest guest_nice
        nums = [int(x) for x in parts[1:]]
        if len(nums) < 4:
            continue
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
        total = sum(nums)
        out.append((idle, total))
    return out or None


def _cpu_pct(prev: List[Tuple[int, int]], cur: List[Tuple[int, int]]) -> List[float]:
    out: List[float] = []
    for (pi, pt), (ci, ct) in zip(prev, cur):
        d_total = ct - pt
        d_idle = ci - pi
        if d_total <= 0:
            out.append(0.0)
        else:
            out.append(round(100.0 * (d_total - d_idle) / d_total, 2))
    return out


def _read_meminfo() -> Tuple[Optional[int], Optional[int]]:
    """Return (total_mb, used_mb). MemAvailable preferred over MemFree."""
    try:
        with open("/proc/meminfo", "r") as fh:
            lines = fh.readlines()
    except (OSError, FileNotFoundError):
        return None, None
    info: Dict[str, int] = {}
    for line in lines:
        k, _, rest = line.partition(":")
        v = rest.strip().split()
        if v and v[0].isdigit():
            info[k] = int(v[0])  # kB
    total = info.get("MemTotal")
    avail = info.get("MemAvailable") or info.get("MemFree")
    if total is None or avail is None:
        return None, None
    return total // 1024, (total - avail) // 1024


def _loadavg() -> Optional[List[float]]:
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# GPU sampling — nvidia-smi CSV output.
# ---------------------------------------------------------------------------

_NVIDIA_QUERY = (
    "index,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,power.draw,temperature.gpu"
)


def _have_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _read_nvidia_smi(timeout: float = 2.0) -> List[Dict]:
    if not _have_nvidia_smi():
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={_NVIDIA_QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    out: List[Dict] = []
    for line in proc.stdout.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 8:
            continue
        def _f(s: str) -> Optional[float]:
            try:
                return float(s)
            except ValueError:
                return None
        def _i(s: str) -> Optional[int]:
            try:
                return int(float(s))
            except ValueError:
                return None
        out.append({
            "index":        _i(cols[0]),
            "name":         cols[1],
            "util_gpu_pct": _f(cols[2]),
            "util_mem_pct": _f(cols[3]),
            "mem_used_mb":  _i(cols[4]),
            "mem_total_mb": _i(cols[5]),
            "power_w":      _f(cols[6]),
            "temp_c":       _f(cols[7]),
        })
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class _Stop:
    flag = False


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        _Stop.flag = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _build_record(prev_stat, cur_stat) -> Dict:
    now_ns = time.time_ns()
    iso = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).isoformat()
    cpu_block: Optional[Dict]
    if cur_stat is None:
        cpu_block = None
    else:
        if prev_stat is None or len(prev_stat) != len(cur_stat):
            # First sample — no delta available yet.
            pcts = [0.0] * len(cur_stat)
        else:
            pcts = _cpu_pct(prev_stat, cur_stat)
        mem_total, mem_used = _read_meminfo()
        cpu_block = {
            "pct":          pcts[0] if pcts else None,
            "per_cpu_pct":  pcts[1:] if len(pcts) > 1 else [],
            "loadavg":      _loadavg(),
            "mem_total_mb": mem_total,
            "mem_used_mb":  mem_used,
            "mem_pct":      (round(100.0 * mem_used / mem_total, 2)
                             if mem_total and mem_used is not None else None),
        }
    return {
        "ts_ns": now_ns,
        "iso":   iso,
        "cpu":   cpu_block,
        "gpu":   _read_nvidia_smi(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True,
                    help="JSONL file to append samples to")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between samples (default 1.0)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = until SIGTERM)")
    args = ap.parse_args()

    _install_signal_handlers()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    prev_stat = _read_proc_stat()

    # Open in line-buffered mode so kill -TERM mid-run still leaves the file
    # in a parseable state. JSONL is append-only so each sample is durable
    # the moment its newline hits disk.
    with open(args.output, "a", buffering=1) as fh:
        # Write a header line so consumers can identify the stream.
        header = {
            "_meta": True,
            "started_iso":   datetime.now(tz=timezone.utc).isoformat(),
            "started_ts_ns": time.time_ns(),
            "interval":      args.interval,
            "pid":           os.getpid(),
            "host":          os.uname().nodename if hasattr(os, "uname") else None,
            "nvidia_smi":    _have_nvidia_smi(),
        }
        fh.write(json.dumps(header) + "\n")

        while not _Stop.flag:
            cur_stat = _read_proc_stat()
            rec = _build_record(prev_stat, cur_stat)
            prev_stat = cur_stat
            fh.write(json.dumps(rec) + "\n")
            if deadline is not None and time.monotonic() >= deadline:
                break
            # Wake up promptly on signal — sleep in small slices.
            slept = 0.0
            while slept < args.interval and not _Stop.flag:
                step = min(0.1, args.interval - slept)
                time.sleep(step)
                slept += step
    return 0


if __name__ == "__main__":
    sys.exit(main())
