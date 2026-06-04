#!/usr/bin/env python3
"""Sample CPU + GPU utilization at a fixed interval and append JSONL.

Stdlib-only on purpose — runs anywhere python3 + (optionally) cAdvisor and
nvidia-smi exist. Designed to be spawned as a sidecar around a containerized
run:

    python3 -m agent_bench.telemetry.util_sampler --output util.jsonl --interval 1.0 &
    PID=$!
    # ... run benchmark ...
    kill -TERM $PID

Or with --duration N for a fixed window. SIGTERM/SIGINT trigger a clean
flush + exit.

CPU is read from /proc/stat (Linux). On platforms without /proc/stat the
"cpu" field is null. GPU is read from `nvidia-smi --query-gpu`; if the
binary is missing the "gpu" field is an empty list.

Per-container utilization uses cAdvisor API and accepts any of:
    --llm-container <name>     # model-serving container name
    --client-container <name>  # gateway / client container name
    --tools-container <name>   # task / sandbox container name

Each option can be repeated.

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
        "mem_pct":      <float>,
        "by_group":     {                 # only present if --*-pid was set
          "llm_engine": {"pct": <float>, "rss_mb": <int>, "n_proc": <int>},
          "client":     {"pct": <float>, "rss_mb": <int>, "n_proc": <int>},
          "tools":      {"pct": <float>, "rss_mb": <int>, "n_proc": <int>},
          "other":      {"pct": <float>}  # host - sum(known)
        }
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
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
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
        with open("/proc/stat", "r", encoding="utf-8") as fh:
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
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
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
# Per-container CPU/RSS via docker stats.
# ---------------------------------------------------------------------------

def _ncpu() -> int:
    return os.cpu_count() or 1


def _read_docker_stats(container_name: str, port: int = 8088, timeout: float = 2.0) -> Optional[Dict]:
    """Fetch metrics from cAdvisor API (replacing docker stats)."""
    #   containers/docker/<name>
    url = f"http://localhost:{port}/api/v1.3/docker/{container_name}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None

    if not data:
        return None

    # The response is usually a dict keyed by container path (e.g. "/docker/<id>")
    info = next(iter(data.values())) if isinstance(data, dict) else data
    stats = info.get("stats")
    if not stats or len(stats) < 2:
        return None

    curr = stats[-1]
    prev = stats[-2]

    # CPU usage in nanoseconds
    d_cpu = curr["cpu"]["usage"]["total"] - prev["cpu"]["usage"]["total"]

    def _parse_ts(ts: str) -> float:
        # Handle RFC3339 with various precision
        ts = ts.replace("Z", "+00:00")
        if "." in ts:
            base, frac = ts.split(".")
            # truncate/pad frac to 6 digits for %f, keeping only trailing offset
            # (e.g. 2024-05-24T12:34:56.789123456Z -> 2024-05-24T12:34:56.789123+00:00)
            frac_part, sep, offset = frac.partition("+")
            if not sep:
                frac_part, sep, offset = frac.partition("-")
            frac_part = (frac_part[:6] + "000000")[:6]
            ts = f"{base}.{frac_part}{sep}{offset}"
        return datetime.fromisoformat(ts).timestamp()

    try:
        d_time = _parse_ts(curr["timestamp"]) - _parse_ts(prev["timestamp"])
    except Exception:
        return None

    if d_time <= 0:
        return None

    # CPU % of total host (as reported by cadvisor stats which is cumulative across cores)
    cpu_pct = round((d_cpu / (d_time * 1e9)) * 100.0, 2)

    mem_used_bytes = curr["memory"]["usage"]
    # memory limit is in the spec block
    mem_total_bytes = info.get("spec", {}).get("memory", {}).get("limit", 0)
    # 0 or very large numbers mean unlimited
    is_unlimited = mem_total_bytes == 0 or mem_total_bytes > 2**60

    mem_used_mb = mem_used_bytes // (1024 * 1024)
    mem_total_mb = (mem_total_bytes // (1024 * 1024)) if not is_unlimited else None
    mem_pct = round(100.0 * mem_used_bytes / mem_total_bytes, 2) if not is_unlimited and mem_total_bytes > 0 else None

    return {
        "name": container_name,
        "pct": cpu_pct,
        "mem_used_mb": mem_used_mb,
        "mem_total_mb": mem_total_mb,
        "mem_pct": mem_pct,
        "n_proc": None,  # cadvisor v1.3 doesn't easily show PID count in stats block
    }


def _read_container_groups(container_groups: Dict[str, List[str]], port: int = 8088) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for role, names in container_groups.items():
        sampled = []
        for name in names:
            if not name:
                continue
            stats = _read_docker_stats(name, port=port)
            if stats is not None:
                sampled.append(stats)
        if not sampled:
            continue
        mem_used_mb = sum((stat.get("mem_used_mb") or 0) for stat in sampled)
        mem_total_mb = sum((stat.get("mem_total_mb") or 0) for stat in sampled)
        mem_pct = round(100.0 * mem_used_mb / mem_total_mb, 2) if mem_total_mb else None
        out[role] = {
            "pct": round(sum((stat.get("pct") or 0.0) for stat in sampled), 2),
            "mem_used_mb": mem_used_mb,
            "mem_total_mb": mem_total_mb or None,
            "mem_pct": mem_pct,
            "n_proc": sum((stat.get("n_proc") or 0) for stat in sampled) or None,
            "containers": [stat["name"] for stat in sampled],
        }
    return out


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
            [
                "nvidia-smi",
                f"--query-gpu={_NVIDIA_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
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
            "index": _i(cols[0]),
            "name": cols[1],
            "util_gpu_pct": _f(cols[2]),
            "util_mem_pct": _f(cols[3]),
            "mem_used_mb": _i(cols[4]),
            "mem_total_mb": _i(cols[5]),
            "power_w": _f(cols[6]),
            "temp_c": _f(cols[7]),
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


def _build_record(
    prev_stat,
    cur_stat,
    *,
    cur_groups: Optional[Dict[str, Dict]] = None,
    ncpu: int = 1,
) -> Dict:
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
            "pct": pcts[0] if pcts else None,
            "per_cpu_pct": pcts[1:] if len(pcts) > 1 else [],
            "loadavg": _loadavg(),
            "mem_total_mb": mem_total,
            "mem_used_mb": mem_used,
            "mem_pct": (
                round(100.0 * mem_used / mem_total, 2)
                if mem_total and mem_used is not None else None
            ),
        }
        # Per-container breakdown.
        if cur_groups is not None:
            by_group: Dict[str, Dict] = {}
            sum_pct = 0.0
            for name, stats in cur_groups.items():
                pct = stats.get("pct")
                by_group[name] = {
                    "pct": pct,
                    "rss_mb": stats.get("mem_used_mb"),
                    "n_proc": stats.get("n_proc"),
                }
                if pct is not None:
                    sum_pct += pct
            host_pct = cpu_block["pct"] or 0.0
            # cAdvisor reports CPU as "% of one core" and can exceed 100.
            # Scale host-wide CPU to the same unit before deriving "other".
            by_group["other"] = {
                "pct": round(max(0.0, host_pct * ncpu - sum_pct), 2),
            }
            cpu_block["by_group"] = by_group
    return {
        "ts_ns": now_ns,
        "iso": iso,
        "cpu": cpu_block,
        "gpu": _read_nvidia_smi(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True,
                    help="JSONL file to append samples to")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between samples (default 1.0)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = until SIGTERM)")
    ap.add_argument("--cadvisor-port", type=int, 
                    default=int(os.environ.get("AB_CADVISOR_PORT", "8088")),
                    help="cAdvisor API port (default 8088 or AB_CADVISOR_PORT env)")
    ap.add_argument("--llm-container", type=str, action="append", default=[],
                    help="model-serving container name. Repeatable.")
    ap.add_argument("--client-container", type=str, action="append", default=[],
                    help="agent / gateway container name. Repeatable.")
    ap.add_argument("--tools-container", type=str, action="append", default=[],
                    help="tool / sandbox container name. Repeatable.")
    args = ap.parse_args()

    _install_signal_handlers()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    ncpu = _ncpu()
    container_groups = {
        "llm_engine": list(args.llm_container),
        "client": list(args.client_container),
        "tools": list(args.tools_container),
    }
    have_groups = bool(any(container_groups.values()))

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    prev_stat = _read_proc_stat()

    # Open in line-buffered mode so kill -TERM mid-run still leaves the file
    # in a parseable state. JSONL is append-only so each sample is durable
    # the moment its newline hits disk.
    with open(args.output, "a", buffering=1, encoding="utf-8") as fh:
        # Write a header line so consumers can identify the stream.
        header = {
            "_meta": True,
            "started_iso": datetime.now(tz=timezone.utc).isoformat(),
            "started_ts_ns": time.time_ns(),
            "interval": args.interval,
            "pid": os.getpid(),
            "host": os.uname().nodename if hasattr(os, "uname") else None,
            "nvidia_smi": _have_nvidia_smi(),
            "ncpu": ncpu,
            "container_groups": {
                "llm_engine": list(args.llm_container),
                "client": list(args.client_container),
                "tools": list(args.tools_container),
            },
        }
        fh.write(json.dumps(header) + "\n")

        while not _Stop.flag:
            cur_stat = _read_proc_stat()
            cur_groups: Optional[Dict[str, Dict]] = None
            if have_groups:
                cur_groups = _read_container_groups(container_groups, port=args.cadvisor_port)
                if not cur_groups:
                    cur_groups = None

            rec = _build_record(
                prev_stat,
                cur_stat,
                cur_groups=cur_groups,
                ncpu=ncpu,
            )
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