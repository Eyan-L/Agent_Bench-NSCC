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

Per-group CPU split (PID-tree accounting). Pass any of:
    --llm-pid <pid>        # vLLM root pid (engine; sums itself + descendants)
    --client-pid <pid>     # driver / agent pid (the harness Python process)
    --gateway-pid <pid>    # OpenClaw gateway pid (its descendants are tools)

Each --*-pid can be repeated. Tool processes are computed as
"descendants of {client + gateway} minus the engine tree". CPU% per
group is summed jiffies in the group between consecutive samples,
divided by total host jiffies, ×100×ncpu so values are comparable to the
host-wide cpu.pct (saturating at ncpu×100).

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
# Per-process CPU/RSS via /proc/<pid>/stat,status — no psutil dependency.
# Accumulates utime+stime jiffies per (group, sample); deltas across samples
# divided by total /proc/stat jiffies give a CPU-percent comparable to the
# host-wide cpu.pct (with the same x100 normalization, except per-group is
# left in "% of one core" so tools_pct + engine_pct + client_pct + other_pct
# = total host work in core-percent).
# ---------------------------------------------------------------------------

def _read_pid_stat(pid: int) -> Optional[Tuple[int, int, int, int]]:
    """Return (utime, stime, ppid, rss_pages) for `pid`, or None if gone."""
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            data = fh.read()
    except (OSError, FileNotFoundError):
        return None
    # comm may contain spaces and parentheses — split off the (..) field first.
    rp = data.rfind(")")
    if rp == -1:
        return None
    rest = data[rp + 2:].split()
    # Fields after comm (man 5 proc, 1-indexed):
    #   3 state, 4 ppid, 14 utime, 15 stime, 24 rss
    # 0-indexed in `rest`: state=0, ppid=1, utime=11, stime=12, rss=21
    try:
        ppid = int(rest[1])
        utime = int(rest[11])
        stime = int(rest[12])
        rss_pages = int(rest[21])
    except (IndexError, ValueError):
        return None
    return utime, stime, ppid, rss_pages


def _list_descendants(roots: List[int]) -> List[int]:
    """Return a deduped list of root PIDs and all descendants by walking
    /proc once and building a parent -> children index. O(N) per sample.

    Excludes pids in `roots` that are gone; descendants are computed off
    whatever's currently alive.
    """
    if not roots:
        return []
    try:
        all_pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return list(set(roots))
    children: Dict[int, List[int]] = {}
    for pid in all_pids:
        s = _read_pid_stat(pid)
        if s is None:
            continue
        _, _, ppid, _ = s
        children.setdefault(ppid, []).append(pid)
    out: set = set()
    stack = list(roots)
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        stack.extend(children.get(pid, []))
    return sorted(out)


def _group_jiffies_rss(pids: List[int]) -> Tuple[int, int, int]:
    """Sum utime+stime jiffies and RSS pages across `pids`. Returns
    (jiffies, rss_pages, n_alive). Dead pids are silently skipped."""
    jiffies = 0
    rss_pages = 0
    n = 0
    for pid in pids:
        s = _read_pid_stat(pid)
        if s is None:
            continue
        u, st, _ppid, r = s
        jiffies += u + st
        rss_pages += r
        n += 1
    return jiffies, rss_pages, n


def _ncpu() -> int:
    return os.cpu_count() or 1


def _page_size_bytes() -> int:
    try:
        return os.sysconf("SC_PAGESIZE")
    except (OSError, ValueError):
        return 4096


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


def _stat_total_jiffies(stat: List[Tuple[int, int]]) -> int:
    """Sum the aggregate (idx 0) total jiffies from _read_proc_stat output."""
    if not stat:
        return 0
    return stat[0][1]


def _group_pct(prev_j: int, cur_j: int, prev_total: int, cur_total: int,
               ncpu: int) -> float:
    """Per-group jiffy delta divided by host total jiffy delta, scaled to a
    "% of one core × ncpu" so it sums comparably with cpu.pct (host-wide)."""
    d_total = cur_total - prev_total
    if d_total <= 0:
        return 0.0
    d_group = max(cur_j - prev_j, 0)
    return round(100.0 * ncpu * d_group / d_total, 2)


def _build_record(
    prev_stat,
    cur_stat,
    *,
    prev_groups: Optional[Dict[str, int]] = None,
    cur_groups: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ncpu: int = 1,
    page_bytes: int = 4096,
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
            "pct":          pcts[0] if pcts else None,
            "per_cpu_pct":  pcts[1:] if len(pcts) > 1 else [],
            "loadavg":      _loadavg(),
            "mem_total_mb": mem_total,
            "mem_used_mb":  mem_used,
            "mem_pct":      (round(100.0 * mem_used / mem_total, 2)
                             if mem_total and mem_used is not None else None),
        }
        # Per-group breakdown (only if pids configured AND we have a delta).
        if cur_groups is not None:
            prev_total = _stat_total_jiffies(prev_stat) if prev_stat else 0
            cur_total = _stat_total_jiffies(cur_stat)
            by_group: Dict[str, Dict] = {}
            sum_pct = 0.0
            for name, (cur_j, rss_pages, n_alive) in cur_groups.items():
                pj = (prev_groups or {}).get(name, cur_j)
                pct = _group_pct(pj, cur_j, prev_total, cur_total, ncpu)
                by_group[name] = {
                    "pct":    pct,
                    "rss_mb": (rss_pages * page_bytes) // (1024 * 1024),
                    "n_proc": n_alive,
                }
                sum_pct += pct
            host_pct = cpu_block["pct"] or 0.0
            # "other" can go negative if descendants drift between samples
            # (race between _list_descendants and _group_jiffies_rss); clamp.
            by_group["other"] = {
                "pct": round(max(0.0, host_pct - sum_pct), 2),
            }
            cpu_block["by_group"] = by_group
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
    ap.add_argument("--llm-pid", type=int, action="append", default=[],
                    help="Engine root pid (vLLM). Repeatable.")
    ap.add_argument("--client-pid", type=int, action="append", default=[],
                    help="Driver / agent root pid. Repeatable.")
    ap.add_argument("--gateway-pid", type=int, action="append", default=[],
                    help="OpenClaw gateway root pid. Tools = its descendants. "
                         "Repeatable.")
    args = ap.parse_args()

    _install_signal_handlers()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)

    have_groups = bool(args.llm_pid or args.client_pid or args.gateway_pid)
    ncpu = _ncpu()
    page_bytes = _page_size_bytes()
    prev_groups: Dict[str, int] = {}

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
            "ncpu":          ncpu,
            "pid_groups":    {
                "llm_engine": list(args.llm_pid),
                "client":     list(args.client_pid),
                "gateway":    list(args.gateway_pid),
            },
        }
        fh.write(json.dumps(header) + "\n")

        while not _Stop.flag:
            cur_stat = _read_proc_stat()
            cur_groups: Optional[Dict[str, Tuple[int, int, int]]] = None
            if have_groups:
                # Resolve descendants this sample (process tree changes over time).
                engine_pids = _list_descendants(args.llm_pid)
                client_pids = _list_descendants(args.client_pid)
                gateway_pids = _list_descendants(args.gateway_pid)
                # Tools = under (client + gateway), excluding engine.
                engine_set = set(engine_pids)
                tools_pids = [
                    p for p in (set(client_pids) | set(gateway_pids))
                    if p not in engine_set
                    and p not in args.client_pid          # exclude client roots
                    and p not in args.gateway_pid         # exclude gateway roots
                ]
                cur_groups = {
                    "llm_engine": _group_jiffies_rss(engine_pids),
                    "client":     _group_jiffies_rss(args.client_pid),
                    "tools":      _group_jiffies_rss(tools_pids),
                }
                # Add in the gateway's own jiffies under "client" so
                # client = (driver + gateway roots) and tools = their kids.
                gw_j, gw_rss, gw_n = _group_jiffies_rss(args.gateway_pid)
                cl_j, cl_rss, cl_n = cur_groups["client"]
                cur_groups["client"] = (cl_j + gw_j, cl_rss + gw_rss, cl_n + gw_n)

            rec = _build_record(
                prev_stat, cur_stat,
                prev_groups=prev_groups if prev_groups else None,
                cur_groups=cur_groups,
                ncpu=ncpu, page_bytes=page_bytes,
            )
            if cur_groups is not None:
                prev_groups = {k: v[0] for k, v in cur_groups.items()}
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
