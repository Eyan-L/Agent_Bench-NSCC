#!/usr/bin/env python3
"""Sample vLLM / SGLang server metrics and emit per-iteration scheduler state
as JSONL — running/waiting sequence counts, token throughput, KV cache usage.

Two collection modes:

    --source metrics   poll the engine's Prometheus /metrics endpoint
                       (default). True iteration-level resolution: gauges
                       like vllm:num_requests_running update every scheduler
                       step, so polling at --interval 0.1 effectively gives
                       you a per-iteration trace. Independent of log format
                       drift across engine versions.

    --source log       tail the engine's stdout log file and regex out the
                       periodic scheduler summary. Falls behind the actual
                       iteration cadence — vLLM's LoggingStatLogger has a
                       hardcoded 5-10s `local_interval`. Useful when /metrics
                       isn't reachable or when you want exact engine-emitted
                       numbers.

Stdlib-only. Designed to run as a sidecar:

    python3 backend/engine_log_tailer.py \
        --engine vllm --source metrics \
        --base-url http://127.0.0.1:8000 \
        --output $LOG_DIR/engine_metrics.jsonl &

SIGTERM/SIGINT trigger a clean flush + exit. In log mode the tailer seeks to
end-of-file on start, follows appends indefinitely, and detects rotation via
inode change.

Output schema (one JSON per parsed line; unknown engine fields stay null):

    {
      "ts_ns": <int>,                 # tailer wall-clock at parse time
      "iso":   "2026-05-07T03:27:00.560Z",
      "engine_ts": "2026-05-07T03:27:00",   # engine-side timestamp if parsable
      "engine": "vllm" | "sglang",
      "running":      <int|null>,     # reqs currently being decoded
      "waiting":      <int|null>,     # reqs queued (vLLM "Pending" or "Waiting")
      "swapped":      <int|null>,     # reqs in CPU KV (vLLM-only)
      "prompt_tps":   <float|null>,   # avg prompt throughput tokens/s
      "gen_tps":      <float|null>,   # avg generation throughput tokens/s
      "kv_cache_pct": <float|null>,   # GPU KV cache utilization (0–100)
      "cpu_cache_pct":<float|null>,   # CPU KV cache (0–100)
      "tokens_in":    <int|null>,     # SGLang `#token` (running + cache)
      "raw":          "<original line>"
    }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Prometheus /metrics polling. vLLM and SGLang both expose Prometheus-format
# gauges/counters that the engine updates every scheduler iteration.
# ---------------------------------------------------------------------------

# Map Prometheus metric names → (engine, normalized field). Multiple metric
# names can map to the same field for cross-engine comparability.
_METRICS_MAP: Dict[str, Tuple[str, str]] = {
    # vLLM (V0 and V1 both expose these names; V1 also exposes
    # vllm:num_running_requests etc. — we accept both spellings).
    "vllm:num_requests_running":      ("vllm", "running"),
    "vllm:num_running_requests":      ("vllm", "running"),
    "vllm:num_requests_waiting":      ("vllm", "waiting"),
    "vllm:num_waiting_requests":      ("vllm", "waiting"),
    "vllm:num_requests_swapped":      ("vllm", "swapped"),
    "vllm:gpu_cache_usage_perc":      ("vllm", "kv_cache_pct"),
    "vllm:cpu_cache_usage_perc":      ("vllm", "cpu_cache_pct"),
    "vllm:prompt_tokens_total":       ("vllm", "prompt_tokens_total"),
    "vllm:generation_tokens_total":   ("vllm", "generation_tokens_total"),
    "vllm:num_preemptions_total":     ("vllm", "preemptions_total"),
    "vllm:iteration_tokens_total":    ("vllm", "iteration_tokens_total"),
    # SGLang.
    "sglang:num_running_reqs":        ("sglang", "running"),
    "sglang:num_queue_reqs":          ("sglang", "waiting"),
    "sglang:token_usage":             ("sglang", "kv_cache_pct"),
    "sglang:num_prompt_tokens_total": ("sglang", "prompt_tokens_total"),
    "sglang:num_generation_tokens_total": ("sglang", "generation_tokens_total"),
    "sglang:gen_throughput":          ("sglang", "gen_tps"),
}

_PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([\-0-9eE.+infaN]+)\s*$")


def _fetch_metrics(url: str, timeout: float = 2.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def _parse_prometheus(body: str) -> Dict:
    """Parse Prometheus exposition format. Aggregates labelled series by
    summing same-name series — vLLM's V1 metrics include `model_name` and
    `engine` labels but for our purposes we just want the total."""
    out: Dict = {"by_name": {}}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        name, _labels, val = m.group(1), m.group(2), m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        out["by_name"][name] = out["by_name"].get(name, 0.0) + v
    return out


def _project_metrics(snapshot: Dict, engine_hint: str) -> Dict:
    """Project a Prometheus snapshot onto the canonical schema used by the
    JSONL output (running/waiting/etc.)."""
    rec: Dict = {}
    by_name = snapshot.get("by_name") or {}
    for raw_name, value in by_name.items():
        match = _METRICS_MAP.get(raw_name)
        if not match:
            continue
        engine, field = match
        if engine_hint and engine != engine_hint:
            continue
        # Counts are integers semantically; throughput / pct stay floats.
        if field in ("running", "waiting", "swapped",
                     "prompt_tokens_total", "generation_tokens_total",
                     "preemptions_total", "iteration_tokens_total"):
            rec[field] = int(value)
        else:
            rec[field] = round(value, 4)
    return rec


# ---------------------------------------------------------------------------
# Engine-specific parsers
# ---------------------------------------------------------------------------

# vLLM (V0 and recent V1 builds).
# Sample lines we want to match:
#   "Avg prompt throughput: 5234.5 tokens/s, Avg generation throughput: 432.1 tokens/s, Running: 8 reqs, Swapped: 0 reqs, Pending: 2 reqs, GPU KV cache usage: 45.6%, CPU KV cache usage: 0.0%"
#   "running: 8 waiting: 2 swapped: 0 prompt_throughput: 5234.5 generation_throughput: 432.1 gpu_kv_cache_usage: 45.6"
# We use a forgiving key=value extractor so version drift in punctuation,
# casing, or label spelling doesn't break the parser.

_VLLM_TS_RE = re.compile(
    r"^\D*(\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})"
)
_VLLM_KEY_PATTERNS: Dict[str, re.Pattern] = {
    "running":       re.compile(r"[Rr]unning[:\s]+(\d+)"),
    "waiting":       re.compile(r"(?:[Ww]aiting|[Pp]ending)[:\s]+(\d+)"),
    "swapped":       re.compile(r"[Ss]wapped[:\s]+(\d+)"),
    "prompt_tps":    re.compile(
        r"(?:[Aa]vg\s+)?[Pp]rompt(?:\s+throughput)?[:\s]+([\d.]+)"
    ),
    "gen_tps":       re.compile(
        r"(?:[Aa]vg\s+)?[Gg]eneration(?:\s+throughput)?[:\s]+([\d.]+)"
    ),
    "kv_cache_pct":  re.compile(
        r"GPU\s+KV\s+cache(?:\s+usage)?[:\s]+([\d.]+)"
    ),
    "cpu_cache_pct": re.compile(
        r"CPU\s+KV\s+cache(?:\s+usage)?[:\s]+([\d.]+)"
    ),
}


def _parse_vllm(line: str) -> Optional[Dict]:
    # Heuristic: only emit a record if at least one of running/waiting/tps is
    # captured. Otherwise we'd emit one record per startup line.
    out: Dict = {}
    for key, pat in _VLLM_KEY_PATTERNS.items():
        m = pat.search(line)
        if m:
            v = m.group(1)
            try:
                out[key] = int(v) if "." not in v else float(v)
            except ValueError:
                continue
    if not any(k in out for k in ("running", "waiting", "gen_tps", "prompt_tps")):
        return None

    out["engine"] = "vllm"

    # Engine-side timestamp (vLLM logs "MM-DD HH:MM:SS").
    m = _VLLM_TS_RE.search(line)
    if m:
        out["engine_ts"] = f"{m.group(1)} {m.group(2)}"
    return out


# SGLang. Sample line:
#   "[router] Decode batch. #running-req: 12, #queue-req: 3, "
#   "gen throughput (token/s): 234.5, #token: 12345, "
#   "#cache-hit-rate: 0.81, #cache-hit (token/s): 1234.5"
# Mixed-case decode/prefill messages also fire — we want them all.

_SGL_KEY_PATTERNS: Dict[str, re.Pattern] = {
    "running":      re.compile(r"#running-req[:\s]+(\d+)"),
    "waiting":      re.compile(r"#queue-req[:\s]+(\d+)"),
    "gen_tps":      re.compile(
        r"gen\s+throughput\s*\(token/s\)[:\s]+([\d.]+)"
    ),
    "prompt_tps":   re.compile(
        r"prefill\s+throughput\s*\(token/s\)[:\s]+([\d.]+)"
    ),
    "tokens_in":    re.compile(r"#token[:\s]+(\d+)"),
    "kv_cache_pct": re.compile(r"token\s+usage[:\s]+([\d.]+)"),
}
_SGL_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


def _parse_sglang(line: str) -> Optional[Dict]:
    out: Dict = {}
    for key, pat in _SGL_KEY_PATTERNS.items():
        m = pat.search(line)
        if m:
            v = m.group(1)
            try:
                out[key] = int(v) if "." not in v else float(v)
            except ValueError:
                continue
    if not any(k in out for k in ("running", "waiting", "gen_tps", "prompt_tps")):
        return None
    out["engine"] = "sglang"
    m = _SGL_TS_RE.search(line)
    if m:
        out["engine_ts"] = m.group(1)
    return out


PARSERS = {"vllm": _parse_vllm, "sglang": _parse_sglang}


# ---------------------------------------------------------------------------
# File tailer — handles rotation by watching the inode of the log path.
# ---------------------------------------------------------------------------

def _open_at_end(path: str) -> Tuple[Optional[object], Optional[int]]:
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
        fh.seek(0, os.SEEK_END)
        ino = os.fstat(fh.fileno()).st_ino
        return fh, ino
    except (FileNotFoundError, OSError):
        return None, None


def _maybe_reopen(path: str, fh, ino: Optional[int]) -> Tuple[object, int]:
    """If the file at `path` has a different inode than `ino`, close and
    re-open from the start (fresh log → start consuming). Caller guarantees
    `path` exists when called from the tail loop."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return fh, ino  # waiting for the path to reappear
    if ino is None or st.st_ino != ino:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        new_fh = open(path, "r", encoding="utf-8", errors="replace")
        return new_fh, st.st_ino
    return fh, ino


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=list(PARSERS), default="vllm")
    ap.add_argument(
        "--source", choices=("metrics", "log"), default="metrics",
        help="metrics: poll Prometheus /metrics (default, iteration-level). "
             "log: tail the engine log file (engine-rate-limited).",
    )
    ap.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="engine base URL (only used in --source metrics mode)",
    )
    ap.add_argument(
        "--metrics-path", default="/metrics",
        help="metrics endpoint path (default /metrics)",
    )
    ap.add_argument(
        "--interval", type=float, default=0.1,
        help="seconds between metrics polls in --source metrics mode "
             "(default 0.1 — captures iteration-level changes)",
    )
    ap.add_argument(
        "--emit-unchanged", action="store_true",
        help="emit a record every poll even if values are identical to the "
             "previous sample (default: only emit on change)",
    )
    ap.add_argument(
        "--log", default=None,
        help="path to engine log file (required for --source log)",
    )
    ap.add_argument("--output", required=True, help="JSONL output path")
    ap.add_argument(
        "--from-start", action="store_true",
        help="log mode: parse from beginning of file instead of seeking to EOF",
    )
    ap.add_argument(
        "--poll-interval", type=float, default=0.5,
        help="log mode: seconds between read attempts when idle (default 0.5)",
    )
    args = ap.parse_args()

    if args.source == "log" and not args.log:
        ap.error("--source log requires --log <path>")

    _install_signal_handlers()
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.source == "metrics":
        return _run_metrics_mode(args)
    return _run_log_mode(args)


def _run_metrics_mode(args) -> int:
    url = args.base_url.rstrip("/") + args.metrics_path

    # Wait briefly for the endpoint to come up — the engine may still be
    # finishing weight load when this sidecar starts.
    deadline = time.monotonic() + 60.0
    while not _Stop.flag and time.monotonic() < deadline:
        if _fetch_metrics(url, timeout=1.0) is not None:
            break
        time.sleep(0.5)
    else:
        if not _Stop.flag:
            sys.stderr.write(
                f"[engine_log_tailer] /metrics not reachable at {url} after 60s\n"
            )
            return 1

    last_payload: Optional[Dict] = None
    n_emitted = 0

    with open(args.output, "a", buffering=1) as out:
        out.write(json.dumps({
            "_meta":         True,
            "started_iso":   datetime.now(tz=timezone.utc).isoformat(),
            "started_ts_ns": time.time_ns(),
            "engine":        args.engine,
            "source":        "metrics",
            "metrics_url":   url,
            "interval":      args.interval,
        }) + "\n")

        while not _Stop.flag:
            body = _fetch_metrics(url, timeout=2.0)
            if body is None:
                # Endpoint hiccup — record and keep going so analysis can
                # tell apart "engine quiet" from "scraper down".
                _Stop.wait_for(args.interval) if hasattr(_Stop, 'wait_for') \
                    else time.sleep(args.interval)
                continue

            snapshot = _parse_prometheus(body)
            rec = _project_metrics(snapshot, args.engine)
            if not rec:
                time.sleep(args.interval)
                continue

            # Skip duplicate consecutive emissions to keep the JSONL compact —
            # vLLM's gauges only change on iteration boundaries, so polling at
            # 0.1s with deduplication still captures every iteration.
            if not args.emit_unchanged and last_payload == rec:
                time.sleep(args.interval)
                continue
            last_payload = dict(rec)

            now_ns = time.time_ns()
            rec["ts_ns"] = now_ns
            rec["iso"] = datetime.fromtimestamp(
                now_ns / 1e9, tz=timezone.utc
            ).isoformat()
            rec["engine"] = args.engine
            rec["source"] = "metrics"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_emitted += 1
            time.sleep(args.interval)

    sys.stderr.write(
        f"[engine_log_tailer] metrics-mode exit; emitted {n_emitted} record(s)\n"
    )
    return 0


def _run_log_mode(args) -> int:
    parser = PARSERS[args.engine]

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Wait briefly for the log file to appear if the engine is still
    # starting up. Don't block forever — the tailer should fail loudly if
    # the log path is wrong.
    deadline = time.monotonic() + 30.0
    while not os.path.exists(args.log) and time.monotonic() < deadline:
        if _Stop.flag:
            return 0
        time.sleep(0.5)
    if not os.path.exists(args.log):
        sys.stderr.write(
            f"[engine_log_tailer] log file did not appear within 30s: {args.log}\n"
        )
        return 1

    if args.from_start:
        fh = open(args.log, "r", encoding="utf-8", errors="replace")
        ino = os.fstat(fh.fileno()).st_ino
    else:
        fh, ino = _open_at_end(args.log)
        if fh is None:
            sys.stderr.write(
                f"[engine_log_tailer] cannot open log: {args.log}\n"
            )
            return 1

    with open(args.output, "a", buffering=1) as out:
        out.write(json.dumps({
            "_meta":         True,
            "started_iso":   datetime.now(tz=timezone.utc).isoformat(),
            "started_ts_ns": time.time_ns(),
            "engine":        args.engine,
            "log_path":      args.log,
            "from_start":    bool(args.from_start),
        }) + "\n")

        n_emitted = 0
        while not _Stop.flag:
            line = fh.readline() if fh else ""
            if not line:
                # Idle — check for rotation, then sleep briefly.
                fh, ino = _maybe_reopen(args.log, fh, ino)
                _Stop.flag = _Stop.flag or False  # placeholder for clarity
                if _Stop.flag:
                    break
                time.sleep(args.poll_interval)
                continue

            rec = parser(line.rstrip("\n"))
            if rec is None:
                continue
            now_ns = time.time_ns()
            rec["ts_ns"] = now_ns
            rec["iso"] = datetime.fromtimestamp(
                now_ns / 1e9, tz=timezone.utc
            ).isoformat()
            rec["raw"] = line.rstrip("\n")[:600]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_emitted += 1

    sys.stderr.write(
        f"[engine_log_tailer] exit; emitted {n_emitted} record(s) -> "
        f"{args.output}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
