#!/usr/bin/env python3
"""Compute TTFT SLO attainment from an LLMLoadgen result.json (JSONL).

Classification is by input length (prompt_tokens), since LLMLoadgen's
input1_output1 runs emit max_tokens=1 and there is no per-request output-length
to key off. TBT is reported as a best-effort average if any requests returned
>1 completion tokens, otherwise skipped.

Default SLO table (override via CLI if needed):

    bucket  input               TTFT SLO    TBT SLO
    S       <256                250 ms      100 ms
    M       <1024               400 ms      100 ms
    L       <=8192              2000 ms     100 ms
    XL      >8192               3000 ms     100 ms  (extension for long inputs)

TTFT per request is computed server-side as
    prefill_finished_ts - request_received_ts
from response.metadata.request_metrics, which covers queue + prefill delay.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Bucket:
    name: str
    # Closed upper bound on input tokens; None means unbounded (last bucket).
    input_upper: int | None
    upper_inclusive: bool  # whether input_upper itself belongs to this bucket
    ttft_slo_ms: float
    tbt_slo_ms: float


DEFAULT_BUCKETS: list[Bucket] = [
    Bucket("S", 256, False, 250.0, 100.0),
    Bucket("M", 1024, False, 400.0, 100.0),
    Bucket("L", 8192, True, 2000.0, 100.0),
    Bucket("XL", None, False, 3000.0, 100.0),
]


def classify(input_tokens: int, buckets: list[Bucket]) -> Bucket:
    for b in buckets:
        if b.input_upper is None:
            return b
        if b.upper_inclusive and input_tokens <= b.input_upper:
            return b
        if (not b.upper_inclusive) and input_tokens < b.input_upper:
            return b
    return buckets[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result_path", help="Path to LLMLoadgen result.json (JSONL).")
    p.add_argument(
        "--summary-out",
        default=None,
        help="Optional JSON path to write the full summary to.",
    )
    p.add_argument(
        "--print-violations",
        type=int,
        default=0,
        help="If >0, print up to N slowest SLO-violating requests per bucket.",
    )
    return p.parse_args()


@dataclass
class Record:
    input_tokens: int
    completion_tokens: int
    ttft_ms: float | None
    tbt_ms: float | None
    error: str
    # TTFT decomposition (ms); each may be None if the server didn't report it.
    queue_ms: float | None = None
    prefill_launch_delay_ms: float | None = None
    prefill_launch_latency_ms: float | None = None
    prefill_compute_ms: float | None = None
    send_ts: float | None = None  # client actual_send_time as epoch seconds
    bucket: Bucket | None = None


_ISO_RE = __import__("re").compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})?$"
)


def _parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    import datetime as _dt

    m = _ISO_RE.match(ts.strip())
    if not m:
        return None
    head = m.group("head")
    frac = (m.group("frac") or "")[:6]  # truncate RFC3339 nanos to micros
    tz = m.group("tz") or ""
    if tz == "Z":
        tz = "+00:00"
    s = head + (f".{frac}" if frac else "") + tz
    try:
        return _dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def extract_record(raw: dict, buckets: list[Bucket]) -> Record | None:
    # Two supported envelopes:
    #   LLMLoadgen:  raw.response.response.{usage, metadata}, raw.error (str),
    #                raw.actual_send_time (ISO 8601 nanos).
    #   Alibaba replayer: raw.response.{usage, metadata}, raw.ok (bool),
    #                     raw.fired_sec (float, seconds since run start).
    resp_outer = raw.get("response") or {}
    if isinstance(resp_outer, dict) and "response" in resp_outer and isinstance(resp_outer["response"], dict):
        # LLMLoadgen: peel one more layer.
        resp = resp_outer["response"]
        err = raw.get("error") or ""
        send_ts = _parse_iso(raw.get("actual_send_time") or raw.get("expected_send_time"))
    else:
        resp = resp_outer
        ok = raw.get("ok")
        err = "" if ok in (True, None) else (f"status={raw.get('status')}")
        send_ts = raw.get("fired_sec")
        if send_ts is None:
            send_ts = raw.get("scheduled_sec")
    usage = resp.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    if input_tokens is None:
        # Alibaba replayer duplicates prompt_len at the top level; fall back.
        input_tokens = raw.get("prompt_len")
    completion_tokens = usage.get("completion_tokens") or 0
    metrics = (resp.get("metadata") or {}).get("request_metrics") or {}

    recv = metrics.get("request_received_ts")
    prefill_done = metrics.get("prefill_finished_ts")
    decode_done = metrics.get("decode_finished_ts")

    ttft_ms = None
    if recv is not None and prefill_done is not None:
        ttft_ms = (prefill_done - recv) * 1000.0

    tbt_ms = None
    if (
        prefill_done is not None
        and decode_done is not None
        and completion_tokens > 1
    ):
        tbt_ms = (decode_done - prefill_done) * 1000.0 / (completion_tokens - 1)

    def _ms(x: float | None) -> float | None:
        return None if x is None else x * 1000.0

    queue_s = metrics.get("queue_time")
    pld_s = metrics.get("prefill_launch_delay")
    pll_s = metrics.get("prefill_launch_latency")
    # "prefill compute" ≈ TTFT - (queue + launch_delay + launch_latency).
    prefill_compute_ms = None
    if ttft_ms is not None:
        overhead = sum(x for x in (queue_s, pld_s, pll_s) if x is not None) * 1000.0
        prefill_compute_ms = max(ttft_ms - overhead, 0.0)

    rec = Record(
        input_tokens=int(input_tokens) if input_tokens is not None else -1,
        completion_tokens=int(completion_tokens),
        ttft_ms=ttft_ms,
        tbt_ms=tbt_ms,
        error=err,
        queue_ms=_ms(queue_s),
        prefill_launch_delay_ms=_ms(pld_s),
        prefill_launch_latency_ms=_ms(pll_s),
        prefill_compute_ms=prefill_compute_ms,
        send_ts=send_ts,
    )
    # Requests without usage (timeouts / 5xx) can't be bucketed by input length,
    # so they go into a synthetic "ERR" bucket and count as SLO failures.
    if input_tokens is None:
        rec.bucket = None
    else:
        rec.bucket = classify(rec.input_tokens, buckets)
    return rec


def load_records(path: Path, buckets: list[Bucket]) -> tuple[list[Record], int, int]:
    records: list[Record] = []
    malformed = 0
    total_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            rec = extract_record(raw, buckets)
            if rec is not None:
                records.append(rec)
    return records, total_lines, malformed


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    if q <= 0:
        return xs_sorted[0]
    if q >= 1:
        return xs_sorted[-1]
    idx = (len(xs_sorted) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(xs_sorted) - 1)
    frac = idx - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def summarize(records: list[Record], buckets: list[Bucket]) -> dict:
    per_bucket: dict[str, dict] = {}
    for b in buckets:
        per_bucket[b.name] = {
            "slo_ttft_ms": b.ttft_slo_ms,
            "slo_tbt_ms": b.tbt_slo_ms,
            "input_upper": b.input_upper,
            "upper_inclusive": b.upper_inclusive,
            "count": 0,
            "errors": 0,
            "missing_ttft": 0,
            "ttft_pass": 0,
            "ttft_fail": 0,
            "ttft_samples": [],
            "tbt_samples": [],
            "violations": [],  # (ttft_ms, input_tokens)
        }
    # Synthetic bucket for requests we couldn't classify (client timeouts,
    # 5xx, etc). They always count as SLO failures.
    per_bucket["ERR"] = {
        "slo_ttft_ms": None,
        "slo_tbt_ms": None,
        "input_upper": None,
        "upper_inclusive": False,
        "count": 0,
        "errors": 0,
        "missing_ttft": 0,
        "ttft_pass": 0,
        "ttft_fail": 0,
        "ttft_samples": [],
        "tbt_samples": [],
        "violations": [],
    }

    for rec in records:
        bname = rec.bucket.name if rec.bucket is not None else "ERR"
        blk = per_bucket[bname]
        blk["count"] += 1
        if rec.error:
            blk["errors"] += 1
            blk["ttft_fail"] += 1  # counts as SLO violation
            continue
        if rec.ttft_ms is None:
            blk["missing_ttft"] += 1
            blk["ttft_fail"] += 1
            continue
        blk["ttft_samples"].append(rec.ttft_ms)
        if rec.ttft_ms <= rec.bucket.ttft_slo_ms:
            blk["ttft_pass"] += 1
        else:
            blk["ttft_fail"] += 1
            blk["violations"].append((rec.ttft_ms, rec.input_tokens))
        if rec.tbt_ms is not None:
            blk["tbt_samples"].append(rec.tbt_ms)
        for key, val in (
            ("queue_samples", rec.queue_ms),
            ("launch_delay_samples", rec.prefill_launch_delay_ms),
            ("launch_latency_samples", rec.prefill_launch_latency_ms),
            ("prefill_compute_samples", rec.prefill_compute_ms),
        ):
            blk.setdefault(key, [])
            if val is not None:
                blk[key].append(val)

    total = {
        "count": sum(b["count"] for b in per_bucket.values()),
        "errors": sum(b["errors"] for b in per_bucket.values()),
        "ttft_pass": sum(b["ttft_pass"] for b in per_bucket.values()),
        "ttft_fail": sum(b["ttft_fail"] for b in per_bucket.values()),
    }

    # Wall-clock window from client send timestamps; goodput is the SLO-
    # meeting-request rate averaged over that window.
    send_ts = [r.send_ts for r in records if r.send_ts is not None]
    if len(send_ts) >= 2:
        wall = max(send_ts) - min(send_ts)
    else:
        wall = 0.0
    total["window_sec"] = wall
    total["throughput_rps"] = (total["count"] / wall) if wall > 0 else 0.0
    total["goodput_rps"] = (total["ttft_pass"] / wall) if wall > 0 else 0.0

    for name, blk in per_bucket.items():
        ts = blk["ttft_samples"]
        blk["ttft_attainment_pct"] = _pct(blk["ttft_pass"], blk["count"])
        blk["throughput_rps"] = (blk["count"] / wall) if wall > 0 else 0.0
        blk["goodput_rps"] = (blk["ttft_pass"] / wall) if wall > 0 else 0.0

        def _mean(xs: list[float]) -> float | None:
            return statistics.fmean(xs) if xs else None

        blk["breakdown"] = {
            "queue_mean_ms": _mean(blk.get("queue_samples", [])),
            "queue_p99_ms": _quantile(blk.get("queue_samples", []), 0.99),
            "launch_delay_mean_ms": _mean(blk.get("launch_delay_samples", [])),
            "launch_latency_mean_ms": _mean(blk.get("launch_latency_samples", [])),
            "prefill_compute_mean_ms": _mean(blk.get("prefill_compute_samples", [])),
            "prefill_compute_p99_ms": _quantile(blk.get("prefill_compute_samples", []), 0.99),
            "tbt_mean_ms": _mean(blk.get("tbt_samples", [])),
            "tbt_p99_ms": _quantile(blk.get("tbt_samples", []), 0.99),
        }
        if ts:
            blk["ttft_mean_ms"] = statistics.fmean(ts)
            blk["ttft_p50_ms"] = _quantile(ts, 0.5)
            blk["ttft_p90_ms"] = _quantile(ts, 0.9)
            blk["ttft_p99_ms"] = _quantile(ts, 0.99)
            blk["ttft_max_ms"] = max(ts)
        blk["violations"].sort(reverse=True)
        # Drop full sample lists from the serialized summary (can be huge).
        for key in (
            "ttft_samples", "tbt_samples",
            "queue_samples", "launch_delay_samples",
            "launch_latency_samples", "prefill_compute_samples",
        ):
            blk.pop(key, None)

    total["ttft_attainment_pct"] = _pct(total["ttft_pass"], total["count"])
    return {"overall": total, "per_bucket": per_bucket}


def print_summary(summary: dict, print_violations: int) -> None:
    buckets_out = summary["per_bucket"]
    overall = summary["overall"]

    header = (
        f"{'bucket':<4} {'input range':<18} {'SLO(ms)':>8} "
        f"{'n':>6} {'errs':>5} {'pass':>6} {'fail':>6} "
        f"{'att%':>7} {'tput':>7} {'good':>7} "
        f"{'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, b in buckets_out.items():
        rng = _range_str(b["input_upper"], b["upper_inclusive"]) if name != "ERR" else "(unclassified)"
        mean = b.get("ttft_mean_ms")
        p50 = b.get("ttft_p50_ms")
        p90 = b.get("ttft_p90_ms")
        p99 = b.get("ttft_p99_ms")
        mx = b.get("ttft_max_ms")
        fmt = lambda x: f"{x:8.1f}" if x is not None else f"{'-':>8}"
        slo_str = f"{b['slo_ttft_ms']:8.0f}" if b["slo_ttft_ms"] is not None else f"{'-':>8}"
        print(
            f"{name:<4} {rng:<18} {slo_str} "
            f"{b['count']:>6} {b['errors']:>5} {b['ttft_pass']:>6} {b['ttft_fail']:>6} "
            f"{b['ttft_attainment_pct']:>6.2f}% "
            f"{b['throughput_rps']:>7.2f} {b['goodput_rps']:>7.2f} "
            f"{fmt(mean)} {fmt(p50)} {fmt(p90)} {fmt(p99)} {fmt(mx)}"
        )
    print("-" * len(header))
    print(
        f"{'ALL':<4} {'':<18} {'':>8} "
        f"{overall['count']:>6} {overall['errors']:>5} "
        f"{overall['ttft_pass']:>6} {overall['ttft_fail']:>6} "
        f"{overall['ttft_attainment_pct']:>6.2f}% "
        f"{overall['throughput_rps']:>7.2f} {overall['goodput_rps']:>7.2f}"
    )
    print(
        f"\nwindow={overall['window_sec']:.2f}s  "
        f"throughput={overall['throughput_rps']:.2f} rps  "
        f"goodput={overall['goodput_rps']:.2f} rps  "
        f"(goodput = SLO-meeting requests / window)"
    )

    # TTFT breakdown (means, per bucket) — exposes whether tail latency comes
    # from queueing, launch overhead, or actual prefill compute.
    print("\nTTFT breakdown (mean ms) — from sglang request_metrics:")
    bd_header = (
        f"{'bucket':<4} {'queue':>8} {'launch_d':>9} {'launch_l':>9} "
        f"{'prefill':>9} {'prefill_p99':>12} {'tbt':>8} {'tbt_p99':>9}"
    )
    print(bd_header)
    print("-" * len(bd_header))
    fmt = lambda x: f"{x:8.2f}" if x is not None else f"{'-':>8}"
    fmt9 = lambda x: f"{x:9.2f}" if x is not None else f"{'-':>9}"
    fmt12 = lambda x: f"{x:12.2f}" if x is not None else f"{'-':>12}"
    for name, b in buckets_out.items():
        if b["count"] == 0:
            continue
        bd = b.get("breakdown", {})
        print(
            f"{name:<4} {fmt(bd.get('queue_mean_ms'))} "
            f"{fmt9(bd.get('launch_delay_mean_ms'))} "
            f"{fmt9(bd.get('launch_latency_mean_ms'))} "
            f"{fmt9(bd.get('prefill_compute_mean_ms'))} "
            f"{fmt12(bd.get('prefill_compute_p99_ms'))} "
            f"{fmt(bd.get('tbt_mean_ms'))} "
            f"{fmt9(bd.get('tbt_p99_ms'))}"
        )

    if print_violations > 0:
        print("\nTop SLO violations per bucket (TTFT ms, input tokens):")
        for name, b in buckets_out.items():
            vs = b.get("violations") or []
            if not vs:
                continue
            print(f"  [{name}] slo={b['slo_ttft_ms']:.0f}ms")
            for ttft, inp in vs[:print_violations]:
                print(f"    ttft={ttft:.1f}ms  input={inp}")


def _range_str(upper: int | None, inclusive: bool) -> str:
    if upper is None:
        return ">prev"
    op = "<=" if inclusive else "<"
    return f"input {op} {upper}"


def run() -> int:
    args = parse_args()
    path = Path(args.result_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"result file not found: {path}")

    buckets = DEFAULT_BUCKETS
    records, total_lines, malformed = load_records(path, buckets)
    if not records:
        raise ValueError(
            f"no usable records parsed from {path} "
            f"(lines={total_lines}, malformed={malformed})"
        )

    summary = summarize(records, buckets)
    print(f"result_path={path}")
    print(f"lines={total_lines} parsed={len(records)} malformed={malformed}\n")
    print_summary(summary, args.print_violations)

    if args.summary_out:
        out = Path(args.summary_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        # Strip violations from the written summary to keep it compact.
        dump = {
            "overall": summary["overall"],
            "per_bucket": {
                k: {kk: vv for kk, vv in v.items() if kk != "violations"}
                for k, v in summary["per_bucket"].items()
            },
        }
        out.write_text(json.dumps(dump, indent=2))
        print(f"\nsummary_out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
