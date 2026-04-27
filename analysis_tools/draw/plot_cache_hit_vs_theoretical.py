#!/usr/bin/env python3
"""Plot per-request actual cache hit tokens vs the theoretical maximum
(= LCP with the immediate parent turn's prompt) for an Alibaba qwen-bailian
trace replay.

For each child request (parent_chat_id != -1):
  theoretical = LCP(child.token_ids, parent.token_ids)
  actual      = response.usage.prompt_tokens_details.cached_tokens

Requests get categorized:
  perfect  — actual >= theoretical (parent fully cached)
  partial  — 0 < actual < theoretical (parent partially evicted)
  fullmiss — actual == 0 but theoretical > 0 (parent fully evicted)

By default only child requests are plotted. Pass --include-root to also show
root requests (parent_chat_id == -1, theoretical is always 0 — they can still
have actual > 0 from cross-session prefix matches in radix cache).

Output: <experiment_dir>/cache_hit_child_only.png  (or cache_hit_all.png with
--include-root).

Usage:
    python3 draw/plot_cache_hit_vs_theoretical.py \
        --experiment-dir experiment/2026-04-17_16-41-33_alibaba

The trace path is auto-detected from <experiment_dir>/run_config.json's
"trace" field; override with --trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", required=True, help="Experiment dir containing responses.json and run_config.json")
    p.add_argument("--trace", default="", help="Override trace path (default: read from run_config.json)")
    p.add_argument("--responses-name", default="responses.json", help="Result file under experiment_dir")
    p.add_argument("--include-root", action="store_true", help="Also plot root requests (parent=-1)")
    p.add_argument("--output-name", default="", help="Override output filename (default: cache_hit_child_only.png / cache_hit_all.png)")
    return p.parse_args()


def lcp_length(a: np.ndarray, b: np.ndarray) -> int:
    n = min(a.size, b.size)
    if n == 0:
        return 0
    eq = a[:n] == b[:n]
    diff = np.flatnonzero(~eq)
    return int(diff[0]) if diff.size else n


def load_trace_tokens(trace_path: Path) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    tokens_by_cid: dict[int, np.ndarray] = {}
    parent_by_cid: dict[int, int] = {}
    with trace_path.open() as f:
        for line in f:
            rec = json.loads(line)
            cid = int(rec["chat_id"])
            parent_by_cid[cid] = int(rec["parent_chat_id"])
            tokens_by_cid[cid] = np.asarray(rec["token_ids"], dtype=np.int32)
    return tokens_by_cid, parent_by_cid


def classify_row(actual: int, theoretical: int, is_root: bool) -> str:
    if is_root:
        return "root_beat" if actual > 0 else "root_miss"
    if actual >= theoretical:
        return "perfect"
    if actual == 0:
        return "fullmiss"
    return "partial"


def collect_rows(resp_path: Path, tokens_by_cid: dict[int, np.ndarray],
                 parent_by_cid: dict[int, int], include_root: bool) -> list[dict]:
    rows = []
    with resp_path.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            resp = r.get("response") or {}
            if not isinstance(resp, dict) or not resp:
                continue
            cid = r["chat_id"]
            my_tok = tokens_by_cid.get(cid)
            if my_tok is None:
                continue
            pid = parent_by_cid.get(cid, -1)
            is_root = pid == -1
            if is_root and not include_root:
                continue
            parent_tok = tokens_by_cid.get(pid) if not is_root else None
            theo = lcp_length(my_tok, parent_tok) if parent_tok is not None else 0
            usage = resp.get("usage") or {}
            actual = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            prompt = usage.get("prompt_tokens", 0) or 0
            # New loadgen writes "requestIndex"; older LLMLoadgen / pre-rename
            # alibaba replayer wrote "recordIndex". Accept either.
            idx = r.get("requestIndex", r.get("recordIndex"))
            rows.append({
                "idx": idx,
                "prompt": prompt,
                "theo": theo,
                "act": actual,
                "lost": max(0, theo - actual),
                "is_root": is_root,
                "cat": classify_row(actual, theo, is_root),
            })
    rows.sort(key=lambda r: r["idx"])
    return rows


# Color per category (child-only and root both covered).
CAT_COLORS = {
    "perfect":   "#1f77b4",
    "partial":   "#8e44ad",
    "fullmiss":  "#d62728",
    "root_beat": "#2ca02c",
    "root_miss": "#aaaaaa",
}
CAT_LABELS = {
    "perfect":   "Parent fully cached (actual == theoretical)",
    "partial":   "Parent partially evicted (0 < actual < theoretical)",
    "fullmiss":  "Parent fully evicted (actual == 0 but theoretical > 0)",
    "root_beat": "Root cross-session hit (actual > 0, no parent)",
    "root_miss": "Root no cache (actual == 0, no parent)",
}


def render(rows: list[dict], out_path: Path, title_prefix: str, include_root: bool) -> None:
    x = np.array([r["idx"] for r in rows])
    theo = np.array([r["theo"] for r in rows])
    act = np.array([r["act"] for r in rows])
    prompt = np.array([r["prompt"] for r in rows])
    lost = np.array([r["lost"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(26, 11), dpi=200, sharex=True)

    counts = {c: sum(1 for r in rows if r["cat"] == c) for c in CAT_COLORS}
    for cat, color in CAT_COLORS.items():
        if counts[cat] == 0:
            continue
        mask = np.array([r["cat"] == cat for r in rows])
        ax1.bar(x[mask], act[mask], width=0.9, color=color, alpha=0.85,
                label=f"{CAT_LABELS[cat]} [n={counts[cat]}]")
    ax1.plot(x, theo, color="#ff7f0e", linewidth=0.4, alpha=0.5,
             label="Theoretical max (LCP vs parent)")
    ax1.set_yscale("symlog", linthresh=100)
    ax1.set_ylabel("Cache hit tokens (symlog)")
    tp, ta, tt = int(prompt.sum()), int(act.sum()), int(theo.sum())
    ax1.set_title(
        f"{title_prefix}  |  n={len(rows)}  theo_rate={tt/tp:.1%}  "
        f"actual_rate={ta/tp:.1%}  realized/theo={ta/max(1,tt):.1%}"
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.bar(x, lost, width=0.9, color="#d62728", alpha=0.7,
            label=f"Tokens lost to eviction (sum={int(lost.sum()):,})")
    ax2.set_yscale("symlog", linthresh=100)
    ax2.set_ylabel("Lost tokens (symlog)\n(theoretical - actual)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"  n={len(rows)}  theo_rate={tt/tp:.1%}  actual_rate={ta/tp:.1%}  realized={ta/max(1,tt):.1%}")
    print(f"  lost to eviction: {int(lost.sum()):,} tokens")
    for cat in ["perfect", "partial", "fullmiss", "root_beat", "root_miss"]:
        if counts[cat]:
            print(f"  {cat:>10s}: {counts[cat]}")


def resolve_trace_path(experiment_dir: Path, override: str) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    cfg_path = experiment_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        # Two layouts:
        #   legacy shell runner:   {"trace": "alibaba/.../foo.jsonl", ...}
        #   Go experiment runner:  {"loadgen": {"alibaba": {"trace": "..."}}, ...}
        t = cfg.get("trace")
        if not t:
            t = (cfg.get("loadgen", {}) or {}).get("alibaba", {}).get("trace") or ""
        if t:
            p = Path(t).expanduser()
            if not p.is_absolute():
                # run_config.json may store a repo-relative path; resolve it
                # against the repo root (three levels up from experiment_dir
                # when it lives at <repo>/experiment/<ts>_<type>/).
                p = (experiment_dir.parent.parent / p).resolve()
            return p.resolve()
    raise FileNotFoundError(
        "cannot determine trace path: --trace not set and run_config.json missing or has no 'trace' field"
    )


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(experiment_dir)
    resp_path = experiment_dir / args.responses_name
    if not resp_path.exists():
        raise FileNotFoundError(resp_path)

    trace_path = resolve_trace_path(experiment_dir, args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    print(f"loading trace: {trace_path}")
    tokens_by_cid, parent_by_cid = load_trace_tokens(trace_path)
    print(f"  {len(tokens_by_cid)} chat_ids")

    rows = collect_rows(resp_path, tokens_by_cid, parent_by_cid, args.include_root)
    if not rows:
        print("no requests to plot — aborting")
        return 1

    out_name = args.output_name or ("cache_hit_all.png" if args.include_root else "cache_hit_child_only.png")
    render(rows, experiment_dir / out_name, experiment_dir.name, args.include_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
