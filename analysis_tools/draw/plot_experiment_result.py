#!/usr/bin/env python3
"""Plot TTFT / TBT / E2E / token metrics from a loadgen experiment directory.

Auto-detects input layout:

  - **Multi-agent mode** — ``<experiment_dir>/<responses-subdir>/<agent>/`` with one
    JSON/JSONL per agent (the shape produced by ``run_agents.sh``). One plot
    is written inside each agent's directory.
  - **Single-file mode** — ``<experiment_dir>/<result-name>`` is a JSON array
    or JSONL produced by ``loadgen alibaba|azure|agent``. One plot is written
    to ``<experiment_dir>/<output-name>``. For alibaba traces, use
    ``--type-filter`` to restrict by record type or ``--per-type`` to render
    one plot per type.

The script used to live in three separate files
(``plot_experiment_{agents_ttft_tbt,alibaba_result,azure_result}.py``); this
merged version covers all of them.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        help="Experiment directory path, e.g. experiment/2026-03-27_06-33-04",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "multi-agent", "single-file"],
        default="auto",
        help=(
            "Which layout to use. 'auto' (default) picks multi-agent when "
            "<experiment_dir>/<responses-subdir>/ exists with at least one "
            "subdirectory, otherwise single-file."
        ),
    )
    parser.add_argument(
        "--responses-subdir",
        default="responses",
        help="Responses subdirectory under experiment_dir for multi-agent mode (default: responses).",
    )
    parser.add_argument(
        "--result-name",
        default="",
        help=(
            "Result file name for single-file mode (relative to experiment_dir). "
            "If empty, auto-detect among {responses.json, result.json, result.jsonl}."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="ttft_tbt_request_cache_tokens.png",
        help=(
            "Output plot filename. In multi-agent mode it is saved into each "
            "agent folder; in single-file mode it is saved into experiment_dir."
        ),
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include records where ok != true (default: only ok=true).",
    )
    parser.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use symlog y-scale for token axes (default: linear). Pass --log-y to enable.",
    )
    parser.add_argument(
        "--type-filter",
        default="",
        help=(
            "Single-file mode only: restrict to records whose 'type' field "
            "matches (e.g. 'coder'). Empty = no filter."
        ),
    )
    parser.add_argument(
        "--per-type",
        action="store_true",
        help=(
            "Single-file mode only: additionally render one plot per distinct "
            "'type' value, saved as <output-stem>_<type>.png."
        ),
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _get_in(mapping: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Parse a result file as either JSON (array/object) or JSONL.

    Detection is content-based, not extension-based, because the alibaba /
    agent loadgen writes JSONL to files named `responses.json` / `result.json`.
    We look at the first non-whitespace character: `[` ⇒ JSON array, `{` with
    multiple records ⇒ JSONL.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped[0] == "[":
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []

    # JSONL (one JSON object per line) — works for both `.jsonl` and `.json`
    # files whose content is line-delimited.
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _extract_ttft_seconds(record: dict[str, Any]) -> float | None:
    for keys in [
        ["ttft_sec"],
        ["ttft"],
        ["response", "metadata", "request_metrics", "time_to_first_token"],
        ["response", "metadata", "request_metrics", "ttft_sec"],
        ["response", "metadata", "request_metrics", "ttft"],
        ["response", "meta_info", "time_to_first_token"],
        ["response", "meta_info", "ttft"],
    ]:
        val = _to_float(_get_in(record, keys))
        if val is not None and val >= 0:
            return val

    prefill_finished_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "prefill_finished_ts"])
    )
    request_received_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "request_received_ts"])
    )
    if prefill_finished_ts is not None and request_received_ts is not None:
        proxy = prefill_finished_ts - request_received_ts
        if proxy >= 0:
            return proxy
    return None


def _extract_tbt_seconds(record: dict[str, Any]) -> float | None:
    for keys in [
        ["tbt_sec"],
        ["tbt"],
        ["response", "metadata", "request_metrics", "inter_token_latency"],
        ["response", "metadata", "request_metrics", "tbt_sec"],
        ["response", "metadata", "request_metrics", "tbt"],
        ["response", "meta_info", "inter_token_latency"],
        ["response", "meta_info", "tbt"],
    ]:
        val = _to_float(_get_in(record, keys))
        if val is not None and val >= 0:
            return val

    decode_throughput = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "decode_throughput"])
    )
    if decode_throughput is not None and decode_throughput > 0:
        return 1.0 / decode_throughput

    completion_tokens = (
        _to_int(_get_in(record, ["response", "usage", "completion_tokens"]))
        or _to_int(_get_in(record, ["response", "usage", "output_tokens"]))
        or 0
    )
    if completion_tokens <= 0:
        return None

    prefill_finished_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "prefill_finished_ts"])
    )
    decode_finished_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "decode_finished_ts"])
    )
    response_sent_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "response_sent_to_client_ts"])
    )
    decode_end = decode_finished_ts if decode_finished_ts is not None else response_sent_ts
    if prefill_finished_ts is None or decode_end is None:
        return None

    decode_duration = decode_end - prefill_finished_ts
    if decode_duration < 0:
        return None
    return decode_duration / completion_tokens


def _extract_queue_latency_seconds(record: dict[str, Any]) -> float | None:
    for keys in [
        ["queue_time"],
        ["queue_latency"],
        ["response", "metadata", "request_metrics", "queue_time"],
        ["response", "metadata", "request_metrics", "queue_latency"],
        ["response", "meta_info", "queue_time"],
        ["response", "meta_info", "queue_latency"],
    ]:
        val = _to_float(_get_in(record, keys))
        if val is not None and val >= 0:
            return val
    return None


def _extract_e2e_latency_seconds(record: dict[str, Any]) -> float | None:
    for keys in [
        ["e2e_latency"],
        ["latency_sec"],
        ["response", "metadata", "request_metrics", "e2e_latency"],
        ["response", "metadata", "request_metrics", "latency_sec"],
        ["response", "meta_info", "e2e_latency"],
        ["response", "meta_info", "latency_sec"],
    ]:
        val = _to_float(_get_in(record, keys))
        if val is not None and val >= 0:
            return val

    request_received_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "request_received_ts"])
    )
    response_sent_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "response_sent_to_client_ts"])
    )
    decode_finished_ts = _to_float(
        _get_in(record, ["response", "metadata", "request_metrics", "decode_finished_ts"])
    )
    e2e_end = response_sent_ts if response_sent_ts is not None else decode_finished_ts
    if request_received_ts is None or e2e_end is None:
        return None
    e2e = e2e_end - request_received_ts
    return e2e if e2e >= 0 else None


def _extract_token_bars(record: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = _to_int(_get_in(record, ["response", "usage", "prompt_tokens"])) or 0
    cached_tokens = _to_int(
        _get_in(record, ["response", "usage", "prompt_tokens_details", "cached_tokens"])
    ) or 0
    cached_tokens = max(0, min(cached_tokens, prompt_tokens))
    no_cache_tokens = max(prompt_tokens - cached_tokens, 0)
    return cached_tokens, no_cache_tokens


def _extract_output_tokens(record: dict[str, Any]) -> int | None:
    for keys in [
        ["output_tokens"],
        ["completion_tokens"],
        ["response", "usage", "output_tokens"],
        ["response", "usage", "completion_tokens"],
    ]:
        val = _to_int(_get_in(record, keys))
        if val is not None and val >= 0:
            return val
    return None


def _pick_response_file(agent_dir: Path) -> Path | None:
    candidates = [
        "load_responses.json",
        "load_responses_single.json",
        "load_responses.jsonl",
        "load_responses_single.jsonl",
    ]
    for name in candidates:
        p = agent_dir / name
        if p.exists():
            return p

    files = sorted(
        p for p in agent_dir.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )
    return files[0] if files else None


def _collect_points(records: list[dict[str, Any]], include_failed: bool) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for rec in records:
        if not include_failed and not bool(rec.get("ok", False)):
            continue

        ttft = _extract_ttft_seconds(rec)
        tbt = _extract_tbt_seconds(rec)
        queue_latency = _extract_queue_latency_seconds(rec)
        e2e_latency = _extract_e2e_latency_seconds(rec)
        output_tokens = _extract_output_tokens(rec)
        cached_tokens, no_cache_tokens = _extract_token_bars(rec)
        req_idx = _to_int(rec.get("requestIndex"))

        if (
            ttft is None
            and tbt is None
            and queue_latency is None
            and e2e_latency is None
            and output_tokens is None
            and cached_tokens == 0
            and no_cache_tokens == 0
        ):
            continue

        points.append(
            {
                "request_idx": req_idx,
                "ttft_sec": ttft,
                "tbt_sec": tbt,
                "queue_latency_sec": queue_latency,
                "e2e_latency_sec": e2e_latency,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "no_cache_tokens": no_cache_tokens,
            }
        )
    return points


def _plot_one_agent(
    points: list[dict[str, Any]],
    out_path: Path,
    title_prefix: str,
    log_y: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if all(p["request_idx"] is not None for p in points):
        points.sort(key=lambda x: int(x["request_idx"]))
        x_vals = [int(p["request_idx"]) for p in points]
    else:
        x_vals = list(range(1, len(points) + 1))

    ttft_vals = [float(p["ttft_sec"]) if p["ttft_sec"] is not None else math.nan for p in points]
    tbt_vals = [float(p["tbt_sec"]) if p["tbt_sec"] is not None else math.nan for p in points]
    queue_vals = [
        float(p["queue_latency_sec"]) if p["queue_latency_sec"] is not None else math.nan
        for p in points
    ]
    e2e_vals = [
        float(p["e2e_latency_sec"]) if p["e2e_latency_sec"] is not None else math.nan
        for p in points
    ]
    output_tokens_vals = [
        int(p["output_tokens"]) if p["output_tokens"] is not None else math.nan for p in points
    ]
    cached_vals = [int(p["cached_tokens"]) for p in points]
    no_cache_vals = [int(p["no_cache_tokens"]) for p in points]

    fig = plt.figure(figsize=(26, 18), dpi=200)
    gs = fig.add_gridspec(
        6,
        4,
        height_ratios=[2.1, 1.1, 1.1, 1.0, 1.0, 1.2],
        hspace=0.35,
        wspace=0.35,
    )
    ax_top = fig.add_subplot(gs[0, :])
    ax_mid = fig.add_subplot(gs[1, :], sharex=ax_top)
    ax_bottom = fig.add_subplot(gs[2, :], sharex=ax_top)
    ax_tokens = fig.add_subplot(gs[3, :], sharex=ax_top)
    ax_compute = fig.add_subplot(gs[4, :], sharex=ax_top)
    ax_cdf_ttft = fig.add_subplot(gs[5, 0])
    ax_cdf_out = fig.add_subplot(gs[5, 1])
    ax_cdf_comp = fig.add_subplot(gs[5, 2])
    ax_cdf_queue = fig.add_subplot(gs[5, 3])
    ax_top_right = ax_top.twinx()

    line_ttft = ax_top.plot(
        x_vals,
        ttft_vals,
        color="#ff1f5b",
        linewidth=2.2,
        marker="o",
        markersize=2.4,
        label="TTFT (sec)",
    )[0]

    bars_no_cache = ax_top_right.bar(
        x_vals,
        no_cache_vals,
        width=0.8,
        color="#2f7fff",
        alpha=0.50,
        label="No-cache prompt tokens",
    )
    bars_cache = ax_top_right.bar(
        x_vals,
        cached_vals,
        width=0.8,
        bottom=no_cache_vals,
        color="#59A14F",
        alpha=0.50,
        label="Cached prompt tokens",
    )

    ax_top.set_title(f"{title_prefix}: TTFT / Prompt Tokens / TBT / E2E by Request")
    ax_top.set_ylabel("TTFT (seconds)", color="#d62728")
    scale_suffix = "symlog" if log_y else "linear"
    ax_top_right.set_ylabel(f"Prompt Tokens (stacked, {scale_suffix})")
    if log_y:
        ax_top_right.set_yscale("symlog", linthresh=100)
    ax_top.grid(True, axis="y", alpha=0.25)

    line_tbt = ax_mid.plot(
        x_vals,
        tbt_vals,
        color="#ff8c00",
        linewidth=2.0,
        marker="o",
        markersize=2.3,
        label="TBT (sec/token)",
    )[0]
    ax_mid.set_xlabel("Request Index")
    ax_mid.set_ylabel("TBT (sec/token)", color="#ff8c00")
    ax_mid.grid(True, axis="y", alpha=0.25)
    ax_mid.set_ylim(bottom=0)

    line_queue = ax_bottom.plot(
        x_vals,
        queue_vals,
        color="#7a00ff",
        linewidth=2.0,
        marker="o",
        markersize=2.3,
        label="Queue latency (sec)",
    )[0]
    line_e2e = ax_bottom.plot(
        x_vals,
        e2e_vals,
        color="#111111",
        linewidth=2.0,
        marker="o",
        markersize=2.1,
        label="E2E latency (sec)",
    )[0]
    ax_bottom.set_xlabel("Request Index")
    ax_bottom.set_ylabel("Latency (sec)")
    ax_bottom.grid(True, axis="y", alpha=0.25)
    ax_bottom.set_ylim(bottom=0)

    line_output_tokens = ax_tokens.plot(
        x_vals,
        output_tokens_vals,
        color="#0b7285",
        linewidth=2.0,
        marker="o",
        markersize=2.2,
        label="Output tokens",
    )[0]
    ax_tokens.set_xlabel("Request Index")
    ax_tokens.set_ylabel(f"Output tokens ({scale_suffix})")
    ax_tokens.grid(True, axis="y", alpha=0.25)
    if log_y:
        ax_tokens.set_yscale("symlog", linthresh=100)

    line_compute_tokens = ax_compute.plot(
        x_vals,
        no_cache_vals,
        color="#2f7fff",
        linewidth=2.0,
        marker="o",
        markersize=2.2,
        label="Compute tokens (non-cached prompt)",
    )[0]
    ax_compute.set_xlabel("Request Index")
    ax_compute.set_ylabel(f"Compute tokens ({scale_suffix})")
    ax_compute.grid(True, axis="y", alpha=0.25)
    if log_y:
        ax_compute.set_yscale("symlog", linthresh=100)

    ax_top.set_xlabel("Request Index")
    ax_top.tick_params(axis="x", labelbottom=True)
    ax_mid.tick_params(axis="x", labelbottom=True)
    ax_bottom.tick_params(axis="x", labelbottom=True)
    ax_tokens.tick_params(axis="x", labelbottom=True)
    ax_compute.tick_params(axis="x", labelbottom=True)

    if len(x_vals) > 30:
        step = max(1, len(x_vals) // 20)
        ax_mid.set_xticks(x_vals[::step])
        ax_bottom.set_xticks(x_vals[::step])
        ax_tokens.set_xticks(x_vals[::step])
        ax_compute.set_xticks(x_vals[::step])
        ax_top.set_xticks(x_vals[::step])

    handles = [line_ttft, bars_no_cache, bars_cache]
    labels = [h.get_label() for h in handles]
    ax_top.legend(handles, labels, loc="upper right")
    ax_mid.legend([line_tbt], [line_tbt.get_label()], loc="upper right")
    ax_bottom.legend(
        [line_queue, line_e2e],
        [line_queue.get_label(), line_e2e.get_label()],
        loc="upper right",
    )
    ax_tokens.legend([line_output_tokens], [line_output_tokens.get_label()], loc="upper right")
    ax_compute.legend([line_compute_tokens], [line_compute_tokens.get_label()], loc="upper right")

    def _draw_cdf(ax, values, color, xlabel, title):
        clean = sorted(v for v in values if v is not None and not math.isnan(float(v)))
        if not clean:
            ax.set_title(f"{title} (no data)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("CDF")
            return
        n = len(clean)
        ys = [(i + 1) / n for i in range(n)]
        ax.plot(clean, ys, color=color, linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=min(0, clean[0]))
        ax.grid(True, alpha=0.25)
        p50 = clean[max(0, int(0.50 * n) - 1)]
        p95 = clean[max(0, int(0.95 * n) - 1)]
        ax.set_title(f"{title}  p50={p50:g} p95={p95:g}")

    _draw_cdf(ax_cdf_ttft, ttft_vals, "#ff1f5b", "TTFT (sec)", "TTFT CDF")
    _draw_cdf(ax_cdf_out, output_tokens_vals, "#0b7285", "Output tokens", "Output tokens CDF")
    _draw_cdf(ax_cdf_comp, no_cache_vals, "#2f7fff", "Compute tokens", "Compute tokens CDF")
    _draw_cdf(ax_cdf_queue, queue_vals, "#7a00ff", "Queue latency (sec)", "Queue latency CDF")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


_SINGLE_FILE_CANDIDATES = ("responses.json", "result.json", "result.jsonl", "loadgen.jsonl")


def _resolve_mode(args: argparse.Namespace, experiment_dir: Path) -> str:
    if args.mode != "auto":
        return args.mode
    responses_root = experiment_dir / args.responses_subdir
    if responses_root.is_dir() and any(p.is_dir() for p in responses_root.iterdir()):
        return "multi-agent"
    return "single-file"


def _resolve_single_file(experiment_dir: Path, result_name: str) -> Path:
    if result_name:
        candidate = experiment_dir / result_name
        if not candidate.exists():
            raise FileNotFoundError(f"--result-name points to missing file: {candidate}")
        return candidate
    for name in _SINGLE_FILE_CANDIDATES:
        candidate = experiment_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no single-file result found under {experiment_dir}; "
        f"tried {list(_SINGLE_FILE_CANDIDATES)}. Pass --result-name to override."
    )


def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Patch legacy schemas so they look like what _collect_points expects.

    Handles two legacy shapes the new loadgen no longer produces but older
    experiment dirs still contain:

      - LLMLoadgen Azure output, where the SGLang body is nested at
        ``record["response"]["response"]`` and there's no explicit ``ok`` — we
        lift the inner body up and synthesize ``ok`` from ``error``.
      - Alibaba replayer output keyed on ``recordIndex`` rather than
        ``requestIndex`` — we copy it across so extractors find it.
    """
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        fixed = dict(rec)

        # Legacy LLMLoadgen: lift nested response.response -> response.
        resp = fixed.get("response")
        if isinstance(resp, dict) and "response" in resp and isinstance(resp["response"], dict):
            inner_metrics = resp.get("response", {})
            if "metadata" in inner_metrics or "usage" in inner_metrics or "choices" in inner_metrics:
                fixed["response"] = inner_metrics
                if "ok" not in fixed:
                    fixed["ok"] = not fixed.get("error")

        # Legacy alibaba: recordIndex -> requestIndex.
        if "requestIndex" not in fixed:
            rec_idx = fixed.get("recordIndex")
            fixed["requestIndex"] = rec_idx if isinstance(rec_idx, int) else idx

        out.append(fixed)
    return out


def _render_single(
    records: list[dict[str, Any]],
    out_path: Path,
    title_prefix: str,
    include_failed: bool,
    log_y: bool,
) -> int:
    points = _collect_points(records, include_failed=include_failed)
    if not points:
        print(f"[skip] no plottable points -> {out_path}")
        return 0
    _plot_one_agent(points, out_path, title_prefix=title_prefix, log_y=log_y)
    valid_ttft = sum(1 for p in points if p["ttft_sec"] is not None)
    valid_tbt = sum(1 for p in points if p["tbt_sec"] is not None)
    valid_e2e = sum(1 for p in points if p["e2e_latency_sec"] is not None)
    valid_out = sum(1 for p in points if p["output_tokens"] is not None)
    print(
        f"[ok] {title_prefix}: records={len(records)} points={len(points)} "
        f"ttft={valid_ttft} tbt={valid_tbt} e2e={valid_e2e} output_tokens={valid_out} "
        f"output={out_path}"
    )
    return 1


def _run_multi_agent(args: argparse.Namespace, experiment_dir: Path) -> int:
    responses_root = (experiment_dir / args.responses_subdir).resolve()
    if not responses_root.exists():
        raise FileNotFoundError(f"responses directory not found: {responses_root}")

    agent_dirs = sorted([p for p in responses_root.iterdir() if p.is_dir()])
    if not agent_dirs:
        raise ValueError(f"No agent response directories found under: {responses_root}")

    plotted = 0
    skipped = 0
    for agent_dir in agent_dirs:
        response_file = _pick_response_file(agent_dir)
        if response_file is None:
            print(f"[skip] {agent_dir.name}: no response json/jsonl file")
            skipped += 1
            continue

        records = _normalize_records(_read_records(response_file))
        rendered = _render_single(
            records,
            agent_dir / args.output_name,
            agent_dir.name,
            args.include_failed,
            args.log_y,
        )
        if rendered:
            plotted += 1
        else:
            skipped += 1

    print(f"experiment_dir={experiment_dir}")
    print(f"responses_root={responses_root}")
    print(f"plotted_agents={plotted}")
    print(f"skipped_agents={skipped}")
    return 0 if plotted > 0 else 1


def _run_single_file(args: argparse.Namespace, experiment_dir: Path) -> int:
    result_path = _resolve_single_file(experiment_dir, args.result_name)
    records = _normalize_records(_read_records(result_path))

    if args.type_filter:
        records = [r for r in records if r.get("type") == args.type_filter]
        if not records:
            raise ValueError(f"no records matched --type-filter={args.type_filter!r}")

    rendered = 0
    out_path = experiment_dir / args.output_name
    rendered += _render_single(
        records, out_path, experiment_dir.name, args.include_failed, args.log_y
    )

    if args.per_type:
        stem = Path(args.output_name).stem
        suffix = Path(args.output_name).suffix or ".png"
        by_type: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            t = rec.get("type")
            if not isinstance(t, str) or not t:
                continue
            by_type.setdefault(t, []).append(rec)
        for type_name, recs in sorted(by_type.items()):
            per_out = experiment_dir / f"{stem}_{type_name}{suffix}"
            rendered += _render_single(
                recs, per_out, f"{experiment_dir.name}::{type_name}", args.include_failed, args.log_y
            )

    return 0 if rendered > 0 else 1


def run() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment_dir not found: {experiment_dir}")

    mode = _resolve_mode(args, experiment_dir)
    if mode == "multi-agent":
        return _run_multi_agent(args, experiment_dir)
    return _run_single_file(args, experiment_dir)


if __name__ == "__main__":
    raise SystemExit(run())
