#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic trace messages with attempt growth and compaction simulation."
    )
    parser.add_argument(
        "--output",
        default="agent_trace/agent_task1/cache/messages.synthetic.json",
        help="Output path for synthetic messages JSON.",
    )
    parser.add_argument(
        "--reference-messages",
        default="agent_trace/agent_task1/cache/messages.json",
        help="Path to original messages.json used to estimate output-token distribution.",
    )
    parser.add_argument(
        "--reference-load-responses",
        default="agent_trace/agent_task1/cache/load_responses.json",
        help="Fallback path to load_responses.json when messages.json is unavailable.",
    )
    parser.add_argument(
        "--reference-replay-plan",
        default="agent_trace/agent_task1/cache/replay_plan.json",
        help="Path to replay_plan.json used to estimate output-token distribution from payload.max_tokens.",
    )
    parser.add_argument(
        "--output-token-source",
        choices=["mix_60_40", "replay_plan", "messages", "load_responses", "auto"],
        default="mix_60_40",
        help="Where to estimate output-token PMF from.",
    )
    parser.add_argument("--attempts", type=int, default=12, help="Number of synthetic attempts.")
    parser.add_argument(
        "--tool-steps-min-per-attempt",
        type=int,
        default=12,
        help="Min assistant-tool cycles for each attempt (random per attempt).",
    )
    parser.add_argument(
        "--tool-steps-max-per-attempt",
        type=int,
        default=26,
        help="Max assistant-tool cycles for each attempt (random per attempt).",
    )
    parser.add_argument(
        "--compaction-threshold",
        type=int,
        default=25000,
        help="Trigger compaction when estimated context tokens exceed this threshold.",
    )
    parser.add_argument(
        "--drop-tokens",
        type=int,
        default=15000,
        help="Drop approximately this many tokens from the head during compaction.",
    )
    parser.add_argument(
        "--summary-tokens",
        type=int,
        default=1000,
        help="Estimated token budget for synthetic compaction summary user message.",
    )
    parser.add_argument(
        "--attempt-query-min-tokens",
        type=int,
        default=200,
        help="Min estimated tokens for each new attempt user query.",
    )
    parser.add_argument(
        "--attempt-query-max-tokens",
        type=int,
        default=500,
        help="Max estimated tokens for each new attempt user query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260320,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3.5-35B-A3B",
        help="Hugging Face tokenizer model id used for random token-id decode text generation.",
    )
    return parser.parse_args()


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(TOKEN_RE.findall(text)))


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def flatten_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        return _safe_json(content)
    if not isinstance(content, list):
        return _safe_json(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(_safe_json(block))
            continue

        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "toolCall":
            parts.append(_safe_json(block))
        else:
            parts.append(_safe_json(block))
    return "\n".join(p for p in parts if p)


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    base = 4
    role_tokens = estimate_text_tokens(str(msg.get("role", "")))
    content_tokens = estimate_text_tokens(flatten_content_to_text(msg.get("content")))
    return base + role_tokens + content_tokens


def estimate_history_tokens(history: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in history)


class RandomTokenDecodeTextGenerator:
    def __init__(self, model_id: str) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError("transformers is required. Install via: pip install transformers") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        vocab = self.tokenizer.get_vocab()
        all_ids = sorted(set(int(v) for v in vocab.values()))
        special_ids = set(int(v) for v in (self.tokenizer.all_special_ids or []))
        candidate_ids = [tid for tid in all_ids if tid not in special_ids]
        if not candidate_ids:
            candidate_ids = all_ids
        if not candidate_ids:
            raise RuntimeError(f"Tokenizer has no available token ids: {model_id}")
        self.candidate_ids = candidate_ids

    def decode_random_token_ids(self, target_tokens: int, rng: random.Random) -> str:
        n = max(0, int(target_tokens))
        if n == 0:
            return ""
        ids = [self.candidate_ids[rng.randrange(len(self.candidate_ids))] for _ in range(n)]
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def make_target_token_text(
    header: str,
    target_tokens: int,
    rng: random.Random,
    text_gen: RandomTokenDecodeTextGenerator,
) -> str:
    body = text_gen.decode_random_token_ids(target_tokens=target_tokens, rng=rng)
    if header:
        return f"{header}\n\n{body}"
    return body


def load_output_token_samples_from_load_responses(path: Path) -> list[int]:
    if not path.exists():
        return [64, 80, 96, 128, 192, 256, 384, 512, 768, 1024]
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [64, 80, 96, 128, 192, 256, 384, 512, 768, 1024]

    values: list[int] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            response = row.get("response")
            if not isinstance(response, dict):
                continue
            usage = response.get("usage")
            if not isinstance(usage, dict):
                continue
            v = usage.get("completion_tokens")
            if isinstance(v, int) and v > 0:
                values.append(v)
    if not values:
        return [64, 80, 96, 128, 192, 256, 384, 512, 768, 1024]
    return values


def load_output_token_samples_from_messages(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    snapshots: list[Any]
    if isinstance(obj, list):
        snapshots = obj
    else:
        return []
    if not snapshots:
        return []

    # messages.json is list of snapshots; use the final snapshot to avoid repeated counting.
    final_snapshot = snapshots[-1]
    if not isinstance(final_snapshot, list):
        return []

    values: list[int] = []
    for msg in final_snapshot:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            continue
        out = usage.get("output")
        if isinstance(out, int) and out >= 0:
            values.append(out)
    return values


def load_output_token_samples_from_replay_plan(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(rows, list):
        return []

    values: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        v = payload.get("max_tokens")
        if isinstance(v, int) and v > 0:
            values.append(v)
    return values


def build_output_token_pmf(samples: list[int]) -> list[tuple[int, float]]:
    if not samples:
        samples = [64, 80, 96, 128, 192, 256, 384, 512, 768, 1024]
    counts = Counter(int(v) for v in samples)
    total = float(sum(counts.values()))
    items = sorted(counts.items(), key=lambda x: x[0])
    return [(value, count / total) for value, count in items]


def sample_output_tokens_from_pmf(pmf: list[tuple[int, float]], rng: random.Random) -> int:
    if not pmf:
        return 128
    r = rng.random()
    acc = 0.0
    for value, prob in pmf:
        acc += prob
        if r <= acc:
            return value
    return pmf[-1][0]


def sample_output_tokens_mix_60_40(rng: random.Random) -> int:
    # 60% requests: [50, 150], 40% requests: [200, 3000]
    if rng.random() < 0.6:
        return rng.randint(50, 150)
    elif rng.random() < 0.9:
        return rng.randint(200, 1000)
    return rng.randint(1200, 3000)


class TsClock:
    def __init__(self, start_ms: int, rng: random.Random) -> None:
        self._ts = start_ms
        self._rng = rng

    def next(self, lo: int = 180, hi: int = 1800) -> int:
        self._ts += self._rng.randint(lo, hi)
        return self._ts


def make_user_message(text: str, ts: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "timestamp": ts,
    }


def make_assistant_tool_message(
    *,
    text: str,
    output_tokens: int,
    input_tokens: int,
    cache_read_tokens: int,
    total_tokens: int,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    ts: int,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "toolCall",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": tool_args,
            },
        ],
        "usage": {
            "input": input_tokens,
            "output": output_tokens,
            "cacheRead": cache_read_tokens,
            "cacheWrite": 0,
            "totalTokens": total_tokens,
        },
        "stopReason": "toolUse",
        "timestamp": ts,
    }


def make_tool_result_message(
    *,
    tool_call_id: str,
    tool_name: str,
    text: str,
    ts: int,
) -> dict[str, Any]:
    return {
        "role": "toolResult",
        "toolCallId": tool_call_id,
        "toolName": tool_name,
        "content": [{"type": "text", "text": text}],
        "isError": False,
        "timestamp": ts,
    }


def make_assistant_final_message(
    *,
    text: str,
    output_tokens: int,
    input_tokens: int,
    cache_read_tokens: int,
    total_tokens: int,
    ts: int,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "usage": {
            "input": input_tokens,
            "output": output_tokens,
            "cacheRead": cache_read_tokens,
            "cacheWrite": 0,
            "totalTokens": total_tokens,
        },
        "stopReason": "stop",
        "timestamp": ts,
    }


def compute_usage_fields(
    history_tokens: int,
    output_tokens: int,
    rng: random.Random,
) -> tuple[int, int, int]:
    input_tokens = rng.randint(40, 160)
    cache_read = max(0, ((history_tokens - rng.randint(200, 1200)) // 128) * 128)
    total_tokens = history_tokens + output_tokens + input_tokens
    return input_tokens, cache_read, total_tokens


def compact_history(
    history: list[dict[str, Any]],
    *,
    drop_tokens: int,
    summary_tokens: int,
    clock: TsClock,
    event_id: int,
    rng: random.Random,
    text_gen: RandomTokenDecodeTextGenerator,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dropped_count = 0
    dropped_token_sum = 0
    for msg in history:
        if dropped_token_sum >= drop_tokens:
            break
        dropped_token_sum += estimate_message_tokens(msg)
        dropped_count += 1

    dropped_count = min(dropped_count, len(history))
    remaining = history[dropped_count:]

    header = (
        "The conversation history before this point was compacted into the following synthetic summary:\n\n"
        "<summary>\n"
        f"Compaction Event: {event_id}\n"
        f"Dropped Messages: {dropped_count}\n"
        f"Dropped Estimated Tokens: {dropped_token_sum}\n"
        "Summary Content:\n"
    )
    summary_text = make_target_token_text(header, summary_tokens, rng, text_gen) + "\n</summary>"
    summary_msg = make_user_message(summary_text, clock.next(600, 2400))

    new_history = [summary_msg] + remaining
    return new_history, {
        "dropped_messages": dropped_count,
        "dropped_tokens_est": dropped_token_sum,
    }


def main() -> None:
    args = parse_args()
    if args.tool_steps_min_per_attempt <= 0 or args.tool_steps_max_per_attempt <= 0:
        raise ValueError("tool-steps min/max must be positive integers.")
    if args.tool_steps_min_per_attempt > args.tool_steps_max_per_attempt:
        raise ValueError("tool-steps-min-per-attempt must be <= tool-steps-max-per-attempt.")

    rng = random.Random(args.seed)
    start_ms = int(time.time() * 1000)
    clock = TsClock(start_ms=start_ms, rng=rng)
    text_gen = RandomTokenDecodeTextGenerator(args.tokenizer_model)

    use_mix_60_40 = args.output_token_source == "mix_60_40"
    output_token_pmf: list[tuple[int, float]] = []
    if not use_mix_60_40:
        output_token_samples: list[int] = []
        if args.output_token_source in ("replay_plan", "auto"):
            output_token_samples = load_output_token_samples_from_replay_plan(Path(args.reference_replay_plan))
        if not output_token_samples and args.output_token_source in ("messages", "auto", "replay_plan"):
            output_token_samples = load_output_token_samples_from_messages(Path(args.reference_messages))
        if not output_token_samples and args.output_token_source in (
            "load_responses",
            "auto",
            "messages",
            "replay_plan",
        ):
            output_token_samples = load_output_token_samples_from_load_responses(
                Path(args.reference_load_responses)
            )
        output_token_pmf = build_output_token_pmf(output_token_samples)

    history: list[dict[str, Any]] = []
    snapshots: list[list[dict[str, Any]]] = []

    compaction_count = 0
    tool_call_seq = 1

    for attempt_idx in range(1, args.attempts + 1):
        # Ensure we compact as needed before appending a new attempt query.
        while estimate_history_tokens(history) > args.compaction_threshold and history:
            compaction_count += 1
            history, _stats = compact_history(
                history,
                drop_tokens=args.drop_tokens,
                summary_tokens=args.summary_tokens,
                clock=clock,
                event_id=compaction_count,
                rng=rng,
                text_gen=text_gen,
            )
            snapshots.append(copy.deepcopy(history))

        query_target = rng.randint(args.attempt_query_min_tokens, args.attempt_query_max_tokens)
        user_query = make_target_token_text(
            header=f"[Attempt {attempt_idx:02d}] Synthetic long user query payload.",
            target_tokens=query_target,
            rng=rng,
            text_gen=text_gen,
        )
        history.append(make_user_message(user_query, clock.next()))
        snapshots.append(copy.deepcopy(history))

        step_count_this_attempt = rng.randint(
            args.tool_steps_min_per_attempt, args.tool_steps_max_per_attempt
        )
        for step_idx in range(1, step_count_this_attempt + 1):
            history_tokens = estimate_history_tokens(history)
            out_toks = (
                int(sample_output_tokens_mix_60_40(rng))
                if use_mix_60_40
                else int(sample_output_tokens_from_pmf(output_token_pmf, rng))
            )
            in_toks, cache_read, total_toks = compute_usage_fields(history_tokens, out_toks, rng)

            tc_id = f"call_synth_{attempt_idx:02d}_{step_idx:02d}_{tool_call_seq:04d}"
            tool_call_seq += 1
            tool_name = "exec"
            tool_args = {
                "command": f"synthetic_step --attempt {attempt_idx} --step {step_idx}",
            }
            assistant_text = make_target_token_text(
                header=f"Attempt {attempt_idx} step {step_idx}: preparing execution plan and tool call.",
                target_tokens=out_toks,
                rng=rng,
                text_gen=text_gen,
            )
            history.append(
                make_assistant_tool_message(
                    text=assistant_text,
                    output_tokens=out_toks,
                    input_tokens=in_toks,
                    cache_read_tokens=cache_read,
                    total_tokens=total_toks,
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    ts=clock.next(),
                )
            )

            tool_output = (
                f"[synthetic tool output] attempt={attempt_idx} step={step_idx} "
                f"artifact_id={rng.randint(100000, 999999)} status=ok"
            )
            history.append(
                make_tool_result_message(
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    text=tool_output,
                    ts=clock.next(),
                )
            )
            snapshots.append(copy.deepcopy(history))

            while estimate_history_tokens(history) > args.compaction_threshold and history:
                compaction_count += 1
                history, _stats = compact_history(
                    history,
                    drop_tokens=args.drop_tokens,
                    summary_tokens=args.summary_tokens,
                    clock=clock,
                    event_id=compaction_count,
                    rng=rng,
                    text_gen=text_gen,
                )
                snapshots.append(copy.deepcopy(history))

        history_tokens = estimate_history_tokens(history)
        out_toks = (
            int(sample_output_tokens_mix_60_40(rng))
            if use_mix_60_40
            else int(sample_output_tokens_from_pmf(output_token_pmf, rng))
        )
        in_toks, cache_read, total_toks = compute_usage_fields(history_tokens, out_toks, rng)
        final_text = make_target_token_text(
            header=f"Attempt {attempt_idx}: consolidated assistant response with deliverable summary.",
            target_tokens=out_toks,
            rng=rng,
            text_gen=text_gen,
        )
        history.append(
            make_assistant_final_message(
                text=final_text,
                output_tokens=out_toks,
                input_tokens=in_toks,
                cache_read_tokens=cache_read,
                total_tokens=total_toks,
                ts=clock.next(),
            )
        )
        snapshots.append(copy.deepcopy(history))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")

    final_history_tokens = estimate_history_tokens(history)
    print(f"Wrote synthetic trace: {out_path}")
    print(f"Snapshots: {len(snapshots)}")
    print(f"Final messages in history: {len(history)}")
    print(f"Compaction events: {compaction_count}")
    print(f"Estimated final history tokens: {final_history_tokens}")
    if use_mix_60_40:
        print("Output-token source: mix_60_40 (60%:[50,150], 40%:[200,3000])")
    else:
        top_pmf = sorted(output_token_pmf, key=lambda x: x[1], reverse=True)[:10]
        top_pmf_text = ", ".join(f"{v}:{p:.3f}" for v, p in top_pmf)
        print(f"Output-token PMF support size: {len(output_token_pmf)}")
        print(f"Output-token PMF top probs: {top_pmf_text}")


if __name__ == "__main__":
    main()
