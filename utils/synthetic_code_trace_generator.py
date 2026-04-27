#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any


# Trace structure (mirrors the search generator):
#   a run       = ATTEMPTS attempts
#   each attempt = 1 user input + ROUNDS rounds; final round has no tool result
#   each round   = 1 assistant_output + 1 tool_result_input
# Compaction is evaluated at the end of each attempt.
ATTEMPTS = 12
ROUNDS_PER_ATTEMPT_RANGE = (22, 27)

# First user message at the start of each attempt.
USER_INPUT_RANGE = (300, 500)

# Tool result input following each assistant output within a round.
TOOL_RESULT_RANGE = (50, 100)

# Assistant output per round — three-segment mixture:
#   60% short thinking / tool prep
#   25% medium reply
#   15% long code-writing turn
OUTPUT_SHORT_RANGE = (60, 250)
OUTPUT_MEDIUM_RANGE = (250, 1500)
OUTPUT_LONG_RANGE = (1500, 4000)
OUTPUT_SHORT_PROB = 0.60
OUTPUT_MEDIUM_PROB = 0.25
OUTPUT_LONG_PROB = 0.15

# Compaction: trigger when history exceeds COMPACTION_THRESHOLD at the end of
# an attempt. After compaction history is collapsed down to a single user input
# sized uniformly from POST_COMPACTION_USER_INPUT_RANGE.
COMPACTION_THRESHOLD = 50000
POST_COMPACTION_USER_INPUT_RANGE = (13000, 18000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a synthetic code-agent trace with real decoded text for "
            "user / tool events and placeholder length-only assistant events. "
            "Controlled by length hyperparameters at the top of this file."
        )
    )
    p.add_argument(
        "--output",
        default="synthetic_trace/synthetic_code/trace.json",
        help="Output path for the trace JSON.",
    )
    p.add_argument("--seed", type=int, default=20260414)
    p.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3.5-35B-A3B",
        help="HF tokenizer id used to decode random token ids into target-length text.",
    )
    return p.parse_args()


class RandomTokenDecodeTextGenerator:
    def __init__(self, model_id: str) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from transformers import AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        vocab = self.tokenizer.get_vocab()
        all_ids = sorted({int(v) for v in vocab.values()})
        special = {int(v) for v in (self.tokenizer.all_special_ids or [])}
        self.candidate_ids = [t for t in all_ids if t not in special] or all_ids

    def decode(self, target_tokens: int, rng: random.Random) -> str:
        n = max(0, int(target_tokens))
        if n == 0:
            return ""
        ids = [self.candidate_ids[rng.randrange(len(self.candidate_ids))] for _ in range(n)]
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


def sample_output_tokens(rng: random.Random) -> int:
    r = rng.random()
    if r < OUTPUT_SHORT_PROB:
        return rng.randint(*OUTPUT_SHORT_RANGE)
    if r < OUTPUT_SHORT_PROB + OUTPUT_MEDIUM_PROB:
        return rng.randint(*OUTPUT_MEDIUM_RANGE)
    return rng.randint(*OUTPUT_LONG_RANGE)


def make_text_chunk(
    kind: str, n: int, rng: random.Random, gen: RandomTokenDecodeTextGenerator
) -> dict[str, Any]:
    text = gen.decode(n, rng)
    return {"kind": kind, "tokens": int(n), "text": text}


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    gen = RandomTokenDecodeTextGenerator(args.tokenizer_model)
    attempts = ATTEMPTS

    events: list[dict[str, Any]] = []
    history_tokens = 0
    compaction_count = 0
    total_rounds = 0
    attempt_round_counts: list[int] = []

    def maybe_compact_end_of_attempt(attempt_idx: int) -> None:
        nonlocal history_tokens, compaction_count
        if COMPACTION_THRESHOLD <= 0:
            return
        if history_tokens <= COMPACTION_THRESHOLD:
            return
        compaction_count += 1
        dropped = history_tokens
        kept_n = rng.randint(*POST_COMPACTION_USER_INPUT_RANGE)
        events.append(
            {
                "kind": "compaction",
                "event_id": compaction_count,
                "attempt": attempt_idx,
                "dropped_tokens": dropped,
                "kept_after_compaction": kept_n,
            }
        )
        chunk = make_text_chunk("compaction_summary", kept_n, rng, gen)
        chunk["attempt"] = attempt_idx
        events.append(chunk)
        history_tokens = kept_n

    for attempt_idx in range(1, attempts + 1):
        user_n = rng.randint(*USER_INPUT_RANGE)
        chunk_user = make_text_chunk("user_input", user_n, rng, gen)
        chunk_user["attempt"] = attempt_idx
        events.append(chunk_user)
        history_tokens += user_n

        rounds = rng.randint(*ROUNDS_PER_ATTEMPT_RANGE)
        attempt_round_counts.append(rounds)
        for round_idx in range(1, rounds + 1):
            total_rounds += 1

            out_n = sample_output_tokens(rng)
            # Assistant output is a placeholder — only the target length matters;
            # the loader will ask the server to produce `tokens` tokens at replay.
            events.append(
                {
                    "kind": "assistant_output",
                    "attempt": attempt_idx,
                    "round": round_idx,
                    "tokens": int(out_n),
                }
            )
            history_tokens += out_n

            # Last round of an attempt is the final answer — no tool result follows.
            if round_idx == rounds:
                break

            tool_n = rng.randint(*TOOL_RESULT_RANGE)
            chunk_tool = make_text_chunk("tool_result_input", tool_n, rng, gen)
            chunk_tool["attempt"] = attempt_idx
            chunk_tool["round"] = round_idx
            events.append(chunk_tool)
            history_tokens += tool_n

        maybe_compact_end_of_attempt(attempt_idx)

    trace = {
        "meta": {
            "generated_at_ms": int(time.time() * 1000),
            "seed": args.seed,
            "tokenizer_model": args.tokenizer_model,
            "attempts": attempts,
            "rounds_per_attempt": attempt_round_counts,
            "rounds_per_attempt_range": list(ROUNDS_PER_ATTEMPT_RANGE),
            "total_rounds": total_rounds,
            "user_input_range": list(USER_INPUT_RANGE),
            "tool_result_range": list(TOOL_RESULT_RANGE),
            "output_short_range": list(OUTPUT_SHORT_RANGE),
            "output_medium_range": list(OUTPUT_MEDIUM_RANGE),
            "output_long_range": list(OUTPUT_LONG_RANGE),
            "output_short_prob": OUTPUT_SHORT_PROB,
            "output_medium_prob": OUTPUT_MEDIUM_PROB,
            "output_long_prob": OUTPUT_LONG_PROB,
            "compaction_threshold": COMPACTION_THRESHOLD,
            "post_compaction_user_input_range": list(POST_COMPACTION_USER_INPUT_RANGE),
            "final_history_tokens": history_tokens,
            "compaction_events": compaction_count,
        },
        "events": events,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote synthetic code trace: {out_path}")
    print(f"Attempts: {attempts}")
    print(f"Rounds per attempt: {attempt_round_counts}  total={total_rounds}")
    print(f"Events: {len(events)}  Compaction events: {compaction_count}")
    print(f"Final history tokens: {history_tokens}")


if __name__ == "__main__":
    main()
