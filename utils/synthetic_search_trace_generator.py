#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any


# Token id sampling range. Use a generic 32k vocab; ids are opaque integers —
# downstream consumers only care about lengths, not decoded text.
VOCAB_SIZE = 32000


# Trace structure:
#   a run = ATTEMPTS attempts
#   each attempt = 1 user input + ROUNDS_PER_ATTEMPT rounds
#   each round    = 1 assistant_output + 1 tool_result_input
# Compaction is evaluated at the end of each attempt.
ATTEMPTS = 7
ROUNDS_PER_ATTEMPT_RANGE = (4, 6)

# User input at the start of each attempt.
USER_INPUT_RANGE = (500, 1000)

# Tool result input following each assistant output within a round.
TOOL_RESULT_RANGE = (2500, 3000)

# Assistant output per round.
OUTPUT_PER_ROUND_RANGE = (50, 130)

# Compaction: trigger when history exceeds COMPACTION_THRESHOLD at the end of
# an attempt. After compaction the history is collapsed down to a single user
# input sized uniformly from POST_COMPACTION_USER_INPUT_RANGE.
COMPACTION_THRESHOLD = 55000
POST_COMPACTION_USER_INPUT_RANGE = (15000, 20000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a synthetic search-style agent trace as raw random token ids, "
            "controlled purely by length hyperparameters: turns, per-turn input/output "
            "token sizes, compaction threshold, and post-compaction kept tokens."
        )
    )
    p.add_argument(
        "--output",
        default="synthetic_trace/synthetic_search/cache/trace.synthetic.json",
        help="Output path for the trace JSON.",
    )
    p.add_argument("--seed", type=int, default=20260413)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    return p.parse_args()


def sample_token_ids(n: int, rng: random.Random, vocab_size: int) -> list[int]:
    n = max(0, int(n))
    return [rng.randrange(vocab_size) for _ in range(n)]


def make_chunk(kind: str, n: int, rng: random.Random, vocab_size: int) -> dict[str, Any]:
    ids = sample_token_ids(n, rng, vocab_size)
    return {"kind": kind, "tokens": len(ids), "token_ids": ids}


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    vocab_size = args.vocab_size
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
        chunk = make_chunk("compaction_summary", kept_n, rng, vocab_size)
        chunk["attempt"] = attempt_idx
        events.append(chunk)
        history_tokens = kept_n

    for attempt_idx in range(1, attempts + 1):
        user_n = rng.randint(*USER_INPUT_RANGE)
        chunk_user = make_chunk("user_input", user_n, rng, vocab_size)
        chunk_user["attempt"] = attempt_idx
        events.append(chunk_user)
        history_tokens += user_n

        rounds = rng.randint(*ROUNDS_PER_ATTEMPT_RANGE)
        attempt_round_counts.append(rounds)
        for round_idx in range(1, rounds + 1):
            total_rounds += 1

            out_n = rng.randint(*OUTPUT_PER_ROUND_RANGE)
            chunk_out = make_chunk("assistant_output", out_n, rng, vocab_size)
            chunk_out["attempt"] = attempt_idx
            chunk_out["round"] = round_idx
            events.append(chunk_out)
            history_tokens += out_n

            # Last round of an attempt is the final answer — no tool result follows.
            if round_idx == rounds:
                break

            tool_n = rng.randint(*TOOL_RESULT_RANGE)
            chunk_tool = make_chunk("tool_result_input", tool_n, rng, vocab_size)
            chunk_tool["attempt"] = attempt_idx
            chunk_tool["round"] = round_idx
            events.append(chunk_tool)
            history_tokens += tool_n

        maybe_compact_end_of_attempt(attempt_idx)

    trace = {
        "meta": {
            "generated_at_ms": int(time.time() * 1000),
            "seed": args.seed,
            "vocab_size": vocab_size,
            "attempts": attempts,
            "rounds_per_attempt": attempt_round_counts,
            "rounds_per_attempt_range": list(ROUNDS_PER_ATTEMPT_RANGE),
            "total_rounds": total_rounds,
            "user_input_range": list(USER_INPUT_RANGE),
            "tool_result_range": list(TOOL_RESULT_RANGE),
            "output_per_round_range": list(OUTPUT_PER_ROUND_RANGE),
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

    print(f"Wrote synthetic search trace: {out_path}")
    print(f"Attempts: {attempts}")
    print(f"Rounds per attempt: {attempt_round_counts}  total={total_rounds}")
    print(f"Events: {len(events)}  Compaction events: {compaction_count}")
    print(f"Final history tokens: {history_tokens}")


if __name__ == "__main__":
    main()
