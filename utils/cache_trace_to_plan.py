#!/usr/bin/env python3
"""Convert cache-trace JSONL directly into a replay-plan JSON.

Combines the logic of cache_trace_turns_to_json.py (extract turns) and
trace_plan_generator.py (build replay plan) into a single pipeline.

Usage:
    python cache_trace_to_plan.py \
        --input path/to/cache-trace.jsonl \
        --model Qwen/Qwen3.5-35B-A3B \
        --plan-json path/to/replay_plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


# ── Turn extraction constants ──────────────────────────────────────────────
DEFAULT_STAGES = ["stream:context"]
ASSISTANT_DROP_KEYS = {"api", "provider", "model"}


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cache-trace JSONL into a replay-plan JSON.",
    )

    # -- input / output --
    parser.add_argument("--input", required=True, help="Path to cache-trace JSONL.")
    parser.add_argument("--plan-jsonl", default="", help="Optional output path for replay plan JSONL (absolute path).")

    # -- turn extraction options --
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=f"Comma-separated stages to scan. Default: {','.join(DEFAULT_STAGES)}",
    )
    parser.add_argument("--stage", default="", help="Single stage alias for --stages.")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Attach source metadata (_source) to each message.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["fingerprint", "semantic", "none"],
        default="fingerprint",
        help="How to de-duplicate messages. Default: fingerprint.",
    )
    parser.add_argument(
        "--dedupe-snapshot",
        action="store_true",
        help="Convenience flag for snapshot-style de-duplication.",
    )
    parser.add_argument(
        "--turn-mode",
        choices=["snapshot", "user_split"],
        default="snapshot",
        help="Turn construction mode. Default: snapshot.",
    )

    # -- plan generation options --
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B", help="Model name written into payload.")
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3.5-35B-A3B",
        help="Hugging Face tokenizer model id for output-token estimation.",
    )
    parser.add_argument("--show-template", action="store_true")
    parser.add_argument("--show-template-max", type=int, default=1)
    parser.add_argument("--default-output-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-requests", type=int, default=0, help="Only generate first N requests (0 = all).")

    return parser.parse_args()


# ── Turn extraction helpers ────────────────────────────────────────────────
def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _semantic_message_key(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    role = str(message.get("role", "unknown"))
    timestamp = message.get("timestamp")

    if role == "toolResult":
        tool_call_id = message.get("toolCallId")
        if tool_call_id:
            return f"sem:toolResult:{tool_call_id}"
        if timestamp is not None:
            return f"sem:toolResult:ts:{timestamp}"

    if role == "assistant":
        content = message.get("content")
        if isinstance(content, list):
            tool_call_ids: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    tcid = block.get("id")
                    if tcid:
                        tool_call_ids.append(str(tcid))
            if tool_call_ids:
                return "sem:assistant:toolCalls:" + ",".join(tool_call_ids)
        if timestamp is not None:
            return f"sem:assistant:ts:{timestamp}"

    if role == "user" and timestamp is not None:
        return f"sem:user:ts:{timestamp}"

    return ""


def _normalize_message(msg: Any, *, keep_source: bool, source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(msg, dict):
        out: dict[str, Any] = {"role": "unknown", "raw": msg}
        if keep_source:
            out["_source"] = source
        return out

    out = {"role": msg.get("role", "unknown")}
    role = str(msg.get("role", "unknown"))
    for key, value in msg.items():
        if key == "role":
            continue
        if role == "assistant" and key in ASSISTANT_DROP_KEYS:
            continue
        if role == "assistant" and key == "usage" and isinstance(value, dict):
            usage_copy = dict(value)
            usage_copy.pop("cost", None)
            out[key] = usage_copy
            continue
        out[key] = value
    if keep_source:
        out["_source"] = source
    return out


def _collect_messages(
    input_path: Path,
    *,
    stages: set[str],
    keep_source: bool,
    dedupe_mode: str,
) -> list[dict[str, Any]]:
    seen_keys: set[str] = set()
    ordered_messages: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if stages and str(event.get("stage", "")) not in stages:
                continue

            messages = event.get("messages")
            if not isinstance(messages, list) or not messages:
                continue

            fingerprints = event.get("messageFingerprints")
            has_fingerprints = isinstance(fingerprints, list) and len(fingerprints) == len(messages)

            source = {
                "line": line_no,
                "runId": event.get("runId"),
                "seq": event.get("seq"),
                "stage": event.get("stage"),
                "ts": event.get("ts"),
            }

            for idx, message in enumerate(messages):
                if dedupe_mode != "none":
                    if dedupe_mode == "semantic":
                        semantic_key = _semantic_message_key(message)
                        if semantic_key:
                            dedupe_key = semantic_key
                        elif has_fingerprints and fingerprints[idx]:
                            dedupe_key = f"fp:{fingerprints[idx]}"
                        else:
                            dedupe_key = f"hash:{_stable_hash(message)}:{idx}"
                    else:
                        if has_fingerprints and fingerprints[idx]:
                            dedupe_key = f"fp:{fingerprints[idx]}"
                        else:
                            dedupe_key = f"hash:{_stable_hash(message)}:{idx}"

                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)

                ordered_messages.append(
                    _normalize_message(message, keep_source=keep_source, source=source)
                )

    return ordered_messages


def _split_into_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role", "unknown"))
        if role == "user":
            if current:
                turns.append(current)
            current = [msg]
        else:
            if not current:
                current = [msg]
            else:
                current.append(msg)

    if current:
        turns.append(current)
    return turns


def _collect_snapshot_turns(
    input_path: Path,
    *,
    stages: set[str],
    keep_source: bool,
    dedupe_mode: str,
) -> list[list[dict[str, Any]]]:
    seen_snapshots: set[str] = set()
    turns: list[list[dict[str, Any]]] = []

    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if stages and str(event.get("stage", "")) not in stages:
                continue

            messages = event.get("messages")
            if not isinstance(messages, list) or not messages:
                continue

            if dedupe_mode != "none":
                digest = event.get("messagesDigest")
                run_id = event.get("runId")
                if isinstance(digest, str) and digest:
                    snapshot_key = f"{run_id}:{digest}"
                else:
                    snapshot_key = f"{run_id}:{_stable_hash(messages)}"
                if snapshot_key in seen_snapshots:
                    continue
                seen_snapshots.add(snapshot_key)

            source = {
                "line": line_no,
                "runId": event.get("runId"),
                "seq": event.get("seq"),
                "stage": event.get("stage"),
                "ts": event.get("ts"),
            }

            normalized = [
                _normalize_message(m, keep_source=keep_source, source=source)
                for m in messages
            ]
            turns.append(normalized)

    return turns


def _extract_turns(args: argparse.Namespace, input_path: Path) -> list[list[dict[str, Any]]]:
    stage_spec = args.stage.strip() if isinstance(args.stage, str) else ""
    if stage_spec:
        stages = {stage_spec}
    else:
        stages = {s.strip() for s in args.stages.split(",") if s.strip()}

    dedupe_mode = "fingerprint" if args.dedupe_snapshot else args.dedupe_mode

    if args.turn_mode == "snapshot":
        turns = _collect_snapshot_turns(
            input_path, stages=stages, keep_source=args.keep_source, dedupe_mode=dedupe_mode,
        )
        msg_count = sum(len(t) for t in turns)
        print(f"[turns] {len(turns)} turns / {msg_count} messages extracted")
    else:
        messages = _collect_messages(
            input_path, stages=stages, keep_source=args.keep_source, dedupe_mode=dedupe_mode,
        )
        turns = _split_into_turns(messages)
        print(f"[turns] {len(turns)} turns / {len(messages)} messages extracted")

    return turns


# ── Plan generation helpers ────────────────────────────────────────────────
def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _build_hf_tokenizer(model_id: str):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "transformers is required. Install via: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _flatten_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        return _safe_json_dumps(content)
    if not isinstance(content, list):
        return _safe_json_dumps(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(_safe_json_dumps(block))
            continue

        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "toolCall":
            tc_payload = {
                "type": "toolCall",
                "id": block.get("id"),
                "name": block.get("name"),
                "arguments": block.get("arguments"),
            }
            parts.append(_safe_json_dumps(tc_payload))
        else:
            parts.append(_safe_json_dumps(block))
    return "\n".join(p for p in parts if p)


def _assistant_to_openai_message(msg: dict[str, Any]) -> dict[str, Any]:
    content = msg.get("content")
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                text_parts.append(_safe_json_dumps(block))
                continue

            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "toolCall":
                arguments = block.get("arguments")
                arguments_str = arguments if isinstance(arguments, str) else _safe_json_dumps(arguments)
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": arguments_str,
                        },
                    }
                )
            else:
                text_parts.append(_safe_json_dumps(block))
    else:
        text_parts.append(_safe_json_dumps(content))

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(p for p in text_parts if p),
    }
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    return assistant_msg


def _to_openai_messages(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for msg in turn:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))

        if role == "user":
            out.append({"role": "user", "content": _flatten_content_to_text(msg.get("content"))})
        elif role == "assistant":
            out.append(_assistant_to_openai_message(msg))
        elif role == "toolResult":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(msg.get("toolCallId", "")),
                    "content": _flatten_content_to_text(msg.get("content")),
                }
            )
        else:
            out.append({"role": role or "user", "content": _flatten_content_to_text(msg.get("content"))})

    return out


def _assistant_template_text(tokenizer: Any, assistant_msg: dict[str, Any]) -> str:
    assistant_text = _flatten_content_to_text(assistant_msg.get("content"))
    template_messages = [
        {"role": "user", "content": "_"},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(
        template_messages, tokenize=False, add_generation_prompt=False,
    )


def _assistant_output_tokens(tokenizer: Any, assistant_msg: dict[str, Any]) -> int:
    assistant_text = _flatten_content_to_text(assistant_msg.get("content"))
    base_messages = [{"role": "user", "content": "_"}]
    full_messages = [
        {"role": "user", "content": "_"},
        {"role": "assistant", "content": assistant_text},
    ]
    base_encoding = tokenizer.apply_chat_template(
        base_messages, tokenize=True, add_generation_prompt=False,
    )
    full_encoding = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
    )
    return _encoding_len(full_encoding) - _encoding_len(base_encoding)


def _encoding_len(enc: Any) -> int:
    """len() across tokenizer return shapes.

    `apply_chat_template(tokenize=True)` returns a plain `list[int]` in
    transformers >= 4.44, and a `BatchEncoding({"input_ids": [...]})` on
    older versions. Handle both + the rare tensor return.
    """
    if isinstance(enc, list):
        return len(enc)
    if hasattr(enc, "keys") and "input_ids" in enc:
        ids = enc["input_ids"]
        if isinstance(ids, list):
            # Could be [int...] or [[int...]] — flatten one level.
            if ids and isinstance(ids[0], list):
                return len(ids[0])
            return len(ids)
        # tensor-like
        return int(ids.shape[-1])
    # Fallback: tensor with .shape
    if hasattr(enc, "shape"):
        return int(enc.shape[-1])
    return len(enc)


def _pick_target_assistant(
    next_turn: list[dict[str, Any]],
    start_index: int = 0,
) -> dict[str, Any] | None:
    """Return the assistant message produced by the request whose input
    context was the predecessor turn. Since turns are append-only snapshots,
    the new assistant lives at index >= start_index (= len(prev_turn)).
    Handles all tail shapes: [..., assistant], [..., assistant, toolResult],
    [..., assistant, user] (framework-injected next sub-task)."""
    if not next_turn:
        return None
    for idx in range(max(0, start_index), len(next_turn)):
        m = next_turn[idx]
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m
    # Fallback for compaction / non-append-only snapshots.
    for m in reversed(next_turn):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m
    return None


def _target_assistant_for_request(
    turns: list[list[dict[str, Any]]], req_index: int,
) -> dict[str, Any] | None:
    next_index = req_index + 1
    if next_index >= len(turns):
        return None
    current_turn = turns[req_index] if isinstance(turns[req_index], list) else []
    return _pick_target_assistant(turns[next_index], start_index=len(current_turn))


def _infer_request_output_tokens(
    turns: list[list[dict[str, Any]]],
    req_index: int,
    *,
    tokenizer: Any,
    default_output_tokens: int,
) -> int:
    target_assistant = _target_assistant_for_request(turns, req_index)
    if target_assistant is None:
        return default_output_tokens
    return _assistant_output_tokens(tokenizer, target_assistant)


def _write_jsonl_line(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()

    # Step 1: extract turns from cache-trace
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Derive output paths: place messages.json and replay_plan.json next to input
    output_dir = input_path.parent

    turns = _extract_turns(args, input_path)

    # Save intermediate turns JSON
    turns_path = output_dir / "messages.json"
    turns_path.write_text(
        json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[turns] saved to {turns_path}")

    # Step 2: generate replay plan
    tokenizer = _build_hf_tokenizer(args.tokenizer_model)

    plan_json_path = output_dir / "replay_plan.json"

    plan_jsonl_path: Path | None = None
    if args.plan_jsonl:
        plan_jsonl_path = Path(args.plan_jsonl).expanduser().resolve()
        plan_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if plan_jsonl_path.exists():
            plan_jsonl_path.unlink()

    max_requests = args.max_requests if args.max_requests > 0 else len(turns)
    max_requests = min(max_requests, len(turns))

    shown_templates = 0
    plan_rows: list[dict[str, Any]] = []

    for i in range(max_requests):
        turn = turns[i]
        if not isinstance(turn, list):
            continue

        target_assistant = _target_assistant_for_request(turns, i)
        max_tokens = _infer_request_output_tokens(
            turns, i, tokenizer=tokenizer, default_output_tokens=args.default_output_tokens,
        )

        if args.show_template and target_assistant is not None and shown_templates < args.show_template_max:
            template_text = _assistant_template_text(tokenizer, target_assistant)
            print(f"===== TEMPLATE requestIndex={i+1} =====")
            print(target_assistant)
            print(template_text)
            print("===== END TEMPLATE =====")
            shown_templates += 1

        messages = _to_openai_messages(turn)
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": args.temperature,
            "ignore_eos": True,
            "stop": [],
            "stop_token_ids": [],
        }

        roles = [m.get("role", "unknown") for m in turn if isinstance(m, dict)]
        plan_row = {
            "requestIndex": i + 1,
            "messageCount": len(messages),
            "messageRoles": roles,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "payload": payload,
        }

        if plan_jsonl_path:
            _write_jsonl_line(plan_jsonl_path, plan_row)
        plan_rows.append(plan_row)

        print(f"[{i+1}/{max_requests}] roles={roles} max_tokens={max_tokens}")

    plan_json_path.write_text(
        json.dumps(plan_rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"finished generated={len(plan_rows)} total={max_requests}")
    print(f"plan_json={plan_json_path}")
    if plan_jsonl_path:
        print(f"plan_jsonl={plan_jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
