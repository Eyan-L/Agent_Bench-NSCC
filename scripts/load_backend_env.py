#!/usr/bin/env python3
"""Emit shell `:=` assignments for the active backend JSON config.

Usage:
    eval "$(scripts/load_backend_env.py config/backend/vllm/tp1.json)"

After eval, the following env vars are set (only if not already set —
existing env wins, so one-off CLI overrides keep working):

    TP                     runtime.tp
    PORT                   runtime.port
    GPU_MEM_UTIL           runtime.gpu_mem_util
    MAX_MODEL_LEN          runtime.max_model_len
    CUDA_VISIBLE_DEVICES   runtime.cuda_visible_devices
    SERVING_PRESET         serving_configs[0].preset   (first preset listed)

Backend configs are engine-only — model selection lives in the harness
config (see scripts/load_harness_env.py). serve.sh sources both.

Intentionally narrow: nested values are skipped. If a field is missing
from the JSON, the assignment is skipped — the consuming shell script
keeps its own fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


_KEYS = [
    # (env var, yaml dotted path)
    ("TP",                   "runtime.tp"),
    ("PORT",                 "runtime.port"),
    ("GPU_MEM_UTIL",         "runtime.gpu_mem_util"),
    ("MAX_MODEL_LEN",        "runtime.max_model_len"),
    ("CUDA_VISIBLE_DEVICES", "runtime.cuda_visible_devices"),
]


def _lookup(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _shell_quote(v) -> str:
    """Single-quote for shell. Doubles any embedded single quotes."""
    s = str(v)
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <backend.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"backend config not found: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text()) or {}

    for var, dotted in _KEYS:
        val = _lookup(data, dotted)
        if val is None or val == "":
            continue
        # `:=` keeps existing env-var value if set; only fills in when empty.
        print(f': "${{{var}:={_shell_quote(val)[1:-1]}}}"')
        print(f"export {var}")

    # First serving preset, when present. Lets serve.sh default --preset
    # to whatever the backend yaml lists first; user can still override.
    sc = data.get("serving_configs") or []
    if sc:
        first = sc[0]
        preset = first.get("preset")
        if preset:
            print(f': "${{SERVING_PRESET:={_shell_quote(preset)[1:-1]}}}"')
            print("export SERVING_PRESET")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
