#!/usr/bin/env python3
"""Emit shell `:=` assignments for the active harness JSON config.

Usage:
    eval "$(scripts/load_harness_env.py config/harness/openclaw_27b.json)"

After eval, the following env vars are set (only if not already set —
existing env wins, so one-off CLI overrides keep working):

    MODEL             model.name
    MODEL_ENDPOINT    model.endpoint
    HARNESS_TYPE      top-level `harness` (e.g. "openclaw", "tb_react")

Companion to scripts/load_backend_env.py. Backend config owns engine knobs;
harness config owns model selection (single source of truth). serve.sh
sources both so a `make serve` knows what to serve AND how.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


_KEYS = [
    # (env var, yaml dotted path)
    ("MODEL",          "model.name"),
    ("MODEL_ENDPOINT", "model.endpoint"),
    ("HARNESS_TYPE",   "harness"),
]


def _lookup(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _shell_quote(v) -> str:
    s = str(v)
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <harness.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"harness config not found: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text()) or {}

    for var, dotted in _KEYS:
        val = _lookup(data, dotted)
        if val is None or val == "":
            continue
        print(f': "${{{var}:={_shell_quote(val)[1:-1]}}}"')
        print(f"export {var}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
