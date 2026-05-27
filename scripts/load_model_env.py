#!/usr/bin/env python3
"""Emit shell assignments for a profiles/models/*.json file.

The emitted variables are intentionally runtime-facing: MODEL is the exact
string passed to vLLM/SGLang and to OpenAI-compatible client requests.
Existing env values win so one-off CLI overrides still work.
"""

import json
import os
import sys
from pathlib import Path


_KEYS = [
    ("AB_MODEL_ID", "config_id"),
    ("MODEL", "name"),
    ("MODEL_TEMPERATURE", "temperature"),
    ("MAX_TOKENS", "max_tokens"),
    ("MODEL_CONTEXT_WINDOW", "context_window"),
    ("MAX_MODEL_LEN", "context_window"),
    ("MODEL_PROVIDER", "provider"),
    ("MODEL_API_KEY", "api_key"),
    ("LITELLM_MODEL", "litellm_model"),
    ("TOOL_CALL_PARSER", "tool_call_parser"),
    ("REASONING_PARSER", "reasoning_parser"),
]


def _shell_quote(v):
    s = os.path.expandvars(str(v))
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _emit_default(var, val):
    print(f'if [ -z "${{{var}:-}}" ]; then {var}={_shell_quote(val)}; fi')
    print(f"export {var}")


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <model.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"model config not found: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text()) or {}
    model = data.get("model") or {}
    if not isinstance(model, dict):
        print(f"model config {path} is missing a `model` object", file=sys.stderr)
        return 1

    for var, key in _KEYS:
        val = model.get(key)
        if val is None or val == "":
            continue
        _emit_default(var, val)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
