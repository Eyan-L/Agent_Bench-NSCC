#!/usr/bin/env python3
"""Patch pinned OpenClaw runtime details that are not configurable upstream.

RoadmapBench currently pins OpenClaw v2026.4.22 in the NSCC runtime SIF. Two
runtime behaviours matter for reliable fan-out on firewalled compute nodes:

* ``sessions_spawn`` patches the freshly-created child session through the
  Gateway with a hard-coded 10 second ``sessions.patch`` deadline. Real NSCC
  runs have observed a valid patch RPC taking >13 seconds, so that deadline can
  reject a child that would otherwise start normally.
* the bundled model-pricing catalogue performs public OpenRouter/LiteLLM HTTP
  fetches even though RoadmapBench uses a fully specified local SGLang model.
  Those requests are both unnecessary and unreliable on NSCC.

This script is intentionally narrow: it edits only JavaScript runtime bundles
containing the exact known anchors, records what it changed, and fails closed
when an expected anchor cannot be found. It is run while building the writable
OpenClaw overlay, never against the immutable upstream SIF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_SESSION_PATCH_TIMEOUT_MS = 45_000

OPENROUTER_PRICING_URL = "https://openrouter.ai/api/v1/models"
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
# Node's fetch() supports data: URLs. Keep the response shapes accepted by the
# two catalogue loaders while guaranteeing that no public-network request is
# issued for pricing metadata.
OPENROUTER_OFFLINE_URL = "data:application/json,%7B%22data%22%3A%5B%5D%7D"
LITELLM_OFFLINE_URL = "data:application/json,%7B%7D"

# v2026.4.22 source shape:
#   callGateway({ method: "sessions.patch", ..., timeoutMs: 10_000 })
# Bundlers may render the literal as 10000 or 1e4, hence the alternatives.
_SESSION_PATCH_TIMEOUT_RE = re.compile(
    r"(method\s*:\s*['\"]sessions\.patch['\"][\s\S]{0,2000}?"
    r"timeoutMs\s*:\s*)(?:10_000|10000|1e4)(?![\w.])"
)


def _javascript_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.js", "*.mjs", "*.cjs"):
        files.extend(path for path in root.rglob(suffix) if path.is_file())
    return sorted(set(files))


def patch_runtime_tree(
    root: Path,
    *,
    session_patch_timeout_ms: int = DEFAULT_SESSION_PATCH_TIMEOUT_MS,
    require_session_patch: bool = True,
    require_pricing: bool = True,
) -> dict[str, Any]:
    """Patch OpenClaw JS bundles below ``root`` and return a status record."""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(f"OpenClaw runtime root does not exist: {root}")
    if session_patch_timeout_ms < 10_000:
        raise ValueError("session patch timeout must be at least 10000 ms")

    files_scanned = 0
    files_changed: list[str] = []
    session_patch_replacements = 0
    openrouter_replacements = 0
    litellm_replacements = 0

    for path in _javascript_files(root):
        files_scanned += 1
        try:
            original = path.read_text(errors="strict")
        except (OSError, UnicodeDecodeError):
            continue
        updated = original

        updated, count = _SESSION_PATCH_TIMEOUT_RE.subn(
            lambda match: f"{match.group(1)}{session_patch_timeout_ms}",
            updated,
        )
        session_patch_replacements += count

        count = updated.count(OPENROUTER_PRICING_URL)
        if count:
            updated = updated.replace(OPENROUTER_PRICING_URL, OPENROUTER_OFFLINE_URL)
            openrouter_replacements += count

        count = updated.count(LITELLM_PRICING_URL)
        if count:
            updated = updated.replace(LITELLM_PRICING_URL, LITELLM_OFFLINE_URL)
            litellm_replacements += count

        if updated != original:
            path.write_text(updated)
            files_changed.append(str(path.relative_to(root)))

    if require_session_patch and session_patch_replacements == 0:
        raise RuntimeError(
            "could not locate the v2026.4.22 sessions.patch 10s timeout; "
            "refusing to build an overlay that may silently reject subagents"
        )
    if require_pricing and (openrouter_replacements == 0 or litellm_replacements == 0):
        raise RuntimeError(
            "could not locate both OpenRouter and LiteLLM pricing endpoints; "
            "refusing to build an overlay that may perform public pricing fetches "
            f"(openrouter={openrouter_replacements}, litellm={litellm_replacements})"
        )

    return {
        "runtime_root": str(root),
        "files_scanned": files_scanned,
        "files_changed": files_changed,
        "session_patch_timeout_ms": session_patch_timeout_ms,
        "sessions_patch_timeout_replacements": session_patch_replacements,
        "openrouter_pricing_url_replacements": openrouter_replacements,
        "litellm_pricing_url_replacements": litellm_replacements,
        "pricing_mode": "offline_data_urls",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/app")
    parser.add_argument(
        "--session-patch-timeout-ms",
        type=int,
        default=DEFAULT_SESSION_PATCH_TIMEOUT_MS,
    )
    parser.add_argument("--status-out", default="")
    args = parser.parse_args()

    status = patch_runtime_tree(
        Path(args.root),
        session_patch_timeout_ms=args.session_patch_timeout_ms,
    )
    rendered = json.dumps(status, indent=2, sort_keys=True)
    if args.status_out:
        out = Path(args.status_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
