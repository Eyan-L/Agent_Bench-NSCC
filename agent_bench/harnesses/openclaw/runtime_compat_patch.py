#!/usr/bin/env python3
"""Patch pinned OpenClaw runtime details that remain NSCC-specific.

The current RoadmapBench migration target is OpenClaw v2026.9.2. Unlike the old
v2026.4.22 runtime, this release already contains native rolling tool-loop
detection, a post-compaction loop guard, and richer OpenTelemetry diagnostics.
Those behaviors must not be backported here.

Two compatibility concerns remain relevant on firewalled NSCC nodes:

* Older OpenClaw builds used a hard-coded 10 second ``sessions.patch`` timeout
  while materializing subagents. The v2026.9.2 migration does not assume that
  old anchor still exists. The patch can still relax it when present, but the
  caller may explicitly allow the anchor to be absent.
* The bundled model-pricing catalogue still contains public OpenRouter/LiteLLM
  endpoints. RoadmapBench uses a fully specified local SGLang model, so those
  requests are unnecessary and unreliable on NSCC. They remain fail-closed by
  default and are replaced with local ``data:`` responses.

This script edits only exact runtime anchors, records every replacement, and is
run while building a writable OpenClaw SIF sandbox.
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
OPENROUTER_OFFLINE_URL = "data:application/json,%7B%22data%22%3A%5B%5D%7D"
LITELLM_OFFLINE_URL = "data:application/json,%7B%7D"

# Historical source shape:
#   callGateway({ method: "sessions.patch", ..., timeoutMs: 10_000 })
# Bundlers may render the literal as 10000 or 1e4. On v2026.9.2 this anchor is
# optional: if upstream has already changed the spawn transport/deadline, the
# migration build must not fail merely because the old literal disappeared.
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
            "could not locate the historical sessions.patch 10s timeout; "
            "refusing to build while require_session_patch=True"
        )
    if require_pricing and (openrouter_replacements == 0 or litellm_replacements == 0):
        raise RuntimeError(
            "could not locate both OpenRouter and LiteLLM pricing endpoints; "
            "refusing to build a runtime that may perform public pricing fetches "
            f"(openrouter={openrouter_replacements}, litellm={litellm_replacements})"
        )

    return {
        "runtime_root": str(root),
        "files_scanned": files_scanned,
        "files_changed": files_changed,
        "session_patch_timeout_ms": session_patch_timeout_ms,
        "session_patch_required": require_session_patch,
        "sessions_patch_timeout_replacements": session_patch_replacements,
        "pricing_required": require_pricing,
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
    parser.add_argument(
        "--allow-missing-session-patch",
        action="store_true",
        help=(
            "do not fail when the historical 10s sessions.patch anchor is absent; "
            "intended for v2026.9.2+ where upstream may already have changed it"
        ),
    )
    parser.add_argument("--status-out", default="")
    args = parser.parse_args()

    status = patch_runtime_tree(
        Path(args.root),
        session_patch_timeout_ms=args.session_patch_timeout_ms,
        require_session_patch=not args.allow_missing_session_patch,
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
