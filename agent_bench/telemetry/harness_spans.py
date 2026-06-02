"""Small JSONL span writer for harness-side subprocess/container work."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_WRITE_LOCK = threading.Lock()


def preview_text(value: Any, limit: int = 4000) -> tuple[str, int, bool]:
    if value is None:
        return "", 0, False
    text = value if isinstance(value, str) else str(value)
    original_len = len(text)
    if limit >= 0 and original_len > limit:
        return text[:limit], original_len, True
    return text, original_len, False


def preview_json(value: Any, limit: int = 4000) -> tuple[str, int, bool]:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return preview_text(text, limit=limit)


def write_harness_span(
    spans_path: Optional[Path],
    *,
    name: str,
    attrs: dict[str, Any],
    start_ns: int,
    end_ns: Optional[int] = None,
    resource: Optional[dict[str, Any]] = None,
) -> None:
    if spans_path is None:
        return
    end_ns = end_ns or time.time_ns()
    spans_path.parent.mkdir(parents=True, exist_ok=True)
    span = {
        "name": name,
        "start_unix_nano": start_ns,
        "end_unix_nano": end_ns,
        "start_iso": datetime.fromtimestamp(
            start_ns / 1e9, tz=timezone.utc,
        ).isoformat(),
        "end_iso": datetime.fromtimestamp(
            end_ns / 1e9, tz=timezone.utc,
        ).isoformat(),
        "duration_ms": (end_ns - start_ns) / 1e6,
        "attrs": attrs,
        "resource": resource or {"service.name": "agent-bench-harness"},
        "synthesized": True,
    }
    with _WRITE_LOCK:
        with spans_path.open("a") as fh:
            fh.write(json.dumps(span, ensure_ascii=False) + "\n")
