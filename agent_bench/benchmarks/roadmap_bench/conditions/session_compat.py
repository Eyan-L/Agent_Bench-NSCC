"""RoadmapBench OpenClaw session-key compatibility.

OpenClaw v2026.9.2 maps an explicit CLI ``--session-id`` to
``agent:<agent-id>:explicit:<session-id>``.  RoadmapBench state directories are
per-run and named by the benchmark run/session id, so formal runs can resolve
the durable requester exactly instead of guessing from ``updatedAt`` or
transcript mtimes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple


def _root_session(value: object) -> bool:
    return (
        isinstance(value, dict)
        and not value.get("spawnedBy")
        and bool(value.get("sessionId"))
    )


def select_roadmap_parent_session(
    cls,
    sessions_dir: Path,
    index: Dict,
) -> Optional[Tuple[str, Dict]]:
    """Resolve the current Roadmap requester without recency heuristics.

    Resolution order is deliberately fail-closed:

    1. exact v2026.9.2 explicit key derived from the per-run state directory;
    2. legacy ``:main`` only when its stored session id matches the run id;
    3. compatibility fallback only when the store has exactly one unambiguous
       root requester (needed by older fixtures/runtimes that do not expose the
       per-run id in the state-directory name).

    Multiple unmatched root sessions are never ranked by recency.
    """
    prefix = f"agent:{cls.agent_id}:"
    try:
        expected_session_id = sessions_dir.parents[2].name.strip()
    except (IndexError, AttributeError):
        expected_session_id = ""

    def matching_entry(key: str, *, require_expected: bool) -> Optional[Tuple[str, Dict]]:
        value = index.get(key)
        if not _root_session(value):
            return None
        session_id = str(value.get("sessionId") or "").strip()
        if require_expected and session_id != expected_session_id:
            return None
        return key, value

    if expected_session_id:
        explicit_key = f"{prefix}explicit:{expected_session_id}"
        exact = matching_entry(explicit_key, require_expected=True)
        if exact is not None:
            return exact

        legacy = matching_entry(f"{prefix}main", require_expected=True)
        if legacy is not None:
            return legacy

    roots: list[Tuple[str, Dict]] = []
    for key, value in index.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if _root_session(value):
            roots.append((key, value))

    if len(roots) != 1:
        return None

    key, value = roots[0]
    session_id = str(value.get("sessionId") or "").strip()
    if key == f"{prefix}main" or key == f"{prefix}explicit:{session_id}":
        return key, value
    return None
