from __future__ import annotations

from pathlib import Path

from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)


def _sessions_dir(root: Path, run_id: str) -> Path:
    path = root / run_id / "agents" / "roadmap_bench" / "sessions"
    path.mkdir(parents=True)
    return path


def test_roadmap_parent_session_prefers_exact_current_explicit_key(tmp_path: Path):
    sessions_dir = _sessions_dir(tmp_path, "current-run")
    current_key = "agent:roadmap_bench:explicit:current-run"
    stale_key = "agent:roadmap_bench:explicit:stale-run"
    index = {
        current_key: {
            "sessionId": "current-run",
            "updatedAt": 1,
        },
        stale_key: {
            "sessionId": "stale-run",
            "updatedAt": 999999,
        },
    }

    selected = RoadmapBenchOpenClawCondition._select_parent_session(
        sessions_dir, index
    )

    assert selected is not None
    assert selected[0] == current_key
    assert selected[1]["sessionId"] == "current-run"


def test_roadmap_parent_session_does_not_rank_ambiguous_roots_by_recency(tmp_path: Path):
    sessions_dir = _sessions_dir(tmp_path, "missing-run")
    index = {
        "agent:roadmap_bench:explicit:first": {
            "sessionId": "first",
            "updatedAt": 1,
        },
        "agent:roadmap_bench:explicit:second": {
            "sessionId": "second",
            "updatedAt": 999999,
        },
    }

    assert (
        RoadmapBenchOpenClawCondition._select_parent_session(
            sessions_dir, index
        )
        is None
    )


def test_roadmap_parent_session_keeps_single_legacy_main_fallback(tmp_path: Path):
    sessions_dir = _sessions_dir(tmp_path, "fixture-run")
    key = "agent:roadmap_bench:main"
    index = {
        key: {
            "sessionId": "legacy-session-id",
            "status": "running",
        }
    }

    selected = RoadmapBenchOpenClawCondition._select_parent_session(
        sessions_dir, index
    )

    assert selected is not None
    assert selected[0] == key
