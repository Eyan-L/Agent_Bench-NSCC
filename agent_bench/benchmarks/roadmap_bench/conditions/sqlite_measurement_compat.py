"""Final Roadmap measurement normalization for SQLite-backed OpenClaw sessions.

The v2026.9.2 SQLite adapter teaches the legacy token aggregator that an
``agent:<id>:explicit:<run-id>`` root is the parent.  The aggregate token
buckets are therefore correct, but the accompanying ``model_usage`` events
are produced before that bucket repair and can still carry ``role=child``.

Keep this final adapter narrow: repair only explicit-root event attribution and
remove the legacy v2026.4.22 wording from runtime metadata when v9.2 is active.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def install_roadmap_sqlite_measurement_compat() -> None:
    from agent_bench.benchmarks.conditions import openclaw_measurement
    from .sqlite_session_compat import _measurement_parent_key, _runtime_version

    current_logical: Callable[..., tuple[dict, list[dict]]] = (
        openclaw_measurement._logical_tokens
    )
    if not getattr(current_logical, "_roadmap_sqlite_usage_role_compat", False):
        def logical_tokens(run_dir: Path, agent_id: str):
            logical, events = current_logical(run_dir, agent_id)
            if agent_id != "roadmap_bench":
                return logical, events
            parent_key = _measurement_parent_key(Path(run_dir), agent_id)
            if parent_key:
                for event in events:
                    if (
                        event.get("session_key") == parent_key
                        and event.get("role") in {"parent", "child"}
                    ):
                        event["role"] = "parent"
            return logical, events

        setattr(logical_tokens, "_roadmap_sqlite_usage_role_compat", True)
        openclaw_measurement._logical_tokens = logical_tokens

    current_runtime: Callable[..., dict] = openclaw_measurement._runtime_observability
    if not getattr(current_runtime, "_roadmap_v9_runtime_metadata_compat", False):
        def runtime_observability(run_dir: Path, events: list[dict], agent_id: str):
            result = current_runtime(run_dir, events, agent_id)
            if agent_id != "roadmap_bench" or _runtime_version() != "v2026.9.2":
                return result
            attribution = result.get("announce_attribution")
            if isinstance(attribution, dict):
                attribution["limitation"] = (
                    "events without explicit child/session identity remain "
                    "unattributed; Agent_Bench does not infer identity from timing"
                )
            return result

        setattr(runtime_observability, "_roadmap_v9_runtime_metadata_compat", True)
        openclaw_measurement._runtime_observability = runtime_observability
