from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from agent_bench.core.profile_io import load_profile


def configured_task_container_cpus(
    env: Mapping[str, str] | None = None,
) -> float | None:
    """Return the Docker CPU budget for benchmark task containers, if set."""
    source_env = os.environ if env is None else env
    raw = source_env.get("AB_TASK_CONTAINER_CPUS")
    if raw:
        return _parse_cpus(raw, "AB_TASK_CONTAINER_CPUS")

    config_path = source_env.get("AB_EXPERIMENT_CONFIG")
    if not config_path:
        return None
    try:
        config = load_profile(Path(config_path))
    except Exception:
        return None
    limits = config.get("limits")
    if not isinstance(limits, dict) or "task_container_cpus" not in limits:
        return None
    return _parse_cpus(limits["task_container_cpus"], "limits.task_container_cpus")


def docker_cpus_args(cpus: float | None) -> list[str]:
    return ["--cpus", str(cpus)] if cpus is not None else []


def _parse_cpus(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive number")
    return parsed
