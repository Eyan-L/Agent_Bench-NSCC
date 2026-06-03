"""Request-Bench profile loader.

Request-Bench is a tiny diagnostic benchmark for natural-language tasks. The
profile owns the request text; materialization converts it into the same
instance_data shape consumed by the runtime driver.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


def list_task_ids(profile: Mapping[str, Any]) -> List[str]:
    """Return configured request ids in profile order."""
    instances = profile.get("instances") or []
    if instances:
        return [str(item) for item in instances]
    requests = profile.get("requests") or {}
    if not isinstance(requests, Mapping):
        raise ValueError("request_bench profile requests must be an object")
    return [str(key) for key in requests]


def load_task(task_id: str, profile: Mapping[str, Any]) -> Dict[str, Any]:
    requests = profile.get("requests") or {}
    if not isinstance(requests, Mapping):
        raise ValueError("request_bench profile requests must be an object")
    if task_id not in requests:
        raise KeyError(f"request_bench task not found: {task_id}")
    raw = requests[task_id]
    if isinstance(raw, str):
        task = {"prompt": raw}
    elif isinstance(raw, Mapping):
        task = dict(raw)
    else:
        raise TypeError(f"requests.{task_id} must be a string or object")

    prompt = task.get("prompt") or task.get("request")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"requests.{task_id}.prompt must be a non-empty string")

    repo_mapping = profile.get("repo_mapping") or {}
    repo_info = repo_mapping.get(task_id) if isinstance(repo_mapping, Mapping) else None
    if not isinstance(repo_info, Mapping):
        repo_info = {}

    answer_file = str(task.get("answer_file") or "/workspace/answer.md")
    docker_image = str(
        task.get("docker_image")
        or profile.get("container_image")
        or "python:3.13-slim-bookworm"
    )
    timeout = int(
        task.get("agent_timeout_sec")
        or profile.get("wall_clock_budget_seconds")
        or 600
    )

    return {
        "instance_id": task_id,
        "benchmark": "request_bench",
        "task_name": task_id,
        "prompt": prompt,
        "request": prompt,
        "description": str(task.get("description") or "Natural request diagnostic task"),
        "docker_image": docker_image,
        "workspace": "/workspace",
        "answer_file": answer_file,
        "agent_timeout_sec": timeout,
        "repo": str(repo_info.get("repo_id") or f"request_bench/{task_id}"),
        "base_commit": str(repo_info.get("base_commit") or "request-bench-v1"),
        "problem_statement": prompt,
    }


def materialize_profile(profile: Mapping[str, Any], out_path: Path) -> None:
    task_ids = list_task_ids(profile)
    if not task_ids:
        raise ValueError("request_bench materialization found no task ids")
    rows = [load_task(task_id, profile) for task_id in task_ids]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":  # pragma: no cover
    import sys
    from agent_bench.core.profile_io import load_profile

    if len(sys.argv) != 3:
        print("usage: loader.py <profile.json> <out.json>", file=sys.stderr)
        raise SystemExit(2)
    materialize_profile(load_profile(sys.argv[1]), Path(sys.argv[2]))
