"""Terminal-Bench 2 task loader.

Maps a TB2 task directory tree (instruction.md + task.toml + tests/ + environment/)
into the same `instance_data` dict shape the agent_bench/runtime driver consumes for
SWE-bench, with a `benchmark="terminal_bench"` discriminator so harness +
condition can branch.

Vendored task layout (one task per dir):

    tasks/
    └── <task-id>/
        ├── instruction.md             # task spec sent to agent
        ├── task.toml                  # docker_image, timeouts, resources
        ├── environment/Dockerfile     # only used as documentation; we pull
        │                              # the prebuilt docker_image instead
        └── tests/
            ├── test.sh                # verifier entrypoint (mounted at /tests)
            └── test_outputs.py        # pytest cases, mounted at /tests

The driver expects `instance_data["instance_id"]`. We use the task directory
name (e.g. "kv-store-grpc"). All other fields are TB-specific and ignored by
the SWE-bench code path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _tasks_root() -> Path:
    """Vendored TB task tree at `agent_bench/benchmarks/terminal_bench/tasks/`.
    Override via OC_TB_TASKS_ROOT."""
    env = os.environ.get("OC_TB_TASKS_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "tasks"


def list_task_ids() -> List[str]:
    """Every immediate subdir of tasks/ that has both instruction.md + task.toml."""
    root = _tasks_root()
    if not root.is_dir():
        return []
    out: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "instruction.md").is_file() and (child / "task.toml").is_file():
            out.append(child.name)
    return out


def load_task(task_id: str) -> Dict:
    """Build the instance_data dict for a single TB task."""
    task_dir = _tasks_root() / task_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"TB task dir not found: {task_dir}")

    instruction = (task_dir / "instruction.md").read_text()
    with (task_dir / "task.toml").open("rb") as fh:
        meta = tomllib.load(fh)

    env = meta.get("environment", {}) or {}
    agent_meta = meta.get("agent", {}) or {}
    verifier_meta = meta.get("verifier", {}) or {}

    docker_image = env.get("docker_image")
    if not docker_image:
        raise ValueError(f"task.toml for {task_id}: missing environment.docker_image")

    return {
        "instance_id": task_id,
        "benchmark": "terminal_bench",
        "task_name": meta.get("task", {}).get("name", task_id),
        "instruction": instruction,
        "docker_image": docker_image,
        # TB tasks all run inside /app per the official environment convention;
        # the per-task Dockerfile sets WORKDIR /app. Override via task.toml
        # in future if a task needs a different root.
        "workspace": env.get("workdir", "/app"),
        "tests_host_path": str(task_dir / "tests"),
        "tests_container_path": "/tests",
        "logs_container_path": "/logs/verifier",
        "verifier_entrypoint": "/tests/test.sh",
        "reward_file": "/logs/verifier/reward.txt",
        "agent_timeout_sec": int(agent_meta.get("timeout_sec", 900)),
        "verifier_timeout_sec": int(verifier_meta.get("timeout_sec", 900)),
        "resources": {
            "cpus": int(env.get("cpus", 1)),
            "memory_mb": int(env.get("memory_mb", 2048)),
            "storage_mb": int(env.get("storage_mb", 10240)),
            "gpus": int(env.get("gpus", 0)),
        },
        # Echo SWE-bench-shaped fields so downstream code that reads them
        # generically (fairness validator, manifest) doesn't NPE.
        "repo": meta.get("task", {}).get("name", task_id),
        "base_commit": "tb2",
        "problem_statement": instruction,
    }


def materialize_dataset(task_ids: List[str], out_path: Path) -> None:
    """Write a JSON file with one entry per task — same shape as SWE-bench's
    verified.json, so the existing driver loader at
    agent_bench/runtime/driver.py:217 picks it up unchanged."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [load_task(t) for t in task_ids]
    with out_path.open("w") as fh:
        json.dump(rows, fh, indent=2)


# CLI: python -m agent_bench.benchmarks.terminal_bench.loader <out.json> [task_id ...]
if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("usage: loader.py <out.json> [task_id ...]", file=sys.stderr)
        sys.exit(2)
    out = Path(sys.argv[1])
    ids = sys.argv[2:] or list_task_ids()
    if not ids:
        print("no tasks found under tasks/", file=sys.stderr)
        sys.exit(1)
    materialize_dataset(ids, out)
    print(f"wrote {len(ids)} tasks -> {out}")
