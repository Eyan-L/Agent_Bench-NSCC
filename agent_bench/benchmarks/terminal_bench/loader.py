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
            ├── test.sh                # verifier entrypoint, injected at /tests
            └── test_outputs.py        # pytest cases, injected at /tests

The driver expects `instance_data["instance_id"]`. We use the task directory
name (e.g. "kv-store-grpc"). All other fields are TB-specific and ignored by
the SWE-bench code path.
"""
from __future__ import annotations

import json
import os
import posixpath
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _tasks_root() -> Path:
    """Vendored TB task tree at `agent_bench/benchmarks/terminal_bench/tasks/`.
    Override via AB_BENCHMARK_TASKS_ROOT."""
    env = os.environ.get("AB_BENCHMARK_TASKS_ROOT")
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


def _normalize_workdir(path: str, current: str = "/") -> str:
    """Normalize a Docker WORKDIR operand using Docker's relative-path rule."""
    raw = (path or "").strip()
    if not raw:
        return current
    if not raw.startswith("/"):
        raw = f"{current.rstrip('/')}/{raw}"
    normalized = posixpath.normpath(raw)
    return normalized if normalized.startswith("/") else f"/{normalized}"


def _strip_dockerfile_comment(line: str) -> str:
    """Strip comments while respecting Dockerfile quoting well enough for WORKDIR."""
    quote: Optional[str] = None
    escaped = False
    out: List[str] = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def _dockerfile_workdir(dockerfile: Path) -> Optional[str]:
    """Return the final WORKDIR from a task Dockerfile, if one is declared."""
    if not dockerfile.is_file():
        return None
    workdir = "/"
    found = False
    try:
        lines = dockerfile.read_text().splitlines()
    except OSError:
        return None
    for raw_line in lines:
        line = _strip_dockerfile_comment(raw_line)
        if not line:
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = line.split()
        if len(parts) < 2 or parts[0].upper() != "WORKDIR":
            continue
        workdir = _normalize_workdir(parts[1], workdir)
        found = True
    return workdir if found else None


def _task_workspace(task_dir: Path, env: Dict) -> Tuple[str, str]:
    """Pick the task workspace and record why for runtime compatibility."""
    if env.get("workdir"):
        return _normalize_workdir(str(env["workdir"])), "task.toml"
    dockerfile_workdir = _dockerfile_workdir(task_dir / "environment" / "Dockerfile")
    if dockerfile_workdir:
        return dockerfile_workdir, "dockerfile"
    return "/app", "default"


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

    workspace, workspace_source = _task_workspace(task_dir, env)

    return {
        "instance_id": task_id,
        "benchmark": "terminal_bench",
        "task_name": meta.get("task", {}).get("name", task_id),
        "instruction": instruction,
        "docker_image": docker_image,
        "workspace": workspace,
        "workspace_source": workspace_source,
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
