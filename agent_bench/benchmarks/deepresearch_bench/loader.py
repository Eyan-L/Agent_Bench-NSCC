"""DeepResearch-Bench query loader.

Maps the vendored ``data/prompt_data/query.jsonl`` (100 PhD-level prompts,
50 EN + 50 ZH, 22 domains) into the same ``instance_data`` dict shape the
agent_bench/runtime driver consumes, with ``benchmark="deepresearch_bench"``
as the discriminator so harness + driver dispatch can branch.

Vendored data lives at::

    agent_bench/benchmarks/deepresearch_bench/eval/vendor/
    └── data/prompt_data/query.jsonl
        each line: {"id": int, "topic": str, "language": "en"|"zh", "prompt": str}

The driver expects ``instance_data["instance_id"]``. We surface
``"dr_<zero-padded-id>"`` so the ID survives shell-arg / filesystem use.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List


_VENDOR_QUERY = (
    Path(__file__).resolve().parent
    / "eval" / "vendor" / "data" / "prompt_data" / "query.jsonl"
)


def _query_path() -> Path:
    """Vendored upstream query file. Override via AB_BENCHMARK_QUERY_FILE for testing."""
    env = os.environ.get("AB_BENCHMARK_QUERY_FILE")
    if env:
        return Path(env)
    return _VENDOR_QUERY


def _format_instance_id(numeric_id: int) -> str:
    """Stable string id: ``dr_001`` .. ``dr_100``."""
    return f"dr_{int(numeric_id):03d}"


def list_task_ids() -> List[str]:
    """Every row's ``dr_NNN`` id, in upstream file order."""
    out: List[str] = []
    with _query_path().open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(_format_instance_id(row["id"]))
    return out


def load_task(task_id: str) -> Dict:
    """Build the ``instance_data`` dict for one DR-Bench task."""
    if not task_id.startswith("dr_"):
        raise ValueError(f"DR-Bench task_id must start with 'dr_', got {task_id!r}")
    try:
        numeric = int(task_id[3:])
    except ValueError as e:
        raise ValueError(f"can't parse numeric id from {task_id!r}") from e

    found = None
    with _query_path().open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("id", -1)) == numeric:
                found = row
                break
    if found is None:
        raise FileNotFoundError(f"DR-Bench task not found in query.jsonl: {task_id}")

    prompt = found["prompt"]
    return {
        "instance_id": task_id,
        "benchmark": "deepresearch_bench",
        "task_name": f"dr_{numeric:03d}",
        # Agent input — written into the sandbox at /workspace/PROMPT.md
        # by the harness's prepare_environment step.
        "prompt": prompt,
        "language": found.get("language", "en"),
        "topic": found.get("topic", "Other"),
        # The DR-Bench sandbox is a single shared lean image, not per-task.
        # Hard-coded here so per-instance config doesn't have to repeat it;
        # override per run via the harness JSON if needed.
        # Published on Docker Hub — `docker pull hongrui0/dr-sandbox:1.0`.
        "docker_image": "hongrui0/dr-sandbox:1.0",
        "workspace": "/workspace",
        # The artifact the harness collects after the agent says DONE.
        "report_file": "/workspace/report.md",
        # Generous default; agent loops doing 10+ search rounds can take a
        # while on smaller hosted models. Override per-condition.
        "agent_timeout_sec": int(os.environ.get("AB_BENCHMARK_AGENT_TIMEOUT_SEC", "3600")),
        # SWE-bench shaped echoes — keeps generic readers (fairness check,
        # manifest emitter) from NPE-ing.
        "repo": "deepresearch_bench",
        "base_commit": "dr-bench-v1",
        "problem_statement": prompt,
    }


def materialize_dataset(task_ids: List[str], out_path: Path) -> None:
    """Write a JSON file with one entry per task. Driver loader at
    ``agent_bench/runtime/driver.py`` reads this shape unchanged."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [load_task(t) for t in task_ids]
    with out_path.open("w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


# CLI: python -m agent_bench.benchmarks.deepresearch_bench.loader <out.json> [task_id ...]
if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("usage: loader.py <out.json> [task_id ...]", file=sys.stderr)
        sys.exit(2)
    out = Path(sys.argv[1])
    ids = sys.argv[2:] or list_task_ids()
    if not ids:
        print("no tasks found in query.jsonl", file=sys.stderr)
        sys.exit(1)
    materialize_dataset(ids, out)
    print(f"wrote {len(ids)} tasks -> {out}")
