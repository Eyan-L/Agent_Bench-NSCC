"""RoadmapBench task materialization using Terminal-Bench task definitions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from agent_bench.benchmarks.terminal_bench import loader as terminal_loader


def list_task_ids(tasks_root: str | Path | None = None) -> List[str]:
    return terminal_loader.list_task_ids(tasks_root=tasks_root)


def load_task(task_id: str, tasks_root: str | Path | None = None) -> Dict:
    return terminal_loader.load_task(
        task_id,
        tasks_root=tasks_root,
        benchmark="roadmap_bench",
        benchmark_variant="roadmapbench",
    )


def materialize_dataset(
    task_ids: List[str],
    out_path: Path,
    tasks_root: str | Path | None = None,
) -> None:
    terminal_loader.materialize_dataset(
        task_ids,
        out_path,
        tasks_root=tasks_root,
        benchmark="roadmap_bench",
        benchmark_variant="roadmapbench",
    )
