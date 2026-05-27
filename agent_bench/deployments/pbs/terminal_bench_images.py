from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from agent_bench.benchmarks.terminal_bench import loader
from agent_bench.benchmarks.terminal_bench.app import TerminalBenchSingularityHarness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull Terminal-Bench task images as Singularity SIFs.")
    parser.add_argument("--sif-dir", required=True, help="Directory where task SIF files are cached.")
    parser.add_argument("--task-id", action="append", default=[], help="Task id to pull. Defaults to all tasks.")
    args = parser.parse_args(argv)

    task_ids = list(args.task_id) or loader.list_task_ids()
    if not task_ids:
        parser.error("no Terminal-Bench tasks found")

    os.environ["OC_TB_SIF_DIR"] = str(Path(args.sif_dir).expanduser())
    harness = TerminalBenchSingularityHarness()
    for task_id in task_ids:
        instance = loader.load_task(task_id)
        sif = harness._ensure_sif(instance["docker_image"])
        print(f"{task_id}: {instance['docker_image']} -> {sif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
