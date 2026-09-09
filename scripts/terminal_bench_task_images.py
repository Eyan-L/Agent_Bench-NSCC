#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from agent_bench.benchmarks.terminal_bench import loader


def sif_name_for(docker_image: str) -> str:
    # Must exactly match TerminalBenchSingularityHarness._sif_name_for().
    return docker_image.replace("/", "__").replace(":", "__") + ".sif"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Terminal-Bench task images required by a tasks tree."
    )
    parser.add_argument(
        "--tasks-root",
        help=(
            "Terminal-Bench tasks directory. Defaults to "
            "AB_BENCHMARK_TASKS_ROOT."
        ),
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only print missing images.",
    )
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Print machine-readable: <docker_image> TAB <sif_path>",
    )
    args = parser.parse_args()

    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit("SCRATCH is not set")

    tasks_root_raw = (
        args.tasks_root
        or os.environ.get("AB_BENCHMARK_TASKS_ROOT")
    )
    if not tasks_root_raw:
        raise SystemExit(
            "tasks root is not set; use --tasks-root or "
            "AB_BENCHMARK_TASKS_ROOT"
        )

    tasks_root = Path(
        os.path.expandvars(os.path.expanduser(tasks_root_raw))
    ).resolve()

    if not tasks_root.is_dir():
        raise SystemExit(f"tasks root not found: {tasks_root}")

    sif_dir = Path(
        os.environ.get(
            "AB_DEPLOYMENT_TASK_IMAGE_DIR",
            str(Path(scratch) / "task-images"),
        )
    ).resolve()

    task_ids = loader.list_task_ids(tasks_root=tasks_root)
    if not task_ids:
        raise SystemExit(f"no Terminal-Bench tasks found under {tasks_root}")

    # image -> tasks that use it
    image_tasks = {}

    for task_id in task_ids:
        instance = loader.load_task(
            task_id,
            tasks_root=tasks_root,
        )
        image = instance["docker_image"]
        image_tasks.setdefault(image, []).append(task_id)

    records = []

    # Preserve first-seen task/image order.
    for image, users in image_tasks.items():
        sif = sif_dir / sif_name_for(image)
        exists = sif.is_file() and sif.stat().st_size > 0
        records.append((image, sif, exists, users))

    if args.missing_only:
        records = [r for r in records if not r[2]]

    if args.tsv:
        for image, sif, _exists, _users in records:
            print(f"{image}\t{sif}")
        return

    print(f"tasks root:    {tasks_root}")
    print(f"tasks:         {len(task_ids)}")
    print(f"unique images: {len(image_tasks)}")
    print(f"sif dir:       {sif_dir}")
    print()

    for i, (image, sif, exists, users) in enumerate(records, 1):
        state = "OK" if exists else "MISSING"

        print(f"{i:3d}  {state:7s}  {image}")
        print(f"              -> {sif}")

        if len(users) <= 3:
            print(f"              tasks: {', '.join(users)}")
        else:
            print(
                f"              tasks: "
                f"{', '.join(users[:3])} ... "
                f"({len(users)} total)"
            )

    total = len(image_tasks)
    ready = sum(
        1
        for image in image_tasks
        if (
            sif_dir / sif_name_for(image)
        ).is_file()
        and (
            sif_dir / sif_name_for(image)
        ).stat().st_size > 0
    )

    print()
    print(
        f"summary: {ready}/{total} ready, "
        f"{total-ready}/{total} missing"
    )


if __name__ == "__main__":
    main()
