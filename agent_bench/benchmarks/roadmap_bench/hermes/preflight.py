"""Verify uploaded Hermes/Roadmap task SIFs without pulling on GPU nodes."""
from __future__ import annotations

import argparse
from pathlib import Path

from agent_bench.benchmarks.terminal_bench.loader import load_task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-sif", type=Path)
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--sif-dir", type=Path)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    if args.runtime_sif:
        if not args.runtime_sif.is_file() or args.runtime_sif.stat().st_size == 0:
            parser.error(f"Upload the patched Hermes runtime SIF before submitting: {args.runtime_sif}")
        print(f"Prepared Hermes runtime SIF: {args.runtime_sif}")
        return 0
    if not args.tasks_root or not args.sif_dir or not args.task_id:
        parser.error("Task preflight requires --tasks-root, --sif-dir and --task-id")
    missing = []
    for task_id in args.task_id:
        row = load_task(task_id, tasks_root=args.tasks_root)
        image = row["docker_image"]
        sif = args.sif_dir / (image.replace("/", "__").replace(":", "__") + ".sif")
        if not sif.is_file() or sif.stat().st_size == 0:
            missing.append(str(sif))
    if missing:
        parser.error("Prepare these task SIFs before submitting:\n" + "\n".join(missing))
    print(f"Prepared Roadmap task SIFs: {len(args.task_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
