#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
from pathlib import Path

from agent_bench.benchmarks.swe_atlas_qa.loader import materialize_profile_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SWE-Atlas-QA task images required by a benchmark profile."
    )
    parser.add_argument(
        "profile",
        help="Benchmark profile, e.g. profiles/benchmarks/swe_atlas_qa/subset10.json",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only print missing images.",
    )
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Print machine-readable: <container_image> TAB <sif_path>",
    )
    args = parser.parse_args()

    profile = Path(args.profile).resolve()
    if not profile.is_file():
        raise SystemExit(f"profile not found: {profile}")

    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit("SCRATCH is not set")

    sif_dir = Path(
        os.environ.get(
            "AB_DEPLOYMENT_TASK_IMAGE_DIR",
            str(Path(scratch) / "task-images"),
        )
    )

    benchmark = json.loads(profile.read_text())

    class FakeResolved:
        pass

    resolved = FakeResolved()
    resolved.benchmark = benchmark
    resolved.profile_paths = {"benchmark": profile}

    with tempfile.TemporaryDirectory(prefix="atlas-image-inventory-") as td:
        materialized = Path(td) / "instances.json"
        materialize_profile_dataset(resolved, materialized)
        rows = json.loads(materialized.read_text())

    # Preserve first-seen order while deduplicating images.
    unique_images = []
    seen = set()

    for row in rows:
        image = row.get("container_image") or row.get("docker_image")
        if not image or image in seen:
            continue
        seen.add(image)
        unique_images.append(image)

    records = []
    for image in unique_images:
        # Must match SWE-Atlas Singularity resolver semantics.
        tag = image.rsplit(":", 1)[-1]
        sif = sif_dir / f"{tag}.sif"
        exists = sif.is_file()
        records.append((image, sif, exists))

    if args.missing_only:
        records = [r for r in records if not r[2]]

    if args.tsv:
        for image, sif, _ in records:
            print(f"{image}\t{sif}")
        return

    print(f"profile:       {profile}")
    print(f"instances:     {len(rows)}")
    print(f"unique images: {len(unique_images)}")
    print(f"sif dir:       {sif_dir}")
    print()

    for i, (image, sif, exists) in enumerate(records, 1):
        state = "OK" if exists else "MISSING"
        print(f"{i:2d}  {state:7s}  {image}")
        print(f"             -> {sif}")

    total = len(unique_images)
    existing = sum(
        1
        for image in unique_images
        if (sif_dir / f"{image.rsplit(':', 1)[-1]}.sif").is_file()
    )

    print()
    print(f"summary: {existing}/{total} ready, {total-existing}/{total} missing")


if __name__ == "__main__":
    main()
