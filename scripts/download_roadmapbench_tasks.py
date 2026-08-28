#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import requests


DEFAULT_REPO = "UnipatAI/RoadmapBench"
DEFAULT_REVISION = "59184e779909300a5a0150b06b945d39da81a099"


def _tree_entries(repo, revision, task_id):
    path = quote(task_id, safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{repo}/tree/"
        f"{revision}/{path}"
    )
    params = {"recursive": "true", "expand": "false", "limit": 1000}

    while url:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        yield from response.json()
        url = response.links.get("next", {}).get("url")
        params = None


def _download_file(repo, revision, repo_path, destination):
    path = quote(repo_path, safe="/")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")

    with requests.get(url, params={"download": "true"}, stream=True,
                      timeout=300) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(
        description="Download runnable RoadmapBench task definitions."
    )
    parser.add_argument("task_ids", nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--include-source-tree",
        action="store_true",
        help="Also download environment/repo, which is already in the task SIF.",
    )
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="Also download the oracle solution (not needed for evaluation).",
    )
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo": args.repo,
        "revision": args.revision,
        "tasks": {},
        "excluded_prefixes": [],
    }
    if not args.include_source_tree:
        manifest["excluded_prefixes"].append("environment/repo/")
    if not args.include_solution:
        manifest["excluded_prefixes"].append("solution/")

    for task_id in args.task_ids:
        files = []
        prefix = task_id + "/"
        for entry in _tree_entries(args.repo, args.revision, task_id):
            if entry.get("type") != "file":
                continue
            repo_path = entry["path"]
            if not repo_path.startswith(prefix):
                continue
            relative = repo_path[len(prefix):]
            if not args.include_source_tree and relative.startswith(
                "environment/repo/"
            ):
                continue
            if not args.include_solution and relative.startswith("solution/"):
                continue
            files.append((repo_path, relative, entry.get("size")))

        required = {"task.toml", "instruction.md", "environment/Dockerfile"}
        found = {relative for _, relative, _ in files}
        missing = sorted(required - found)
        if missing:
            raise SystemExit(f"{task_id}: missing required files: {missing}")
        if not any(path.startswith("tests/") for path in found):
            raise SystemExit(f"{task_id}: no tests found")

        print(f"{task_id}: downloading {len(files)} files", flush=True)
        for repo_path, relative, _size in files:
            destination = root / task_id / relative
            _download_file(args.repo, args.revision, repo_path, destination)

        manifest["tasks"][task_id] = {
            "file_count": len(files),
            "files": [
                {"path": relative, "size": size}
                for _, relative, size in files
            ],
        }

    manifest_path = root / "_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
