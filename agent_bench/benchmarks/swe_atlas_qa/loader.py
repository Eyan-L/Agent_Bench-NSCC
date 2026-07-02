from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from agent_bench.benchmarks.swe_atlas_qa.common import (
    DEFAULT_DATASET,
    DEFAULT_DOCKERHUB_USERNAME,
    DEFAULT_SPLIT,
)
from agent_bench.core.spec import ConfigError


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ConfigError(f"SWE-Atlas-QnA instance data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, Mapping)]
    if isinstance(data, dict) and isinstance(data.get("instances"), list):
        return [dict(row) for row in data["instances"] if isinstance(row, Mapping)]
    if isinstance(data, dict):
        return [dict(row) for row in data.values() if isinstance(row, Mapping)]
    raise ConfigError(f"unsupported SWE-Atlas-QnA instance data shape: {path}")


def _load_hf_rows(*, dataset_name: str, split: str) -> list[dict[str, Any]]:
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ConfigError("SWE-Atlas-QnA benchmark profile dataset must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ConfigError("SWE-Atlas-QnA benchmark profile split must be a non-empty string")
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ConfigError(
            "SWE-Atlas-QnA materialization requires the `datasets` package "
            "or benchmark_instance_data/AB_BENCHMARK_INSTANCE_DATA"
        ) from exc
    return [dict(row) for row in load_dataset(dataset_name, split=split)]


def resolve_profile_instance_data(resolved: Any) -> Path | None:
    value = resolved.benchmark.get("benchmark_instance_data")
    if not isinstance(value, str) or not value:
        return None
    benchmark_path = resolved.profile_paths.get("benchmark")
    base_dir = benchmark_path.parent if benchmark_path is not None else Path.cwd()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_profile = (base_dir / path).resolve()
    if from_profile.exists():
        return from_profile
    return (Path.cwd() / path).resolve()


def default_materialized_path(resolved: Any) -> Path:
    sweep_name = resolved.benchmark.get("sweep_name") or resolved.spec.name
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(sweep_name))
    return Path("data/swe_atlas_qa") / f"{safe}_instances.json"


def normalize_instance(
    row: Mapping[str, Any],
    *,
    dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME,
) -> dict[str, Any]:
    data = dict(row)
    instance_id = str(data.get("instance_id") or data.get("task_id", ""))
    repo_url = str(data.get("repository_url", ""))
    if repo_url and not repo_url.startswith("http") and not repo_url.startswith("git"):
        repo_url = f"https://github.com/{repo_url}"
    base_commit = str(data.get("repository_base_commit", ""))
    docker_image = data.get("docker_image") or data.get("container_image")

    # Normalize rubric
    rubric_val = data.get("rubric")
    normalized_rubric = {}
    if isinstance(rubric_val, str):
        try:
            import json
            rubric_list = json.loads(rubric_val)
        except Exception:
            rubric_list = rubric_val
    else:
        rubric_list = rubric_val

    if isinstance(rubric_list, list):
        positives = []
        negatives = []
        for item in rubric_list:
            if isinstance(item, dict):
                title = item.get("title", "")
                ann = item.get("annotations") or {}
                ann_type = ann.get("type", "")
                if "positive" in ann_type:
                    positives.append(title)
                elif "negative" in ann_type:
                    negatives.append(title)
        normalized_rubric = {
            "positive_verifiers": positives,
            "negative_verifiers": negatives,
        }
    elif isinstance(rubric_list, dict):
        normalized_rubric = rubric_list

    data.update({
        "benchmark": "swe_atlas_qa",
        "instance_id": instance_id,
        "repo": repo_url,
        "repo_id": repo_url,
        "base_commit": base_commit,
        "problem_statement": data.get("question", "") or data.get("prompt", ""),
        "docker_image": str(docker_image),
        "container_image": str(docker_image),
        "workspace_root": "/app",
        "rubric": normalized_rubric,
    })
    return data


def load_instances(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        row["instance_id"]: row
        for row in (normalize_instance(item) for item in load_rows(Path(dataset_path)))
    }


def materialize_profile_dataset(resolved: Any, out: Path) -> Path:
    benchmark = resolved.benchmark
    profile_path = resolve_profile_instance_data(resolved)
    if profile_path is not None:
        rows = load_rows(profile_path)
    else:
        rows = _load_hf_rows(
            dataset_name=benchmark.get("dataset") or benchmark.get("dataset_name") or DEFAULT_DATASET,
            split=benchmark.get("split", DEFAULT_SPLIT),
        )

    wanted = [str(item) for item in (benchmark.get("instances") or [])]
    if not wanted:
        raise ConfigError("SWE-Atlas-QnA benchmark profile requires a non-empty instances list")
    wanted_set = set(wanted)

    dockerhub_username = str(
        benchmark.get("dockerhub_username")
        or benchmark.get("dockerhub")
        or DEFAULT_DOCKERHUB_USERNAME
    )
    selected = [
        normalize_instance(row, dockerhub_username=dockerhub_username)
        for row in rows
        if str(row.get("instance_id") or row.get("task_id") or "") in wanted_set
    ]
    missing = wanted_set - {row["instance_id"] for row in selected}
    if missing:
        raise ConfigError(f"SWE-Atlas-QnA dataset is missing requested instances: {sorted(missing)}")

    order = {instance_id: idx for idx, instance_id in enumerate(wanted)}
    selected.sort(key=lambda row: order[row["instance_id"]])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return out.resolve()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: loader.py <app.json> <out.json>", file=sys.stderr)
        sys.exit(2)
    app_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    class FakeResolved:
        def __init__(self, path):
            self.benchmark = json.load(open(path))
            self.profile_paths = {"benchmark": Path(path)}

    resolved = FakeResolved(app_path)
    materialize_profile_dataset(resolved, out_path)
    print(f"wrote materialized tasks -> {out_path}")