from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from agent_bench.benchmarks.swebench_pro.common import (
    DEFAULT_DATASET,
    DEFAULT_DOCKERHUB_USERNAME,
    DEFAULT_SPLIT,
    normalize_instance,
)
from agent_bench.core.spec import ConfigError


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
    return Path("data/swebench_pro") / f"{safe}_instances.json"


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
        raise ConfigError("SWE-bench Pro benchmark profile requires a non-empty instances list")
    wanted_set = set(wanted)

    dockerhub_username = str(
        benchmark.get("dockerhub_username")
        or benchmark.get("dockerhub")
        or DEFAULT_DOCKERHUB_USERNAME
    )
    selected = [
        normalize_instance(row, dockerhub_username=dockerhub_username)
        for row in rows
        if str(row.get("instance_id")) in wanted_set
    ]
    missing = wanted_set - {row["instance_id"] for row in selected}
    if missing:
        raise ConfigError(f"SWE-bench Pro dataset is missing requested instances: {sorted(missing)}")

    order = {instance_id: idx for idx, instance_id in enumerate(wanted)}
    selected.sort(key=lambda row: order[row["instance_id"]])
    selected = attach_official_resources(
        selected,
        resource_root=resolve_resource_root(benchmark),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return out.resolve()


def attach_official_resources(
    rows: list[dict[str, Any]],
    *,
    resource_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Embed official SWE-bench Pro evaluation resources for selected rows.

    The public Hugging Face dataset carries the task metadata, but the official
    evaluator keeps per-instance run scripts, parsers, and Dockerfile ENV lines
    in the SWE-bench_Pro-os repository. Embedding only the selected resources
    keeps the materialized JSON self-contained and avoids requiring a parallel
    run_scripts/ tree in every run directory.
    """
    if not rows:
        return rows
    root = resource_root or resolve_resource_root({})
    return [_attach_one_resource(row, root) for row in rows]


def resolve_resource_root(benchmark: Mapping[str, Any]) -> Path:
    explicit = (
        os.environ.get("AB_SWEBENCH_PRO_ROOT")
        or benchmark.get("swebench_pro_root")
        or benchmark.get("resource_root")
        or benchmark.get("resources_root")
    )
    if explicit:
        root = Path(str(explicit)).expanduser().resolve()
        _validate_resource_root(root)
        return root

    tmp_root = Path("/tmp/swebench_pro_os")
    if _is_resource_root(tmp_root):
        return tmp_root

    cache_root = Path(
        os.environ.get("AB_SWEBENCH_PRO_RESOURCE_CACHE")
        or "data/swebench_pro/resources/SWE-bench_Pro-os"
    ).expanduser()
    if not cache_root.is_absolute():
        cache_root = (Path.cwd() / cache_root).resolve()
    if not _is_resource_root(cache_root):
        _clone_resource_repo(cache_root)
    _validate_resource_root(cache_root)
    return cache_root


def _attach_one_resource(row: dict[str, Any], root: Path) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    data = dict(row)
    _set_if_blank(data, "run_script", _read_required(root / "run_scripts" / instance_id / "run_script.sh"))
    _set_if_blank(data, "parsing_script", _read_required(root / "run_scripts" / instance_id / "parser.py"))
    base = root / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile"
    instance = root / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile"
    if base.is_file():
        _set_if_blank(data, "base_dockerfile", base.read_text(encoding="utf-8"))
    if instance.is_file():
        _set_if_blank(data, "instance_dockerfile", instance.read_text(encoding="utf-8"))
    return data


def _set_if_blank(data: dict[str, Any], key: str, value: str) -> None:
    current = data.get(key)
    if not isinstance(current, str) or not current.strip():
        data[key] = value


def _read_required(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"SWE-bench Pro resource file missing: {path}")
    return path.read_text(encoding="utf-8")


def _validate_resource_root(root: Path) -> None:
    if not _is_resource_root(root):
        raise ConfigError(
            f"SWE-bench Pro resource root is missing run_scripts/dockerfiles: {root}"
        )


def _is_resource_root(root: Path) -> bool:
    return (root / "run_scripts").is_dir() and (root / "dockerfiles").is_dir()


def _clone_resource_repo(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/scaleapi/SWE-bench_Pro-os.git",
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise ConfigError(
            "failed to clone SWE-bench Pro official resources; "
            "set AB_SWEBENCH_PRO_ROOT to a local SWE-bench_Pro-os checkout. "
            f"git stderr: {(proc.stderr or '')[-500:]}"
        )


def load_instances(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        row["instance_id"]: row
        for row in (normalize_instance(item) for item in load_rows(Path(dataset_path)))
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ConfigError(f"SWE-bench Pro instance data file not found: {path}")
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
    raise ConfigError(f"unsupported SWE-bench Pro instance data shape: {path}")


def _load_hf_rows(*, dataset_name: str, split: str) -> list[dict[str, Any]]:
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ConfigError("SWE-bench Pro benchmark profile dataset must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ConfigError("SWE-bench Pro benchmark profile split must be a non-empty string")
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ConfigError(
            "SWE-bench Pro materialization requires the `datasets` package "
            "or benchmark_instance_data/AB_BENCHMARK_INSTANCE_DATA"
        ) from exc
    return [dict(row) for row in load_dataset(dataset_name, split=split)]
