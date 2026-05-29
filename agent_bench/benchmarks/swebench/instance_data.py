from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_bench.core.spec import ConfigError


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"


def resolve_profile_instance_data(resolved: Any) -> Path | None:
    value = resolved.benchmark.get("instance_data")
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


def materialize_profile_dataset(resolved: Any, out: Path) -> Path:
    benchmark = resolved.benchmark
    dataset_name = benchmark.get("dataset") or benchmark.get("dataset_name")
    split = benchmark.get("split", DEFAULT_SPLIT)
    instances = benchmark.get("instances") or []

    if not isinstance(dataset_name, str) or not dataset_name:
        raise ConfigError(
            "SWE-bench experiments require instance_data in the experiment TOML, "
            "INSTANCE_DATA in the environment, or benchmark profile dataset/split fields"
        )
    if not isinstance(split, str) or not split:
        raise ConfigError("SWE-bench benchmark profile split must be a non-empty string")
    if not instances:
        raise ConfigError("SWE-bench benchmark profile requires a non-empty instances list")

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise ConfigError(
            "SWE-bench dataset materialization requires the `datasets` package "
            "or an explicit INSTANCE_DATA/instance_data path"
        ) from exc

    wanted = {str(instance_id) for instance_id in instances}
    selected = [dict(row) for row in load_dataset(dataset_name, split=split) if row.get("instance_id") in wanted]
    missing = wanted - {row.get("instance_id") for row in selected}
    if missing:
        raise ConfigError(
            f"SWE-bench dataset {dataset_name}:{split} is missing requested instances: {sorted(missing)}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return out.resolve()


def default_materialized_path(resolved: Any) -> Path:
    sweep_name = resolved.benchmark.get("sweep_name") or resolved.spec.name
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(sweep_name))
    return Path("data/swebench") / f"{safe}_instances.json"
