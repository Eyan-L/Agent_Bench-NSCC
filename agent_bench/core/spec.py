from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from agent_bench.core.profile_io import ProfileLoadError, load_profile


BENCHMARKS = {"swebench", "swebench_pro", "terminal_bench", "deepresearch_bench", "longbench", "request_bench", "swe_atlas_qa"}
HARNESSES = {"openclaw", "openhands", "hermes", "claude_code", "react", "ar"}
AB_SERVING_BACKENDS = {"vllm", "sglang", "litellm", "existing_endpoint", "openai_compatible"}
DEPLOYMENTS = {"client_only", "local_docker", "nscc_pbs", "ccds_cloud"}
STAGE_NAMES = (
    "prepare_environment",
    "materialize_benchmark",
    "prepare_images",
    "start_serving",
    "start_otel",
    "run_benchmark",
    "cleanup",
)
STAGE_CONFIG_KEYS = (*STAGE_NAMES, "rebuild_images")


class ConfigError(ValueError):
    """Raised when an experiment or profile file is invalid."""


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config_path: Path
    benchmark_name: str
    benchmark_profile: Path | None
    harness_name: str
    harness_profile: Path | None
    model_profile: Path
    serving_profile: Path
    deployment_profile: Path
    output_dir: Path
    seeds: tuple[int, ...]
    stages: dict[str, bool]
    limits: dict[str, Any]
    telemetry: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ResolvedExperiment:
    spec: ExperimentSpec
    benchmark: dict[str, Any]
    harness: dict[str, Any]
    model: dict[str, Any]
    serving: dict[str, Any]
    deployment: dict[str, Any]
    profile_paths: dict[str, Path]
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "run_id": self.run_id,
            "config_path": str(self.spec.config_path),
            "benchmark": self.benchmark,
            "harness": self.harness,
            "model": self.model,
            "serving": self.serving,
            "deployment": self.deployment,
            "output_dir": str(self.spec.output_dir),
            "seeds": list(self.spec.seeds),
            "stages": dict(self.spec.stages),
            "limits": self.spec.limits,
            "telemetry": self.spec.telemetry,
            "profiles": {key: str(value) for key, value in self.profile_paths.items()},
        }


def load_resolved_experiment(path: str | Path, *, root: str | Path | None = None) -> ResolvedExperiment:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()
    repo_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    raw = _load_toml(config_path)
    spec = _parse_experiment_spec(raw, config_path=config_path, repo_root=repo_root)

    model_raw = _load_profile(spec.model_profile)
    serving_raw = _load_profile(spec.serving_profile)
    deployment_raw = _load_profile(spec.deployment_profile)

    model = _unwrap(model_raw, "model")
    serving = _unwrap(serving_raw, "serving")
    deployment = _unwrap(deployment_raw, "deployment")

    benchmark = _named_section(spec.benchmark_name, spec.benchmark_profile, "benchmark", fallback_key="application")
    harness = _named_section(spec.harness_name, spec.harness_profile, "harness")

    profile_paths = {
        "model": spec.model_profile,
        "serving": spec.serving_profile,
        "deployment": spec.deployment_profile,
    }
    if spec.benchmark_profile is not None:
        profile_paths["benchmark"] = spec.benchmark_profile
    if spec.harness_profile is not None:
        profile_paths["harness"] = spec.harness_profile

    resolved = ResolvedExperiment(
        spec=spec,
        benchmark=benchmark,
        harness=harness,
        model=model,
        serving=serving,
        deployment=deployment,
        profile_paths=profile_paths,
        run_id="",
    )
    run_id = _stable_run_id(resolved)
    resolved = ResolvedExperiment(
        spec=spec,
        benchmark=benchmark,
        harness=harness,
        model=model,
        serving=serving,
        deployment=deployment,
        profile_paths=profile_paths,
        run_id=run_id,
    )
    validate_resolved_experiment(resolved)
    return resolved


def validate_resolved_experiment(resolved: ResolvedExperiment) -> None:
    errors: list[str] = []
    if resolved.spec.benchmark_name not in BENCHMARKS:
        errors.append(f"benchmark.name must be one of {sorted(BENCHMARKS)}")
    if resolved.spec.harness_name not in HARNESSES:
        errors.append(f"harness.name must be one of {sorted(HARNESSES)}")

    model_id = resolved.model.get("config_id") or resolved.model.get("name")
    if not model_id:
        errors.append("model profile must define model.config_id or model.name")

    backend = resolved.serving.get("backend")
    if backend not in AB_SERVING_BACKENDS:
        errors.append(f"serving.backend must be one of {sorted(AB_SERVING_BACKENDS)}")

    deployment_kind = resolved.deployment.get("kind") or resolved.deployment.get("name")
    if deployment_kind not in DEPLOYMENTS:
        errors.append(f"deployment.kind must be one of {sorted(DEPLOYMENTS)}")

    if deployment_kind == "client_only" and backend in {"vllm", "sglang"}:
        errors.append("client_only deployment requires litellm, existing_endpoint, or openai_compatible serving")

    if errors:
        raise ConfigError("; ".join(errors))


def _parse_experiment_spec(raw: dict[str, Any], *, config_path: Path, repo_root: Path) -> ExperimentSpec:
    name = _required_str(raw, "name")
    benchmark = _required_table(raw, "benchmark")
    harness = _required_table(raw, "harness")
    model = _required_table(raw, "model")
    serving = _required_table(raw, "serving")
    deployment = _required_table(raw, "deployment")

    benchmark_name = _required_str(benchmark, "name", "benchmark.name")
    harness_name = _required_str(harness, "name", "harness.name")
    output_dir = _resolve_path(config_path.parent, raw.get("output_dir", f"runs/{name}"), repo_root)
    seeds = _parse_seeds(raw.get("seeds", [42]))
    limits = _parse_limits(raw.get("limits", {}))
    telemetry = raw.get("telemetry", {})
    if not isinstance(telemetry, dict):
        raise ConfigError("telemetry must be a table when present")
    stages = _parse_stages(raw.get("stages", {}))

    return ExperimentSpec(
        name=name,
        config_path=config_path,
        benchmark_name=benchmark_name,
        benchmark_profile=_optional_profile(config_path, repo_root, benchmark.get("profile")),
        harness_name=harness_name,
        harness_profile=_optional_profile(config_path, repo_root, harness.get("profile")),
        model_profile=_required_profile(config_path, repo_root, model, "model.profile"),
        serving_profile=_required_profile(config_path, repo_root, serving, "serving.profile"),
        deployment_profile=_required_profile(config_path, repo_root, deployment, "deployment.profile"),
        output_dir=output_dir,
        seeds=seeds,
        stages=stages,
        limits=limits,
        telemetry=telemetry,
        raw=raw,
    )


def _named_section(
    name: str,
    profile_path: Path | None,
    wrapper_key: str,
    *,
    fallback_key: str | None = None,
) -> dict[str, Any]:
    if profile_path is None:
        return {"name": name}
    raw = _load_profile(profile_path)
    wrapped = raw.get(wrapper_key)
    if isinstance(wrapped, dict):
        data = dict(wrapped)
    elif fallback_key is not None and isinstance(raw.get(fallback_key), dict):
        data = dict(raw[fallback_key])
    else:
        data = dict(raw)
    if "name" not in data:
        data = {"name": name, **data}
    return data


def _load_profile(path: Path) -> dict[str, Any]:
    try:
        return load_profile(path)
    except ProfileLoadError as exc:
        raise ConfigError(str(exc)) from exc


def _load_toml(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".toml":
        raise ConfigError(f"experiment config must be a TOML file: {path}")
    return _load_profile(path)


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ConfigError(f"expected JSON profile file: {path}")
    return _load_profile(path)


def _unwrap(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if isinstance(value, dict):
        return dict(value)
    return dict(raw)


def _stable_run_id(resolved: ResolvedExperiment) -> str:
    payload = {
        "name": resolved.spec.name,
        "benchmark": resolved.benchmark.get("name") or resolved.spec.benchmark_name,
        "harness": resolved.harness.get("harness") or resolved.harness.get("name") or resolved.spec.harness_name,
        "model": resolved.model.get("config_id") or resolved.model.get("name"),
        "serving": resolved.serving.get("config_id") or resolved.serving.get("backend"),
        "deployment": resolved.deployment.get("kind") or resolved.deployment.get("name"),
        "seeds": list(resolved.spec.seeds),
        "limits": resolved.spec.limits,
    }
    if resolved.spec.stages:
        payload["stages"] = resolved.spec.stages
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _required_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a table")
    return value


def _required_str(raw: dict[str, Any], key: str, label: str | None = None) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label or key} must be a non-empty string")
    return value


def _parse_seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("seeds must be a non-empty list of integers")
    seeds: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ConfigError("seeds must be a non-empty list of integers")
        seeds.append(item)
    return tuple(seeds)


def _parse_limits(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("limits must be a table when present")

    allowed = {"max_workers", "task_container_cpus"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(
            f"limits contains unknown key(s): {', '.join(unknown)}; "
            f"expected one of {', '.join(sorted(allowed))}"
        )

    limits = dict(value)
    if "max_workers" in limits:
        item = limits["max_workers"]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ConfigError("limits.max_workers must be a positive integer")
    if "task_container_cpus" in limits:
        item = limits["task_container_cpus"]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
            raise ConfigError("limits.task_container_cpus must be a positive number")
        limits["task_container_cpus"] = float(item)
    return limits


def _parse_stages(value: Any) -> dict[str, bool]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("stages must be a table when present")
    unknown = sorted(set(value) - set(STAGE_CONFIG_KEYS))
    if unknown:
        raise ConfigError(
            f"stages contains unknown stage(s): {', '.join(unknown)}; "
            f"expected one of {', '.join(STAGE_CONFIG_KEYS)}"
        )
    stages: dict[str, bool] = {}
    for key, item in value.items():
        if not isinstance(item, bool):
            raise ConfigError(f"stages.{key} must be a boolean")
        stages[key] = item
    return stages


def _required_profile(config_path: Path, repo_root: Path, section: dict[str, Any], label: str) -> Path:
    value = section.get("profile")
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty path string")
    return _resolve_path(config_path.parent, value, repo_root)


def _optional_profile(config_path: Path, repo_root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError("profile must be a non-empty path string when present")
    return _resolve_path(config_path.parent, value, repo_root)


def _resolve_path(base_dir: Path, value: Any, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError("path values must be non-empty strings")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_config = (base_dir / path).resolve()
    if from_config.exists():
        return from_config
    return (repo_root / path).resolve()
