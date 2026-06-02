"""Configuration schema for experiment runs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json
from pathlib import Path
import os
import shlex
from urllib.parse import urlunparse

from agent_bench.core.profile_io import load_profile, resolve_profile_ref


@dataclass
class ModelConfig:
    """LLM identity and model-level limits."""
    config_id: str
    name: str
    endpoint: str  # OpenAI-compatible base URL from the active serving config.
    temperature: float = 0.0
    max_tokens: int = 8192
    context_window: int = 0
    provider: str = "custom"
    api_key: str = "EMPTY"
    litellm_model: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "provider": self.provider,
            "api_key": self.api_key,
            "litellm_model": self.litellm_model,
            "tool_call_parser": self.tool_call_parser,
            "reasoning_parser": self.reasoning_parser,
        }


@dataclass
class ServingConfig:
    """Concrete serving deployment for a model."""
    config_id: str
    backend: str  # vllm | sglang | litellm
    endpoint: str
    bind_host: str
    client_host: str
    port: int
    gpu_mem_util: float
    tensor_parallel_size: Optional[int]
    cuda_visible_devices: Optional[str]
    max_num_seqs: int
    enable_prefix_caching: bool
    preset: str = "default"
    serving_mode: Optional[str] = None
    image: Optional[str] = None
    sif: Optional[str] = None
    provider_name: Optional[str] = None
    provider_base_url: Optional[str] = None
    provider_api_key_env: Optional[str] = None
    provider_model: Optional[str] = None
    proxy_api_key: Optional[str] = None
    proxy_agent_provider: Optional[str] = None
    additional_flags: Dict[str, Any] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "backend": self.backend,
            "endpoint": self.endpoint,
            "bind_host": self.bind_host,
            "client_host": self.client_host,
            "port": self.port,
            "gpu_mem_util": self.gpu_mem_util,
            "tensor_parallel_size": self.tensor_parallel_size,
            "cuda_visible_devices": self.cuda_visible_devices,
            "preset": self.preset,
            "serving_mode": self.serving_mode,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "image": self.image,
            "sif": self.sif,
            "provider_name": self.provider_name,
            "provider_base_url": self.provider_base_url,
            "provider_api_key_env": self.provider_api_key_env,
            "provider_model": self.provider_model,
            "proxy_api_key": self.proxy_api_key,
            "proxy_agent_provider": self.proxy_agent_provider,
            "additional_flags": self.additional_flags,
            "extra_args": self.extra_args,
        }


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    max_steps: int = 500
    max_time_seconds: int = 1800
    retry_limit: int = 10
    tool_profile: str = "code-agent-minimal"
    allowed_tools: List[str] = field(default_factory=lambda: [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep"
    ])
    
    def to_dict(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "max_time_seconds": self.max_time_seconds,
            "retry_limit": self.retry_limit,
            "tool_profile": self.tool_profile,
            "allowed_tools": self.allowed_tools,
        }


@dataclass
class ConditionConfig:
    """Configuration for one experimental condition."""
    condition_id: str
    condition_type: str  # ar_issue_only, ar_static_context, openclaw_agent, ablation
    description: str
    model: ModelConfig
    agent: Optional[AgentConfig] = None
    ablation_flags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        d = {
            "condition_id": self.condition_id,
            "condition_type": self.condition_type,
            "description": self.description,
            "model": self.model.to_dict(),
            "ablation_flags": self.ablation_flags,
        }
        if self.agent:
            d["agent"] = self.agent.to_dict()
        return d


@dataclass
class RunConfig:
    """Configuration for a single run."""
    condition: ConditionConfig
    instance_id: str
    seed: int
    serving: ServingConfig
    plan_version: str
    
    repo_id: str
    base_commit: str
    container_image: str
    wall_clock_budget_seconds: int
    
    workspace_root: str = "/testbed"
    output_dir: Path = Path("data/experiments/_manual/runs/_unknown")
    
    def compute_run_id(self) -> str:
        """Deterministic run ID from hash of key parameters."""
        hash_input = {
            "condition_id": self.condition.condition_id,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "config_id": self.serving.config_id,
            "model_name": self.condition.model.name,
            "plan_version": self.plan_version,
        }
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.compute_run_id(),
            "condition": self.condition.to_dict(),
            "instance_id": self.instance_id,
            "seed": self.seed,
            "serving": self.serving.to_dict(),
            "plan_version": self.plan_version,
            "repo_id": self.repo_id,
            "base_commit": self.base_commit,
            "container_image": self.container_image,
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "workspace_root": self.workspace_root,
        }


@dataclass
class SweepConfig:
    """Configuration for a full sweep (Phase 0/1/2)."""
    sweep_name: str
    plan_version: str
    instances: List[str]
    conditions: List[ConditionConfig]
    serving_configs: List[ServingConfig]
    seeds: List[int]
    
    repo_mapping: Dict[str, dict]  # instance_id -> {repo_id, base_commit}
    container_image: str
    wall_clock_budget_seconds: int
    output_dir: Path
    
    def generate_runs(self) -> List[RunConfig]:
        """Generate all RunConfig objects for this sweep."""
        runs = []
        for condition in self.conditions:
            for instance_id in self.instances:
                for seed in self.seeds:
                    for serving in self.serving_configs:
                        repo_info = self.repo_mapping[instance_id]
                        run = RunConfig(
                            condition=condition,
                            instance_id=instance_id,
                            seed=seed,
                            serving=serving,
                            plan_version=self.plan_version,
                            repo_id=repo_info["repo_id"],
                            base_commit=repo_info["base_commit"],
                            container_image=self.container_image,
                            wall_clock_budget_seconds=self.wall_clock_budget_seconds,
                            output_dir=self.output_dir,
                        )
                        runs.append(run)
        return runs


def load_composed_config(
    benchmark_path: Path,
    harness_path: Path,
    model_path: Path,
    serving_path: Path,
) -> SweepConfig:
    """Compose a SweepConfig from independent profile files.

    Folder convention:
        profiles/benchmarks/<benchmark>/<sweep>.json   instances/repo_mapping/container/budget/output
        profiles/harnesses/<name>.yaml        harness/runtime behavior + conditions[]
        profiles/models/<name>.json           model identity and model-level limits
        profiles/serving/<engine>/<name>.yaml serving endpoint / engine deployment

    Each call writes under:

        data/experiments/<AB_RUN_STAMP>/runs/<benchmark-output-leaf>/

    Override `AB_RUN_STAMP` for stable / human-named runs, or
    `AB_RUN_DIR`/`AB_EXPERIMENT_DIR` to move the experiment root. The two
    directory variables are strict aliases.

    `_doc` keys at the top level of any file are ignored for older JSON
    profiles; YAML profiles should prefer native comments.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import os as _os

    benchmark = _read_profile(benchmark_path)
    harness = _read_profile(harness_path)
    serving = load_serving_config(serving_path)
    model = load_model_config(model_path, serving=serving)

    if "model" in harness:
        raise ValueError(
            f"harness config {harness_path} still has a top-level `model` block; "
            "model identity now belongs in profiles/models/*.json"
        )

    _validate_model_config(model, model_path=model_path)
    serving_configs = [serving]

    # --- conditions (from harness) -----------------------------------
    conditions = []
    for cond in harness.get("conditions", []):
        agent_cfg = None
        if "agent" in cond:
            ad = cond["agent"]
            agent_cfg = AgentConfig(
                max_steps=ad.get("max_steps", 500),
                max_time_seconds=ad.get("max_time_seconds", 1800),
                retry_limit=ad.get("retry_limit", 10),
                tool_profile=ad.get("tool_profile", "code-agent-minimal"),
                allowed_tools=ad.get("allowed_tools", [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep"
                ]),
            )
        conditions.append(ConditionConfig(
            condition_id=cond["condition_id"],
            condition_type=cond["condition_type"],
            description=cond.get("description", ""),
            model=model,
            agent=agent_cfg,
            ablation_flags=cond.get("ablation_flags", {}),
        ))

    # --- benchmark-level (instances/budget/output) -----------------
    # Singapore time so experiment folder names match wall-clock on this host.
    # Override AB_RUN_STAMP to pin / resume.
    stamp = (
        _os.environ.get("AB_RUN_STAMP")
        or datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    benchmark_output_dir = Path(benchmark["output_dir"])
    experiment_dir = _aliased_run_dir_from_env(_os.environ)
    if experiment_dir:
        output_dir = experiment_dir / "runs" / benchmark_output_dir.name
    else:
        output_dir = Path("data/experiments") / stamp / "runs" / benchmark_output_dir.name

    return SweepConfig(
        sweep_name=benchmark["sweep_name"],
        plan_version=benchmark["plan_version"],
        instances=benchmark["instances"] or [],
        conditions=conditions,
        serving_configs=serving_configs,
        seeds=benchmark.get("seeds", [42]),
        repo_mapping=benchmark.get("repo_mapping") or {},
        container_image=benchmark["container_image"],
        wall_clock_budget_seconds=benchmark["wall_clock_budget_seconds"],
        output_dir=output_dir,
    )


def _read_json(path: Path) -> dict:
    return load_profile(path)


def _aliased_run_dir_from_env(env: dict[str, str]) -> Optional[Path]:
    run_dir = env.get("AB_RUN_DIR")
    experiment_dir = env.get("AB_EXPERIMENT_DIR")
    if run_dir and experiment_dir:
        left = Path(run_dir).expanduser()
        right = Path(experiment_dir).expanduser()
        if left != right:
            raise ValueError(
                "AB_RUN_DIR and AB_EXPERIMENT_DIR are aliases and must match "
                f"when both are set (got {left} and {right})"
            )
        return left
    value = run_dir or experiment_dir
    return Path(value).expanduser() if value else None

def _read_profile(path: Path) -> dict:
    return load_profile(path)


def _endpoint_value(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        direct = value.get("base_url") or value.get("url")
        if direct:
            return direct
        scheme = value.get("scheme", "http")
        host = value.get("client_host") or value.get("host")
        if not host:
            return None
        port = int(value.get("port", 443 if scheme == "https" else 80))
        path = value.get("path", "/v1")
        if not path.startswith("/"):
            path = "/" + path
        return urlunparse((scheme, f"{host}:{port}", path, "", "", ""))
    return None


def _active_provider(serving_data: dict) -> dict:
    provider = serving_data.get("provider") or {}
    if not isinstance(provider, dict):
        return {}
    providers = provider.get("providers")
    if isinstance(providers, dict):
        active = provider.get("active") or provider.get("name")
        if active:
            selected = providers.get(active)
            if isinstance(selected, dict):
                merged = dict(selected)
                merged.setdefault("name", active)
                return merged
    return provider


def _endpoint_from_serving(serving_data: dict) -> str:
    endpoint = serving_data.get("endpoint") or {}
    endpoint_direct = _endpoint_value(endpoint)
    if endpoint_direct:
        return endpoint_direct
    if not isinstance(endpoint, dict):
        endpoint = {}
    scheme = endpoint.get("scheme", "http")
    client_host = endpoint.get("client_host") or endpoint.get("host") or "127.0.0.1"
    port = int(endpoint.get("port", 443 if scheme == "https" else 8000))
    path = endpoint.get("path", "/v1")
    if not path.startswith("/"):
        path = "/" + path
    return urlunparse((scheme, f"{client_host}:{port}", path, "", "", ""))


def load_serving_config(serving_path: Path) -> ServingConfig:
    data = _read_profile(serving_path)
    serving_data = data.get("serving")
    if not isinstance(serving_data, dict):
        raise ValueError(f"serving config {serving_path} is missing `serving`")

    endpoint = serving_data.get("endpoint") or {}
    endpoint_obj = endpoint if isinstance(endpoint, dict) else {}
    runtime = serving_data.get("runtime") or {}
    scheduler = serving_data.get("scheduler") or {}
    proxy = serving_data.get("proxy") or {}
    provider = _active_provider(serving_data)
    backend = serving_data["backend"]
    if backend in {"provider", "api"}:
        raise ValueError(
            f"serving config {serving_path} uses unsupported backend "
            f"{backend!r}; use backend 'litellm' for provider-backed models"
        )

    port = int(endpoint_obj.get("port", runtime.get("port", 8000)))
    bind_host = endpoint_obj.get("bind_host", endpoint_obj.get("host", "0.0.0.0"))
    client_host = endpoint_obj.get("client_host", "127.0.0.1")

    return ServingConfig(
        config_id=serving_data["config_id"],
        backend=backend,
        endpoint=_endpoint_from_serving(serving_data),
        bind_host=bind_host,
        client_host=client_host,
        port=port,
        gpu_mem_util=float(runtime.get("gpu_mem_util", 0.90)),
        tensor_parallel_size=_optional_int(runtime.get("tensor_parallel_size", runtime.get("tp"))),
        cuda_visible_devices=_optional_str(runtime.get("cuda_visible_devices")),
        preset=serving_data.get("preset", "default"),
        serving_mode=_optional_serving_mode(serving_data, runtime, backend),
        max_num_seqs=int(scheduler.get("max_num_seqs", 128)),
        enable_prefix_caching=bool(scheduler.get("enable_prefix_caching", True)),
        image=serving_data.get("image"),
        sif=serving_data.get("sif"),
        provider_name=provider.get("name") or provider.get("provider"),
        provider_base_url=provider.get("base_url") or provider.get("url"),
        provider_api_key_env=provider.get("api_key_env"),
        provider_model=provider.get("model"),
        proxy_api_key=(
            proxy.get("api_key", "sk-agent-bench-local")
            if backend == "litellm" else None
        ),
        proxy_agent_provider=proxy.get("agent_provider"),
        additional_flags=serving_data.get("additional_flags", {}),
        extra_args=_extra_args_value(serving_data.get("extra_args", runtime.get("extra_args", []))),
    )


def _optional_serving_mode(serving_data: dict, runtime: dict, backend: str) -> Optional[str]:
    value = (
        serving_data.get("serving_mode")
        or serving_data.get("servingmode")
        or runtime.get("serving_mode")
        or runtime.get("servingmode")
    )
    if value is None or value == "":
        return None
    mode = str(value).strip().lower()
    allowed = {"native", "docker", "singularity"}
    if mode not in allowed:
        raise ValueError(
            f"serving mode for backend {backend!r} must be one of "
            f"{sorted(allowed)}; got {value!r}"
        )
    return mode


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("integer config values must not be boolean")
    return int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _extra_args_value(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("serving.extra_args must be a string or list")

def load_model_config(model_path: Path, *, serving: Optional[ServingConfig] = None) -> ModelConfig:
    data = _read_profile(model_path)
    model_data = data.get("model")
    if not isinstance(model_data, dict):
        raise ValueError(f"model config {model_path} is missing `model`")

    provider = model_data.get("provider", "custom")
    if serving and serving.provider_name and (not provider or provider == "custom"):
        provider = serving.provider_name
    api_key = model_data.get("api_key", "EMPTY")
    if serving and serving.proxy_api_key:
        api_key = serving.proxy_api_key
    if serving and serving.proxy_agent_provider:
        provider = serving.proxy_agent_provider

    return ModelConfig(
        config_id=model_data["config_id"],
        name=model_data["name"],
        endpoint=serving.endpoint if serving else model_data.get("endpoint", ""),
        temperature=float(model_data.get("temperature", 0.0)),
        max_tokens=int(model_data.get("max_tokens", 8192)),
        context_window=int(model_data.get("context_window", 0)),
        provider=provider,
        api_key=api_key,
        litellm_model=model_data.get("litellm_model"),
        tool_call_parser=model_data.get("tool_call_parser"),
        reasoning_parser=model_data.get("reasoning_parser"),
    )


def _validate_model_config(
    model: ModelConfig,
    *,
    model_path: Path,
) -> None:
    if model.context_window <= 0:
        raise ValueError(
            f"{model_path} model.context_window must be set to a positive integer"
        )
    if model.max_tokens > model.context_window:
        raise ValueError(
            f"{model_path} model.max_tokens={model.max_tokens} exceeds "
            f"model.context_window={model.context_window}"
        )


def load_active_model_and_serving(
    *,
    repo_root: Optional[str] = None,
) -> tuple[ModelConfig, ServingConfig]:
    """Load model + serving paths injected by the driver or shell launcher."""
    import os

    root = Path(repo_root or os.environ.get("AB_REPO_ROOT", os.getcwd()))

    def _resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else root / p

    model_path = os.environ.get("AB_MODEL_CONFIG", "qwen36_27b")
    serving_path = os.environ.get("AB_SERVING_CONFIG", "vllm/default")
    model_path = str(
        resolve_profile_ref(
            model_path,
            root / "profiles" / "models",
            [".json", ".yaml", ".yml"],
        )
    )
    serving_path = str(
        resolve_profile_ref(
            serving_path,
            root / "profiles" / "serving",
            [".yaml", ".yml", ".json", ".toml"],
        )
    )

    model_path_p = _resolve(model_path)
    serving_path_p = _resolve(serving_path)

    serving = load_serving_config(serving_path_p)
    model = load_model_config(model_path_p, serving=serving)
    _validate_model_config(model, model_path=model_path_p)
    return model, serving


def endpoint_for_container(endpoint: str, host_address: str) -> str:
    """Rewrite a host-perspective endpoint for use inside a harness container."""
    for marker in ("://localhost:", "://127.0.0.1:", "://0.0.0.0:"):
        if marker in endpoint:
            scheme, rest = endpoint.split("://", 1)
            _, tail = rest.split(":", 1)
            return f"{scheme}://{host_address}:{tail}"
    return endpoint
