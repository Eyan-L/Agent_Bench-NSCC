from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import stat
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from agent_bench.core.config import ModelConfig, ServingConfig, load_model_config, load_serving_config
from agent_bench.core.profile_io import resolve_profile_ref


class LaunchPlanError(ValueError):
    """Raised when a serving launch plan cannot be resolved safely."""


@dataclass(frozen=True)
class ServingLaunchPlan:
    backend: str
    mode: str
    argv: tuple[str, ...]
    env: dict[str, str]
    endpoint: str
    readiness_url: str
    readiness_headers: dict[str, str]
    log_path: str | None
    pid_name: str
    config_id: str
    model_name: str
    secret_env_names: tuple[str, ...] = ()
    upstream_api_key_env: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def command_for_display(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "argv": list(self.argv),
            "env": redact_env(self.env, self.secret_env_names),
            "endpoint": self.endpoint,
            "readiness_url": self.readiness_url,
            "readiness_headers": redact_env(self.readiness_headers, self.secret_env_names),
            "log_path": self.log_path,
            "pid_name": self.pid_name,
            "config_id": self.config_id,
            "model_name": self.model_name,
            "secret_env_names": list(self.secret_env_names),
            "metadata": dict(self.metadata),
        }


def build_launch_plan(
    *,
    model_path: str | Path,
    serving_path: str | Path,
    backend: str | None = None,
    mode: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
) -> ServingLaunchPlan:
    base_env = dict(os.environ if env is None else env)
    serving = load_serving_config(Path(serving_path))
    if backend is not None and backend != serving.backend:
        raise LaunchPlanError(
            f"requested backend {backend!r} does not match serving profile backend {serving.backend!r}"
        )
    backend = serving.backend
    if backend not in {"vllm", "sglang", "litellm"}:
        raise LaunchPlanError(f"unsupported serving backend for launch: {backend}")

    model = load_model_config(Path(model_path), serving=serving)
    selected_mode = (
        mode
        or base_env.get(_backend_env_name(backend, "MODE"))
        or base_env.get("AB_SERVING_MODE")
        or serving.serving_mode
        or "native"
    )
    selected_mode = selected_mode.lower()
    log_value = str(log_path) if log_path is not None else _default_log_path(backend, base_env)

    if backend == "vllm":
        return _build_vllm_plan(model, serving, selected_mode, base_env, log_value)
    if backend == "sglang":
        return _build_sglang_plan(model, serving, selected_mode, base_env, log_value)
    return _build_litellm_plan(model, serving, selected_mode, base_env, log_value)


def build_launch_plan_from_env(
    *,
    backend: str | None = None,
    mode: str | None = None,
    env: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
    log_path: str | Path | None = None,
) -> ServingLaunchPlan:
    base_env = dict(os.environ if env is None else env)
    root = Path(repo_root or base_env.get("AB_REPO_ROOT") or Path.cwd()).expanduser().resolve()

    selected_backend = backend or base_env.get("AB_SERVING_BACKEND")
    model_path = _resolve_model_path(base_env, root, selected_backend)
    serving_path = _resolve_serving_path(base_env, root, selected_backend)
    if selected_backend is None:
        selected_backend = load_serving_config(serving_path).backend

    return build_launch_plan(
        model_path=model_path,
        serving_path=serving_path,
        backend=selected_backend,
        mode=mode,
        env=base_env,
        log_path=log_path,
    )


def build_launch_environment(
    plan: ServingLaunchPlan,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.update(plan.env)
    if plan.backend == "litellm":
        source = plan.upstream_api_key_env
        if not source:
            raise LaunchPlanError("LiteLLM serving profile must define provider api_key_env")
        upstream_key = env.get(source)
        if not upstream_key:
            raise LaunchPlanError(f"upstream API key env {source} is empty")
        env["LITELLM_UPSTREAM_API_KEY"] = upstream_key
    return env


def prepare_launch_artifacts(
    plan: ServingLaunchPlan,
    *,
    launch_env: Mapping[str, str],
    repo_root: str | Path | None = None,
) -> None:
    root = Path(repo_root or launch_env.get("AB_REPO_ROOT") or Path.cwd()).expanduser().resolve()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "serving_config_id").write_text(plan.config_id, encoding="utf-8")
    if plan.metadata:
        (data_dir / "serving_metadata.json").write_text(
            json.dumps(plan.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if plan.backend == "litellm":
        from agent_bench.serving.render_litellm_config import render_config

        config_path = Path(launch_env["LITELLM_CONFIG_PATH"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with _patched_environ(launch_env):
            config_path.write_text(render_config(), encoding="utf-8")
        config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def redact_env(values: Mapping[str, str], secret_names: Sequence[str] = ()) -> dict[str, str]:
    secret_set = set(secret_names)
    redacted: dict[str, str] = {}
    for key, value in values.items():
        upper = key.upper()
        if key in secret_set or upper == "AUTHORIZATION" or "SECRET" in upper or "TOKEN" in upper or "KEY" in upper:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = str(value)
    return redacted


def _build_vllm_plan(
    model: ModelConfig,
    serving: ServingConfig,
    mode: str,
    env: Mapping[str, str],
    log_path: str | None,
) -> ServingLaunchPlan:
    if mode not in {"native", "docker", "singularity"}:
        raise LaunchPlanError(f"unsupported vLLM launch mode: {mode}")
    port = _env_int(env, "VLLM_PORT", "AB_SERVING_PORT", default=serving.port)
    bind_host = _env_str(env, "VLLM_HOST", "AB_SERVING_BIND_HOST", default=serving.bind_host)
    client_host = _env_str(env, "AB_SERVING_CLIENT_HOST", default=serving.client_host)
    endpoint = _endpoint_with_host_port(serving.endpoint, host=client_host, port=port)
    model_name = _env_str(env, "VLLM_MODEL", "AB_MODEL_NAME", default=model.name)
    max_model_len = _env_int(
        env,
        "VLLM_MAX_LEN",
        "AB_SERVING_MAX_MODEL_LEN",
        "AB_MODEL_CONTEXT_WINDOW",
        default=model.context_window,
    )
    tp = _env_int(
        env,
        "VLLM_TP",
        "AB_SERVING_TENSOR_PARALLEL_SIZE",
        default=serving.tensor_parallel_size or 1,
    )
    cuda_devices = _env_str(
        env,
        "VLLM_CUDA_VISIBLE_DEVICES",
        "AB_SERVING_CUDA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        default=serving.cuda_visible_devices or "1",
    )
    mig_uuid = env.get("VLLM_MIG_UUID", "") if mode == "docker" else ""
    if mig_uuid:
        tp = 1
    gpu_util = _env_float(env, "VLLM_GPU_UTIL", "AB_SERVING_GPU_MEMORY_UTILIZATION", default=serving.gpu_mem_util)
    preset = _env_str(env, "AB_SERVING_PRESET", default=serving.preset)
    tool_parser = _env_optional(env, "AB_MODEL_TOOL_CALL_PARSER", default=model.tool_call_parser or "qwen3_coder")
    reasoning_parser = _env_optional(env, "AB_MODEL_REASONING_PARSER", default=model.reasoning_parser)
    device = _env_str(env, "VLLM_DEVICE", default="cuda")

    base_flags = [
        "--host",
        bind_host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--trust-remote-code",
    ]
    if mode == "native":
        base_flags.append("--enable-auto-tool-choice")
    if tool_parser:
        if mode == "docker":
            base_flags.append("--enable-auto-tool-choice")
        base_flags.extend(["--tool-call-parser", tool_parser])
    if reasoning_parser:
        base_flags.extend(["--reasoning-parser", reasoning_parser])

    config_id, preset_flags = _vllm_preset_flags(
        preset=preset,
        config_id=_env_str(env, "AB_SERVING_CONFIG_ID", default=serving.config_id),
        max_num_seqs=_env_int(env, "AB_SERVING_MAX_NUM_SEQS", default=serving.max_num_seqs),
        enable_prefix_caching=_env_bool(
            env,
            "AB_SERVING_ENABLE_PREFIX_CACHING",
            default=serving.enable_prefix_caching,
        ),
    )
    all_flags = (
        base_flags
        + preset_flags
        + _additional_flags(serving.additional_flags)
        + _extra_args(serving.extra_args)
        + shlex.split(env.get("VLLM_EXTRA_ARGS", ""))
    )

    launch_env = {
        "AB_SERVING_BASE_URL": endpoint,
        "OPENAI_BASE_URL": endpoint,
        "AB_SERVING_BACKEND": "vllm",
        "AB_SERVING_CONFIG_ID": config_id,
        "CUDA_VISIBLE_DEVICES": cuda_devices,
    }
    if mode == "singularity":
        launch_env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = cuda_devices
    metadata = {
        "tensor_parallel_size": str(tp),
        "cuda_visible_devices": cuda_devices,
        "preset": preset,
        "mode": mode,
    }
    if mode == "native":
        argv = ("vllm", "serve", model_name, *all_flags)
    elif mode == "docker":
        image = _env_str(env, "VLLM_IMAGE", "AB_SERVING_IMAGE", default=serving.image or "vllm/vllm-openai:v0.11.0")
        container = _env_str(env, "VLLM_CONTAINER_NAME", default="oc-vllm-serve")
        hf_home = _env_str(env, "VLLM_HOST_HF_HOME", "HF_HOME", default=str(Path.home() / ".cache/huggingface"))
        docker_args = ["run", "--name", container, "--network", "host", "--ipc", "host", "--shm-size", "16g"]
        if mig_uuid:
            mig = mig_uuid if mig_uuid.startswith("MIG-") else f"MIG-{mig_uuid}"
            docker_args.extend(["--device", f"nvidia.com/gpu={mig}"])
        else:
            docker_args.extend(["--gpus", _docker_gpu_request(cuda_devices)])
        docker_args.extend(["-v", f"{hf_home}:/root/.cache/huggingface:rw", "-e", "VLLM_USAGE_SOURCE=oc-eval-docker"])
        if env.get("HUGGING_FACE_HUB_TOKEN"):
            docker_args.extend(["-e", "HUGGING_FACE_HUB_TOKEN"])
        if _env_bool(env, "VLLM_DETACH", default=True):
            docker_args.append("-d")
        else:
            docker_args.extend(["--rm", "-it"])
        docker_args.extend(shlex.split(env.get("VLLM_EXTRA_DOCKER_ARGS", "")))
        argv = ("docker", *docker_args, image, "--model", model_name, *all_flags)
        metadata["container_name"] = container
        metadata["image"] = image
    else:
        sif = _env_str(env, "VLLM_SIF", "AB_SERVING_SIF", default=serving.sif or "")
        if not sif:
            raise LaunchPlanError("VLLM_SIF or serving.sif is required for singularity mode")
        gpu_flag = ("--nv",) if device == "cuda" else ()
        argv = ("singularity", "exec", *gpu_flag, "--containall", sif, "vllm", "serve", model_name, *all_flags)

    return ServingLaunchPlan(
        backend="vllm",
        mode=mode,
        argv=tuple(argv),
        env=launch_env,
        endpoint=endpoint,
        readiness_url=endpoint.rstrip("/") + "/models",
        readiness_headers={},
        log_path=log_path,
        pid_name="vllm",
        config_id=config_id,
        model_name=model_name,
        metadata=metadata,
    )


def _build_sglang_plan(
    model: ModelConfig,
    serving: ServingConfig,
    mode: str,
    env: Mapping[str, str],
    log_path: str | None,
) -> ServingLaunchPlan:
    if mode not in {"native", "docker", "singularity"}:
        raise LaunchPlanError(f"unsupported SGLang launch mode: {mode}")
    port = _env_int(env, "SGLANG_PORT", "AB_SERVING_PORT", default=serving.port)
    bind_host = _env_str(env, "SGLANG_HOST", "AB_SERVING_BIND_HOST", default=serving.bind_host)
    client_host = _env_str(env, "AB_SERVING_CLIENT_HOST", default=serving.client_host)
    endpoint = _endpoint_with_host_port(serving.endpoint, host=client_host, port=port)
    cuda_devices = _env_str(
        env,
        "SGLANG_CUDA_VISIBLE_DEVICES",
        "AB_SERVING_CUDA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        default=serving.cuda_visible_devices or "0",
    )
    tp = _env_int(
        env,
        "SGLANG_TP",
        "AB_SERVING_TENSOR_PARALLEL_SIZE",
        default=serving.tensor_parallel_size or 1,
    )
    model_name = _env_str(env, "SGLANG_MODEL_PATH", "AB_MODEL_NAME", default=model.name)
    gpu_util = _env_float(env, "SGLANG_MEM_FRACTION_STATIC", "AB_SERVING_GPU_MEMORY_UTILIZATION", default=serving.gpu_mem_util)
    context_length = _env_optional(
        env,
        "SGLANG_CONTEXT_LENGTH",
        "AB_SERVING_MAX_MODEL_LEN",
        "AB_MODEL_CONTEXT_WINDOW",
        default=str(model.context_window) if model.context_window else None,
    )
    page_size = _env_str(env, "SGLANG_PAGE_SIZE", default="64")
    hicache_size = _env_str(env, "SGLANG_HICACHE_SIZE", default="60")
    tool_parser = _env_optional(env, "SGLANG_TOOL_CALL_PARSER", "AB_MODEL_TOOL_CALL_PARSER", default=model.tool_call_parser or "qwen")
    reasoning_parser = _env_optional(env, "SGLANG_REASONING_PARSER", "AB_MODEL_REASONING_PARSER", default=model.reasoning_parser)
    hierarchical_cache = _env_bool(env, "SGLANG_ENABLE_HIERARCHICAL_CACHE", default=True)

    server_flags = [
        "--model-path",
        model_name,
        "--host",
        bind_host,
        "--port",
        str(port),
        "--enable-metrics",
        "--enable-cache-report",
        "--mem-fraction-static",
        str(gpu_util),
        "--page-size",
        page_size,
    ]
    if tp > 1:
        server_flags.extend(["--tp", str(tp)])
    if context_length:
        server_flags.extend(["--context-length", str(context_length)])
    if hierarchical_cache:
        server_flags.extend(["--enable-hierarchical-cache", "--hicache-size", hicache_size])
    if tool_parser:
        server_flags.extend(["--tool-call-parser", tool_parser])
    if reasoning_parser:
        server_flags.extend(["--reasoning-parser", reasoning_parser])
    server_flags.extend(_additional_flags(serving.additional_flags))
    server_flags.extend(_extra_args(serving.extra_args))
    server_flags.extend(shlex.split(env.get("SGLANG_EXTRA_ARGS", "")))

    launch_env = {
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "AB_SERVING_BASE_URL": endpoint,
        "OPENAI_BASE_URL": endpoint,
        "AB_SERVING_BACKEND": "sglang",
        "AB_SERVING_CONFIG_ID": _env_str(env, "AB_SERVING_CONFIG_ID", default=serving.config_id),
        "SGLANG_KV_MEMTRACE": "0",
        "SGLANG_KV_MEMTRACE_STRICT": "0",
    }
    if mode == "singularity":
        launch_env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = cuda_devices
    pid_file = env.get("SGLANG_PID_FILE")
    if pid_file:
        launch_env["SGLANG_PID_FILE"] = pid_file
    metadata = {
        "tensor_parallel_size": str(tp),
        "cuda_visible_devices": cuda_devices,
        "mode": mode,
    }

    if mode == "native":
        argv = ("python3", "-m", "sglang.launch_server", *server_flags)
    elif mode == "docker":
        image = _docker_image_ref(
            _env_str(
                env,
                "SGLANG_IMAGE",
                "AB_SERVING_IMAGE",
                default=serving.image or "lmsysorg/sglang:v0.5.12.post1-cu130",
            )
        )
        container = _env_str(env, "SGLANG_CONTAINER_NAME", default="oc-sglang-serve")
        hf_home = _env_str(env, "SGLANG_HOST_HF_HOME", "HF_HOME", default=str(Path.home() / ".cache/huggingface"))
        docker_args = ["run", "--name", container, "--network", "host", "--ipc", "host", "--shm-size", "16g"]
        docker_args.extend(["--gpus", _docker_gpu_request(cuda_devices)])
        docker_args.extend(["-v", f"{hf_home}:/root/.cache/huggingface:rw"])
        model_mount = _docker_model_mount(model_name)
        if model_mount:
            docker_args.extend(["-v", model_mount])
        if env.get("HUGGING_FACE_HUB_TOKEN"):
            docker_args.extend(["-e", "HUGGING_FACE_HUB_TOKEN"])
        if env.get("HF_TOKEN"):
            docker_args.extend(["-e", "HF_TOKEN"])
        if _env_bool(env, "SGLANG_DETACH", default=True):
            docker_args.append("-d")
        else:
            docker_args.extend(["--rm", "-it"])
        docker_args.extend(shlex.split(env.get("SGLANG_EXTRA_DOCKER_ARGS", "")))
        argv = ("docker", *docker_args, image, "python3", "-m", "sglang.launch_server", *server_flags)
        metadata["container_name"] = container
        metadata["image"] = image
    else:
        sif = _env_str(env, "SGLANG_SIF", "AB_SERVING_SIF", default=serving.sif or "")
        if not sif:
            raise LaunchPlanError("SGLANG_SIF or serving.sif is required for singularity mode")
        argv = ("singularity", "exec", "--nv", "--containall", sif, "python3", "-m", "sglang.launch_server", *server_flags)

    return ServingLaunchPlan(
        backend="sglang",
        mode=mode,
        argv=tuple(argv),
        env=launch_env,
        endpoint=endpoint,
        readiness_url=endpoint.rstrip("/") + "/models",
        readiness_headers={},
        log_path=log_path,
        pid_name="sglang",
        config_id=launch_env["AB_SERVING_CONFIG_ID"],
        model_name=model_name,
        metadata=metadata,
    )


def _build_litellm_plan(
    model: ModelConfig,
    serving: ServingConfig,
    mode: str,
    env: Mapping[str, str],
    log_path: str | None,
) -> ServingLaunchPlan:
    if mode not in {"native", "docker", "singularity"}:
        raise LaunchPlanError(f"unsupported LiteLLM launch mode: {mode}")
    port = _env_int(env, "LITELLM_PORT", "AB_SERVING_PORT", default=serving.port)
    bind_host = _env_str(env, "LITELLM_HOST", "AB_SERVING_BIND_HOST", default=serving.bind_host)
    client_host = _env_str(env, "AB_SERVING_CLIENT_HOST", default=serving.client_host)
    endpoint = _endpoint_with_host_port(serving.endpoint, host=client_host, port=port)
    master_key = _env_str(env, "LITELLM_MASTER_KEY", "LITELLM_PROXY_API_KEY", default=serving.proxy_api_key or "sk-agent-bench-local")
    model_alias = _env_str(env, "LITELLM_MODEL_ALIAS", "AB_MODEL_NAME", default=model.name)
    upstream_model = _env_str(
        env,
        "LITELLM_UPSTREAM_MODEL",
        "LITELLM_MODEL",
        default=serving.provider_model or model.litellm_model or model.name,
    )
    upstream_base_url = _env_optional(env, "LITELLM_UPSTREAM_BASE_URL", "AB_SERVING_PROVIDER_BASE_URL", default=serving.provider_base_url)
    num_workers = _env_int(env, "LITELLM_NUM_WORKERS", "LITELLM_PROXY_NUM_WORKERS", default=4)
    upstream_api_key_env = _env_optional(env, "AB_SERVING_PROVIDER_API_KEY_ENV", default=serving.provider_api_key_env)
    config_path = _env_str(env, "LITELLM_CONFIG_PATH", default=str(Path(env.get("AB_REPO_ROOT", os.getcwd())) / "data" / "litellm" / "litellm_config.yaml"))

    launch_env = {
        "LITELLM_MASTER_KEY": master_key,
        "LITELLM_MODEL_ALIAS": model_alias,
        "LITELLM_UPSTREAM_MODEL": upstream_model,
        "LITELLM_PORT": str(port),
        "LITELLM_NUM_WORKERS": str(num_workers),
        "LITELLM_CONFIG_PATH": config_path,
        "AB_SERVING_BASE_URL": endpoint,
        "OPENAI_BASE_URL": endpoint,
        "AB_SERVING_API_KEY": master_key,
        "OPENAI_API_KEY": master_key,
        "HERMES_INFERENCE_PROVIDER": _env_str(env, "LITELLM_AGENT_PROVIDER", default=serving.proxy_agent_provider or "custom"),
        "NO_TELEMETRY": _env_str(env, "NO_TELEMETRY", default="1"),
        "AB_SERVING_BACKEND": "litellm",
        "AB_SERVING_CONFIG_ID": _env_str(env, "AB_SERVING_CONFIG_ID", default=serving.config_id),
    }
    if upstream_base_url:
        launch_env["LITELLM_UPSTREAM_BASE_URL"] = upstream_base_url

    metadata = {"upstream_model": upstream_model, "mode": mode}
    if mode == "native":
        argv = (
            "litellm",
            "--config",
            config_path,
            "--host",
            bind_host,
            "--port",
            str(port),
            "--num_workers",
            str(num_workers),
        )
    elif mode == "docker":
        image = _docker_image_ref(
            _env_str(
                env,
                "LITELLM_IMAGE",
                "AB_SERVING_IMAGE",
                default=serving.image or "docker.litellm.ai/berriai/litellm:main-stable",
            )
        )
        container_name = _env_str(env, "LITELLM_CONTAINER_NAME", default=f"agent-bench-litellm-{port}")
        docker_args = [
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "host",
            "-v",
            f"{config_path}:{config_path}:ro",
            "-e",
            "LITELLM_UPSTREAM_API_KEY",
            "-e",
            "LITELLM_MASTER_KEY",
            "-e",
            "NO_TELEMETRY",
        ]
        if upstream_base_url:
            docker_args.extend(["-e", "LITELLM_UPSTREAM_BASE_URL"])
        docker_args.extend(shlex.split(env.get("LITELLM_EXTRA_DOCKER_ARGS", "")))
        argv = (
            "docker",
            *docker_args,
            image,
            "--config",
            config_path,
            "--host",
            bind_host,
            "--port",
            str(port),
            "--num_workers",
            str(num_workers),
        )
        metadata["container_name"] = container_name
        metadata["image"] = image
    else:
        sif = _env_str(env, "LITELLM_SIF", "AB_SERVING_SIF", default=serving.sif or "")
        if not sif:
            raise LaunchPlanError("LITELLM_SIF or serving.sif is required for singularity mode")
        argv = (
            "singularity",
            "exec",
            "--containall",
            "--bind",
            f"{config_path}:{config_path}:ro",
            sif,
            "litellm",
            "--config",
            config_path,
            "--host",
            bind_host,
            "--port",
            str(port),
            "--num_workers",
            str(num_workers),
        )

    secret_names = tuple(
        name for name in (upstream_api_key_env, "LITELLM_UPSTREAM_API_KEY") if name
    )
    return ServingLaunchPlan(
        backend="litellm",
        mode=mode,
        argv=tuple(argv),
        env=launch_env,
        endpoint=endpoint,
        readiness_url=endpoint.rstrip("/") + "/models",
        readiness_headers={"Authorization": f"Bearer {master_key}"},
        log_path=log_path,
        pid_name="litellm",
        config_id=launch_env["AB_SERVING_CONFIG_ID"],
        model_name=model_alias,
        secret_env_names=secret_names,
        upstream_api_key_env=upstream_api_key_env,
        metadata=metadata,
    )


def _docker_image_ref(value: str) -> str:
    image = value.removeprefix("docker://")
    if not image:
        raise LaunchPlanError("Docker image reference must be non-empty")
    return image


def _vllm_preset_flags(
    *,
    preset: str,
    config_id: str,
    max_num_seqs: int,
    enable_prefix_caching: bool,
) -> tuple[str, list[str]]:
    if preset == "default":
        flags = ["--max-num-seqs", str(max_num_seqs)]
        if enable_prefix_caching:
            flags.append("--enable-prefix-caching")
        return config_id, flags
    if preset == "latency":
        return "vllm-latency", ["--max-num-seqs", "8", "--enable-prefix-caching", "--scheduling-policy", "priority"]
    if preset == "throughput":
        return "vllm-throughput", ["--max-num-seqs", "256", "--enable-prefix-caching", "--enable-expert-parallel"]
    if preset == "context":
        return "vllm-context", ["--max-num-seqs", "32", "--enable-prefix-caching", "--kv-cache-dtype", "fp8"]
    raise LaunchPlanError(f"unknown vLLM preset: {preset}")


def _docker_gpu_request(cuda_devices: str) -> str:
    value = str(cuda_devices).strip()
    if not value or value.lower() == "all":
        return "all"
    return f"device={value}"


def _docker_model_mount(model_name: str) -> str | None:
    path = Path(model_name).expanduser()
    if not path.is_absolute() or not path.exists():
        return None
    mount_root = path
    if path.parent.name == "snapshots":
        mount_root = path.parent.parent
    if path.is_file():
        mount_root = path.parent
    return f"{mount_root}:{mount_root}:ro"


def _extra_args(args: Sequence[str] | str | None) -> list[str]:
    if not args:
        return []
    if isinstance(args, str):
        return shlex.split(args)
    return [str(item) for item in args]


def _additional_flags(flags: Mapping[str, Any] | None) -> list[str]:
    if not flags:
        return []
    out: list[str] = []
    for key, value in flags.items():
        if value is None or value is False:
            continue
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            out.append(flag)
        elif isinstance(value, list):
            for item in value:
                out.extend([flag, str(item)])
        else:
            out.extend([flag, str(value)])
    return out


def _resolve_model_path(env: Mapping[str, str], root: Path, backend: str | None) -> Path:
    keys = []
    if backend:
        keys.append(_backend_env_name(backend, "MODEL_CONFIG"))
    keys.append("AB_MODEL_CONFIG")
    for key in keys:
        value = env.get(key)
        if value:
            return _resolve_path(value, root)
    profile = env.get("AB_MODEL_CONFIG", "qwen36_27b")
    return resolve_profile_ref(profile, root / "profiles" / "models", [".json", ".yaml", ".yml"])


def _resolve_serving_path(env: Mapping[str, str], root: Path, backend: str | None) -> Path:
    keys = []
    if backend:
        keys.append(_backend_env_name(backend, "SERVING_CONFIG"))
    keys.append("AB_SERVING_CONFIG")
    for key in keys:
        value = env.get(key)
        if value:
            return _resolve_path(value, root)
    serving = f"{backend}/default" if backend else "vllm/default"
    return resolve_profile_ref(serving, root / "profiles" / "serving", [".yaml", ".yml", ".json", ".toml"])


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    direct = (Path.cwd() / path).resolve()
    if direct.exists():
        return direct
    return (root / path).resolve()


def _backend_env_name(backend: str, suffix: str) -> str:
    return f"{backend.upper()}_{suffix}"


def _env_str(env: Mapping[str, str], *keys: str, default: str) -> str:
    for key in keys:
        value = env.get(key)
        if value not in (None, ""):
            return _expand_vars(str(value), env)
    return _expand_vars(str(default), env)


def _env_optional(env: Mapping[str, str], *keys: str, default: str | None) -> str | None:
    for key in keys:
        value = env.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _env_int(env: Mapping[str, str], *keys: str, default: int) -> int:
    return int(_env_str(env, *keys, default=str(default)))


def _env_float(env: Mapping[str, str], *keys: str, default: float) -> float:
    return float(_env_str(env, *keys, default=str(default)))


def _env_bool(env: Mapping[str, str], *keys: str, default: bool) -> bool:
    value = _env_optional(env, *keys, default=None)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _default_log_path(backend: str, env: Mapping[str, str]) -> str | None:
    keys = {
        "vllm": ("VLLM_LOG_FILE",),
        "sglang": ("SGLANG_LOG_FILE",),
        "litellm": ("LITELLM_LOG_FILE",),
    }[backend]
    for key in keys:
        if env.get(key):
            return env[key]
    return None


def _endpoint_with_host_port(endpoint: str, *, host: str, port: int) -> str:
    parsed = urlsplit(endpoint)
    if not parsed.scheme:
        return endpoint.rstrip("/")
    netloc = f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path or "/v1", "", "")).rstrip("/")


def _expand_vars(value: str, env: Mapping[str, str]) -> str:
    expanded = value
    for key, env_value in env.items():
        expanded = expanded.replace(f"${key}", str(env_value))
        expanded = expanded.replace("${" + key + "}", str(env_value))
    return os.path.expanduser(os.path.expandvars(expanded))


@contextmanager
def _patched_environ(values: Mapping[str, str]):
    old: dict[str, str | None] = {}
    for key, value in values.items():
        old[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
