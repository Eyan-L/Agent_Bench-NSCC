from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from agent_bench.serving.launch_plan import (
    build_launch_environment,
    build_launch_plan,
    prepare_launch_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan(model: str, serving: str, **kwargs):
    env = {"AB_REPO_ROOT": str(ROOT), **kwargs.pop("env", {})}
    return build_launch_plan(
        model_path=ROOT / model,
        serving_path=ROOT / serving,
        env=env,
        **kwargs,
    )


def test_vllm_native_launch_plan_uses_profile_and_env_overrides() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/vllm/default.yaml",
        backend="vllm",
        mode="native",
        env={"AB_SERVING_PORT": "8123", "AB_SERVING_TENSOR_PARALLEL_SIZE": "2", "CUDA_VISIBLE_DEVICES": "0,1"},
    )

    assert plan.backend == "vllm"
    assert plan.mode == "native"
    assert plan.endpoint == "http://127.0.0.1:8123/v1"
    assert plan.readiness_url == "http://127.0.0.1:8123/v1/models"
    assert plan.argv[:3] == ("vllm", "serve", "Qwen/Qwen3.5-9B")
    assert "--tensor-parallel-size" in plan.argv
    assert plan.argv[plan.argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-prefix-caching" in plan.argv
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_vllm_docker_launch_plan_does_not_inline_forwarded_token() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/vllm/default.yaml",
        backend="vllm",
        mode="docker",
        env={"HUGGING_FACE_HUB_TOKEN": "hf-secret", "VLLM_DETACH": "0"},
    )

    assert plan.argv[:2] == ("docker", "run")
    assert "--gpus" in plan.argv
    assert "--model" in plan.argv
    assert "Qwen/Qwen3.5-9B" in plan.argv
    assert plan.metadata["container_name"] == "oc-vllm-serve"
    assert "hf-secret" not in " ".join(plan.argv)


def test_sglang_native_and_singularity_launch_plans() -> None:
    native = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
        env={"SGLANG_CUDA_VISIBLE_DEVICES": "0,1", "SGLANG_TP": "2"},
    )

    assert native.argv[:3] == (sys.executable, "-m", "sglang.launch_server")
    assert "--tp" in native.argv
    assert native.argv[native.argv.index("--tp") + 1] == "2"
    assert native.env["SGLANG_KV_MEMTRACE"] == "0"

    singularity = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="singularity",
        env={"AB_SERVING_SIF": "/tmp/sglang.sif"},
    )

    assert singularity.argv[:4] == ("singularity", "exec", "--nv", "--containall")
    assert "/tmp/sglang.sif" in singularity.argv


def test_sglang_pd_mooncake_launch_plan() -> None:
    plan = _plan(
        "profiles/models/qwen36_35b_a3b_fp8_local.json",
        "profiles/serving/sglang/pd_mooncake.yaml",
        backend="sglang",
    )

    assert plan.mode == "pd"
    assert plan.endpoint == "http://127.0.0.1:8000/v1"
    assert plan.readiness_url == "http://127.0.0.1:8000/v1/models"
    assert plan.argv[1:3] == ("-m", "agent_bench.serving.sglang.pd_supervisor")
    assert "--transfer-backend" in plan.argv
    assert plan.argv[plan.argv.index("--transfer-backend") + 1] == "mooncake"
    assert "--mooncake-custom-mem-pool" in plan.argv
    assert plan.argv[plan.argv.index("--mooncake-custom-mem-pool") + 1] == "NVLINK"
    assert "--prefill-cuda-visible-devices" in plan.argv
    assert plan.argv[plan.argv.index("--prefill-cuda-visible-devices") + 1] == "0"
    assert "--decode-cuda-visible-devices" in plan.argv
    assert plan.argv[plan.argv.index("--decode-cuda-visible-devices") + 1] == "1"
    assert "--warmup-count" in plan.argv
    assert plan.argv[plan.argv.index("--warmup-count") + 1] == "1"
    assert "--warmup-max-tokens" in plan.argv
    assert plan.argv[plan.argv.index("--warmup-max-tokens") + 1] == "16"
    assert "--startup-timeout" in plan.argv
    assert plan.argv[plan.argv.index("--startup-timeout") + 1] == "1800"
    assert "--server-flag=--enable-request-time-stats-logging" in plan.argv
    assert "--server-flag=--page-size" in plan.argv
    assert "--server-flag=1" in plan.argv
    assert plan.env["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] == "NVLINK"
    assert plan.env["FLASHINFER_WORKSPACE_BASE"] == "/mnt/vdb/boxi"
    assert plan.metadata["transfer_backend"] == "mooncake"
    assert plan.metadata["startup_timeout_seconds"] == "1800"
    assert plan.metadata["readiness_timeout_seconds"] == "3600"
    assert plan.metadata["warmup_count"] == "1"
    assert plan.metadata["page_size"] == "1"
    assert plan.metadata["requires_fresh_process"] == "1"
    assert plan.metadata["managed_ports"] == "router=8000,prefill=30000,decode=30001,bootstrap=8998"


def test_sglang_pd_nixl_launch_plan() -> None:
    plan = _plan(
        "profiles/models/qwen36_35b_a3b_fp8_local.json",
        "profiles/serving/sglang/pd_nixl.yaml",
        backend="sglang",
    )

    assert plan.mode == "pd"
    assert plan.argv[plan.argv.index("--transfer-backend") + 1] == "nixl"
    assert "--mooncake-custom-mem-pool" not in plan.argv
    assert plan.env["SGLANG_DISAGGREGATION_NIXL_BACKEND"] == "UCX"
    assert plan.env["SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS"] == "{}"
    assert plan.env["UCX_TLS"] == "cuda_ipc,cuda_copy,sm,tcp,self"
    assert plan.env["UCX_MEMTYPE_CACHE"] == "n"
    assert plan.metadata["transfer_backend"] == "nixl"
    assert plan.metadata["nixl_backend"] == "UCX"


def test_serving_profile_mode_selects_sglang_docker() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        env={"SGLANG_CUDA_VISIBLE_DEVICES": "0", "SGLANG_DETACH": "0"},
    )

    assert plan.mode == "docker"
    assert plan.argv[:2] == ("docker", "run")
    assert "lmsysorg/sglang:v0.5.12.post1-cu130" in plan.argv
    assert "python3" in plan.argv
    assert "sglang.launch_server" in plan.argv
    assert "--gpus" in plan.argv
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert plan.metadata["container_name"] == "oc-sglang-serve"


def test_env_serving_mode_overrides_profile_mode() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        env={"AB_SERVING_MODE": "native"},
    )

    assert plan.mode == "native"
    assert plan.argv[:3] == (sys.executable, "-m", "sglang.launch_server")


def test_sglang_native_python_can_be_overridden() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
        env={"SGLANG_PYTHON": "/opt/sglang/bin/python"},
    )

    assert plan.argv[:3] == ("/opt/sglang/bin/python", "-m", "sglang.launch_server")
    assert plan.metadata["python"] == "/opt/sglang/bin/python"


def test_sglang_native_python_bin_is_prepended_to_path() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
        env={"PATH": "/usr/bin", "SGLANG_PYTHON": "/opt/sglang/bin/python"},
    )

    assert plan.env["PATH"] == "/opt/sglang/bin:/usr/bin"


def test_serving_profile_runtime_defaults_and_extra_args(tmp_path: Path) -> None:
    serving = tmp_path / "vllm_profile.yaml"
    serving.write_text(
        """serving:
  config_id: vllm-profile-runtime
  backend: vllm
  preset: default
  endpoint:
    scheme: http
    bind_host: 0.0.0.0
    client_host: 127.0.0.1
    port: 8000
    path: /v1
  runtime:
    tensor_parallel_size: 2
    cuda_visible_devices: "0,1"
    gpu_mem_util: 0.88
  scheduler:
    max_num_seqs: 16
    enable_prefix_caching: false
  additional_flags: {}
  extra_args:
    - --disable-log-requests
    - --max-num-batched-tokens
    - "4096"
""",
        encoding="utf-8",
    )

    plan = _plan(
        "profiles/models/qwen35_9b.json",
        str(serving),
        backend="vllm",
    )

    assert plan.argv[plan.argv.index("--tensor-parallel-size") + 1] == "2"
    assert plan.argv[plan.argv.index("--gpu-memory-utilization") + 1] == "0.88"
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "--disable-log-requests" in plan.argv
    assert plan.argv[plan.argv.index("--max-num-batched-tokens") + 1] == "4096"
    assert plan.metadata["tensor_parallel_size"] == "2"
    assert plan.metadata["cuda_visible_devices"] == "0,1"


def test_vllm_runtime_block_size_attention_backend_and_extra_flags(tmp_path: Path) -> None:
    serving = tmp_path / "decoupled_like.yaml"
    serving.write_text(
        """serving:
  config_id: vllm-decoupled
  backend: vllm
  preset: default
  endpoint:
    scheme: http
    bind_host: 0.0.0.0
    client_host: 127.0.0.1
    port: 8000
    path: /v1
  runtime:
    gpu_mem_util: 0.90
    block_size: 16
    attention_backend: FLASH_ATTN
  scheduler:
    max_num_seqs: 128
    enable_prefix_caching: true
  extra_flags:
    - --disable-custom-all-reduce
""",
        encoding="utf-8",
    )

    plan = _plan("profiles/models/qwen35_9b.json", str(serving), backend="vllm", mode="native")

    assert plan.argv[plan.argv.index("--block-size") + 1] == "16"
    assert plan.argv[plan.argv.index("--attention-backend") + 1] == "FLASH_ATTN"
    # `extra_flags` is honored as an alias for `extra_args`.
    assert "--disable-custom-all-reduce" in plan.argv

    # Env overrides win over the profile.
    overridden = _plan(
        "profiles/models/qwen35_9b.json",
        str(serving),
        backend="vllm",
        mode="native",
        env={"VLLM_BLOCK_SIZE": "32", "VLLM_ATTENTION_BACKEND": "FLASHINFER"},
    )
    assert overridden.argv[overridden.argv.index("--block-size") + 1] == "32"
    assert overridden.argv[overridden.argv.index("--attention-backend") + 1] == "FLASHINFER"


def test_sglang_uses_tp_env_instead_of_visible_gpu_count() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
        env={"SGLANG_CUDA_VISIBLE_DEVICES": "0,1"},
    )

    assert "--tp" not in plan.argv
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert plan.metadata["tensor_parallel_size"] == "1"

    override = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
        env={"AB_SERVING_TENSOR_PARALLEL_SIZE": "2", "CUDA_VISIBLE_DEVICES": "0,1"},
    )

    assert override.argv[override.argv.index("--tp") + 1] == "2"
    assert override.metadata["tensor_parallel_size"] == "2"


def test_sglang_maps_max_num_seqs_to_effective_running_requests() -> None:
    plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
    )

    assert plan.argv[plan.argv.index("--max-running-requests") + 1] == "128"
    assert plan.metadata["max_running_requests"] == "128"
    assert plan.env["SGLANG_MAX_RUNNING_REQUESTS"] == "128"

    overridden = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
        env={"AB_SERVING_MAX_RUNNING_REQUESTS": "32"},
    )

    assert overridden.argv[
        overridden.argv.index("--max-running-requests") + 1
    ] == "32"
    assert overridden.metadata["max_running_requests"] == "32"


def test_sglang_hierarchical_cache_defaults_off_and_can_be_configured(tmp_path: Path) -> None:
    default_plan = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="native",
    )

    assert "--enable-hierarchical-cache" not in default_plan.argv
    assert "--hicache-size" not in default_plan.argv
    assert default_plan.env["SGLANG_ENABLE_HIERARCHICAL_CACHE"] == "0"
    assert default_plan.metadata["enable_hierarchical_cache"] == "0"

    serving = tmp_path / "sglang_hicache.yaml"
    serving.write_text(
        """serving:
  config_id: sglang-hicache
  backend: sglang
  preset: default
  serving_mode: native
  endpoint:
    scheme: http
    bind_host: 0.0.0.0
    client_host: 127.0.0.1
    port: 8000
    path: /v1
  runtime:
    gpu_mem_util: 0.85
    enable_hierarchical_cache: true
    hicache_size: 24
  scheduler:
    max_num_seqs: 128
    enable_prefix_caching: true
  additional_flags: {}
  extra_args: []
""",
        encoding="utf-8",
    )

    configured_plan = _plan(
        "profiles/models/qwen35_9b.json",
        str(serving),
        backend="sglang",
    )

    assert configured_plan.argv[configured_plan.argv.index("--hicache-size") + 1] == "24"
    assert configured_plan.env["SGLANG_ENABLE_HIERARCHICAL_CACHE"] == "1"
    assert configured_plan.env["SGLANG_HICACHE_SIZE"] == "24"
    assert configured_plan.metadata["hicache_size"] == "24"

    disabled_by_env = _plan(
        "profiles/models/qwen35_9b.json",
        str(serving),
        backend="sglang",
        env={"SGLANG_ENABLE_HIERARCHICAL_CACHE": "0"},
    )

    assert "--enable-hierarchical-cache" not in disabled_by_env.argv
    assert "--hicache-size" not in disabled_by_env.argv


def test_litellm_native_launch_plan_uses_proxy_auth_without_upstream_secret() -> None:
    plan = _plan(
        "profiles/models/gemini_3_5_flash.json",
        "profiles/serving/litellm_gemini.toml",
        backend="litellm",
        mode="native",
    )

    assert plan.argv[0] == "litellm"
    assert plan.mode == "native"
    assert plan.env["AB_SERVING_API_KEY"] == "sk-agent-bench-local"
    assert plan.readiness_headers == {"Authorization": "Bearer sk-agent-bench-local"}
    assert plan.upstream_api_key_env == "GEMINI_API_KEY"
    assert "GEMINI_API_KEY" in plan.secret_env_names
    assert "LITELLM_UPSTREAM_API_KEY" not in plan.env


def test_litellm_docker_launch_plan_uses_host_network_and_env_secrets(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    plan = _plan(
        "profiles/models/minimax_m2_7.json",
        "profiles/serving/litellm/minimax_global.yaml",
        backend="litellm",
        mode="docker",
        env={
            "LITELLM_CONFIG_PATH": str(config_path),
            "LITELLM_CONTAINER_NAME": "litellm-test",
            "LITELLM_EXTRA_DOCKER_ARGS": "--pull never",
        },
    )

    assert plan.argv[:2] == ("docker", "run")
    assert plan.mode == "docker"
    assert "--rm" in plan.argv
    assert plan.argv[plan.argv.index("--name") + 1] == "litellm-test"
    assert plan.argv[plan.argv.index("--network") + 1] == "host"
    assert plan.argv[plan.argv.index("-v") + 1] == f"{config_path}:{config_path}:ro"
    image_index = plan.argv.index("docker.litellm.ai/berriai/litellm:main-stable")
    assert plan.argv[image_index + 1] == "--config"
    assert "docker://docker.litellm.ai/berriai/litellm:main-stable" not in plan.argv
    assert "LITELLM_UPSTREAM_API_KEY" in plan.argv
    assert "--pull" in plan.argv
    assert "never" in plan.argv
    assert plan.metadata["container_name"] == "litellm-test"
    assert "LITELLM_UPSTREAM_API_KEY" not in plan.env
    assert "sk-upstream-secret" not in " ".join(plan.argv)


def test_litellm_singularity_launch_plan_still_uses_sif() -> None:
    plan = _plan(
        "profiles/models/minimax_m2_7.json",
        "profiles/serving/litellm/minimax_global.yaml",
        backend="litellm",
        mode="singularity",
        env={"AB_SERVING_SIF": "/tmp/litellm.sif"},
    )

    assert plan.argv[:4] == ("singularity", "exec", "--containall", "--bind")
    assert "/tmp/litellm.sif" in plan.argv
    assert plan.mode == "singularity"


def test_litellm_launch_environment_and_config_do_not_emit_upstream_secret(tmp_path: Path) -> None:
    plan = _plan(
        "profiles/models/minimax_m2_7.json",
        "profiles/serving/litellm/minimax_global.yaml",
        backend="litellm",
        env={"LITELLM_CONFIG_PATH": str(tmp_path / "litellm.yaml")},
    )
    launch_env = build_launch_environment(plan, base_env={"MINIMAX_API_KEY": "sk-upstream-secret"})

    assert launch_env["LITELLM_UPSTREAM_API_KEY"] == "sk-upstream-secret"
    redacted = json.dumps(plan.to_redacted_dict())
    assert "sk-upstream-secret" not in redacted

    prepare_launch_artifacts(plan, launch_env=launch_env, repo_root=tmp_path)
    rendered = (tmp_path / "litellm.yaml").read_text()
    assert 'api_base: "os.environ/LITELLM_UPSTREAM_BASE_URL"' in rendered
    assert "os.environ/LITELLM_UPSTREAM_API_KEY" in rendered
    assert "sk-upstream-secret" not in rendered


def test_shell_wrappers_delegate_to_python_launcher_without_launching() -> None:
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "agent_bench/serving/vllm/serve.sh"),
            "--model-config",
            "qwen35_9b",
            "--serving-config",
            "vllm/default",
            "--print-json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["backend"] == "vllm"
    assert payload["argv"][:3] == ["vllm", "serve", "Qwen/Qwen3.5-9B"]
