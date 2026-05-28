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
    env = {"KIT_ROOT": str(ROOT), **kwargs.pop("env", {})}
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
        env={"PORT": "8123", "TP": "2", "CUDA_VISIBLE_DEVICES": "0,1"},
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
    assert "hf-secret" not in " ".join(plan.argv)


def test_sglang_native_and_singularity_launch_plans() -> None:
    native = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        env={"SGLANG_CUDA_VISIBLE_DEVICES": "0,1"},
    )

    assert native.argv[:3] == ("python3", "-m", "sglang.launch_server")
    assert "--tp" in native.argv
    assert native.argv[native.argv.index("--tp") + 1] == "2"
    assert native.env["SGLANG_KV_MEMTRACE"] == "0"

    singularity = _plan(
        "profiles/models/qwen35_9b.json",
        "profiles/serving/sglang/default.yaml",
        backend="sglang",
        mode="singularity",
        env={"SERVING_SIF": "/tmp/sglang.sif"},
    )

    assert singularity.argv[:4] == ("singularity", "exec", "--nv", "--containall")
    assert "/tmp/sglang.sif" in singularity.argv


def test_litellm_launch_plan_uses_proxy_auth_without_upstream_secret() -> None:
    plan = _plan(
        "profiles/models/gemini_3_5_flash.json",
        "profiles/serving/litellm_gemini.toml",
        backend="litellm",
    )

    assert plan.argv[0] == "litellm"
    assert plan.env["MODEL_API_KEY"] == "sk-agent-bench-local"
    assert plan.readiness_headers == {"Authorization": "Bearer sk-agent-bench-local"}
    assert plan.upstream_api_key_env == "GEMINI_API_KEY"
    assert "GEMINI_API_KEY" in plan.secret_env_names
    assert "LITELLM_UPSTREAM_API_KEY" not in plan.env


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

    prepare_launch_artifacts(plan, launch_env=launch_env, kit_root=tmp_path)
    rendered = (tmp_path / "litellm.yaml").read_text()
    assert 'api_base: "os.environ/LITELLM_UPSTREAM_BASE_URL"' in rendered
    assert "os.environ/LITELLM_UPSTREAM_API_KEY" in rendered
    assert "sk-upstream-secret" not in rendered


def test_shell_wrappers_delegate_to_python_launcher_without_launching() -> None:
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "agent_bench/serving/vllm/serve.sh"),
            "--model-profile",
            "qwen35_9b",
            "--serving",
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
