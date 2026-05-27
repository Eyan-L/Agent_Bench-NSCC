"""Checks for provider-backed LiteLLM proxy serving profiles."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from agent_bench.core.config import load_model_config, load_serving_config


ROOT = Path(__file__).resolve().parents[1]


def _write_provider_profile(tmp_path: Path) -> Path:
    path = tmp_path / "minimax_provider.json"
    path.write_text(json.dumps({
        "serving": {
            "config_id": "provider-minimax-global-test",
            "backend": "provider",
            "preset": "minimax-global",
            "provider": {
                "active": "minimax",
                "providers": {
                    "minimax": {
                        "name": "minimax",
                        "api": "openai-compatible",
                        "base_url": "https://api.minimax.io/v1",
                        "api_key_env": "MINIMAX_API_KEY",
                        "internal_endpoint": "https://api.minimax.io/v1",
                    },
                },
            },
            "endpoint": {
                "base_url": "https://api.minimax.io/v1",
            },
        },
    }))
    return path


def test_direct_provider_serving_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-minimax")
    profile = _write_provider_profile(tmp_path)

    with pytest.raises(ValueError, match="unsupported backend"):
        load_serving_config(profile)

    proc = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/load_serving_env.py"),
            str(profile),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "no longer supported" in proc.stderr


def test_litellm_serving_hides_upstream_key_from_agents(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-minimax")

    serving = load_serving_config(ROOT / "profiles/serving/litellm/minimax_global.json")
    model = load_model_config(
        ROOT / "profiles/models/minimax_m2_7.json",
        serving=serving,
    )

    assert serving.backend == "litellm"
    assert serving.endpoint == "http://127.0.0.1:8000/v1"
    assert serving.provider_name == "minimax"
    assert serving.proxy_api_key == "sk-agent-bench-local"
    assert model.endpoint == "http://127.0.0.1:8000/v1"
    assert model.api_key == "sk-agent-bench-local"
    assert model.provider == "custom"


def test_litellm_serving_loader_exports_proxy_defaults(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-minimax")
    proc = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/load_serving_env.py"),
            str(ROOT / "profiles/serving/litellm/minimax_global.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SERVING_BACKEND='litellm'" in proc.stdout
    assert "MODEL_ENDPOINT='http://127.0.0.1:8000/v1'" in proc.stdout
    assert "SERVING_PROVIDER_BASE_URL='https://api.minimax.io/v1'" in proc.stdout
    assert "SERVING_PROVIDER_API_KEY_ENV='MINIMAX_API_KEY'" in proc.stdout
    assert "MODEL_API_KEY='sk-agent-bench-local'" in proc.stdout
    assert "OPENAI_API_KEY='sk-agent-bench-local'" in proc.stdout
    assert "OC_VLLM_API_KEY='sk-agent-bench-local'" in proc.stdout
    assert "HERMES_INFERENCE_PROVIDER='custom'" in proc.stdout
    assert "sk-test-minimax" not in proc.stdout


def test_litellm_config_renderer_omits_optional_api_base_and_upstream_secret(tmp_path, monkeypatch):
    out = tmp_path / "litellm.yaml"
    monkeypatch.setenv("MODEL", "gemini/gemini-3.5-flash")
    monkeypatch.setenv("LITELLM_UPSTREAM_MODEL", "gemini/gemini-3.5-flash")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local-proxy")
    monkeypatch.setenv("LITELLM_UPSTREAM_API_KEY", "sk-upstream-secret")
    monkeypatch.delenv("LITELLM_UPSTREAM_BASE_URL", raising=False)

    subprocess.run(
        [
            "python3",
            "-m",
            "agent_bench.serving.render_litellm_config",
            "--output",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )

    rendered = out.read_text()
    assert 'model_name: "gemini/gemini-3.5-flash"' in rendered
    assert "api_base:" not in rendered
    assert "os.environ/LITELLM_UPSTREAM_API_KEY" in rendered
    assert "sk-upstream-secret" not in rendered
    assert oct(os.stat(out).st_mode & 0o777) == "0o600"


def test_litellm_config_renderer_includes_api_base_when_configured(tmp_path, monkeypatch):
    out = tmp_path / "litellm.yaml"
    monkeypatch.setenv("LITELLM_MODEL_ALIAS", "MiniMax-M2.7")
    monkeypatch.setenv("LITELLM_UPSTREAM_MODEL", "openai/MiniMax-M2.7")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-local-proxy")
    monkeypatch.setenv("LITELLM_UPSTREAM_BASE_URL", "https://api.minimax.io/v1")

    subprocess.run(
        [
            "python3",
            "-m",
            "agent_bench.serving.render_litellm_config",
            "--output",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )

    rendered = out.read_text()
    assert 'model_name: "MiniMax-M2.7"' in rendered
    assert 'model_name: "openai/MiniMax-M2.7"' in rendered
    assert 'api_base: "os.environ/LITELLM_UPSTREAM_BASE_URL"' in rendered
    assert "https://api.minimax.io/v1" not in rendered


def test_pbs_uses_serving_stage_for_sidecars():
    pbs = (ROOT / "agent_bench/deployments/pbs/run_benchmark.pbs").read_text()
    tb_pbs = (ROOT / "agent_bench/deployments/pbs/run_benchmark_tb.pbs").read_text()

    assert "STAGE_VLLM" not in pbs
    assert "VLLM_PROFILE" not in pbs
    assert "STAGE_SERVING" in pbs
    assert 'export OC_VLLM_API_KEY="${OC_VLLM_API_KEY:-$MODEL_API_KEY}"' in pbs
    assert "litellm)" in pbs
    assert "provider_proxy" not in pbs
    assert "LITELLM_PORT_AUTO" in pbs
    assert "LITELLM_UPSTREAM_API_KEY" in pbs
    assert 'SERVING_BACKEND=sglang is recognized' in pbs
    assert '! "${SERVING_BACKEND:-vllm}" == "litellm"' not in pbs
    assert "provider|api)" not in pbs
    assert "direct remote endpoint" not in pbs
    assert "provider|api)" not in tb_pbs
    assert "STAGE_VLLM" not in tb_pbs
    assert "STAGE_SERVING" in tb_pbs
