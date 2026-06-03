from pathlib import Path

import pytest

from agent_bench.core.config import runtime_endpoint_for_container

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_endpoint_for_container_rewrites_serving_env_loopback():
    endpoint = runtime_endpoint_for_container(
        "http://default-host:9000/v1",
        "vllm-host",
        {"AB_SERVING_BASE_URL": "http://127.0.0.1:8000/v1"},
    )

    assert endpoint == "http://vllm-host:8000/v1"


def test_runtime_endpoint_for_container_prefers_runtime_openai_base_url():
    endpoint = runtime_endpoint_for_container(
        "http://default-host:9000/v1",
        "vllm-host",
        {"OPENAI_BASE_URL": "http://localhost:8123/v1"},
    )

    assert endpoint == "http://vllm-host:8123/v1"


def test_runtime_endpoint_for_container_leaves_external_url_alone():
    endpoint = runtime_endpoint_for_container(
        "https://api.example.com/v1",
        "vllm-host",
        {},
    )

    assert endpoint == "https://api.example.com/v1"


@pytest.mark.parametrize(
    "rel",
    [
        "agent_bench/benchmarks/terminal_bench/app.py",
        "agent_bench/benchmarks/deepresearch_bench/app.py",
        "agent_bench/benchmarks/longbench/app.py",
    ],
)
def test_openclaw_docker_benchmarks_rewrite_runtime_endpoint_for_container(rel):
    source = (ROOT / rel).read_text()

    assert "runtime_endpoint_for_container(" in source
    assert "vllm-host:host-gateway" in source
