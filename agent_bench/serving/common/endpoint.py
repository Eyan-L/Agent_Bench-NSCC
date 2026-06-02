"""Resolve OpenAI-compatible endpoint URL for the active Agent Bench serving backend.

vLLM, SGLang, and LiteLLM all expose `/v1` chat/completions locally. Eval and
stress code only need the URL string; this helper centralizes the env contract
so neither side hardcodes ports.
"""
from __future__ import annotations

import os


def resolve_endpoint(backend: str | None = None) -> str:
    backend = backend or os.environ.get("AB_SERVING_BACKEND")
    if not backend:
        raise RuntimeError("AB_SERVING_BACKEND not set and no backend argument provided")

    if backend in {"existing_endpoint", "openai_compatible"}:
        endpoint = os.environ.get("AB_SERVING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not endpoint:
            raise RuntimeError("AB_SERVING_BASE_URL or OPENAI_BASE_URL must be set for existing endpoint serving")
        return endpoint.rstrip("/")

    if backend == "vllm":
        host = os.environ.get("VLLM_HOST", "127.0.0.1")
        port = os.environ.get("VLLM_PORT", "8000")
    elif backend == "sglang":
        host = os.environ.get("SGLANG_HOST", "127.0.0.1")
        port = os.environ.get("SGLANG_PORT", "30000")
    elif backend == "litellm":
        host = os.environ.get("LITELLM_HOST", os.environ.get("AB_SERVING_CLIENT_HOST", "127.0.0.1"))
        port = os.environ.get("LITELLM_PORT", os.environ.get("AB_SERVING_PORT", "8000"))
    else:
        raise ValueError(f"unsupported backend: {backend}")

    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{port}/v1"
