"""Resolve OpenAI-compatible endpoint URL for the active backend.

Both vLLM and sglang expose `/v1` chat/completions. Eval and stress code only
need the URL string; this helper centralizes the env contract so neither side
hardcodes ports.
"""
from __future__ import annotations

import os


def resolve_endpoint(backend: str | None = None) -> str:
    backend = backend or os.environ.get("SERVING_BACKEND")
    if not backend:
        raise RuntimeError("SERVING_BACKEND not set and no backend argument provided")

    if backend == "vllm":
        host = os.environ.get("VLLM_HOST", "127.0.0.1")
        port = os.environ.get("VLLM_PORT", "8000")
    elif backend == "sglang":
        host = os.environ.get("SGLANG_HOST", "127.0.0.1")
        port = os.environ.get("SGLANG_PORT", "30000")
    else:
        raise ValueError(f"unsupported backend: {backend}")

    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{port}/v1"
