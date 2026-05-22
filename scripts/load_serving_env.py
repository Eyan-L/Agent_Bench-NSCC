#!/usr/bin/env python3
"""Emit shell assignments for a config/serving/<backend>/*.json file."""

import json
import os
import sys
from pathlib import Path


def _shell_quote(v):
    s = os.path.expandvars(str(v))
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _emit_default(var, val):
    print(f'if [ -z "${{{var}:-}}" ]; then {var}={_shell_quote(val)}; fi')
    print(f"export {var}")


def _as_endpoint(value):
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
        port = value.get("port")
        path = value.get("path", "/v1")
        if not str(path).startswith("/"):
            path = "/" + str(path)
        if port is None:
            port = 443 if scheme == "https" else 80
        return f"{scheme}://{host}:{port}{path}"
    return None


def _active_provider(serving):
    provider = serving.get("provider") or {}
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


def _endpoint(serving):
    ep = serving.get("endpoint") or {}
    endpoint = _as_endpoint(ep)
    if endpoint:
        return endpoint
    scheme = ep.get("scheme", "http")
    host = ep.get("client_host") or ep.get("host") or "127.0.0.1"
    port = ep.get("port")
    path = ep.get("path", "/v1")
    if not str(path).startswith("/"):
        path = "/" + str(path)
    if port is None:
        default_port = 443 if scheme == "https" else 80
        port = default_port
    return f"{scheme}://{host}:{port}{path}"


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <serving.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"serving config not found: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text()) or {}
    serving = data.get("serving") or {}
    if not isinstance(serving, dict):
        print(f"serving config {path} is missing a `serving` object", file=sys.stderr)
        return 1

    endpoint = serving.get("endpoint") or {}
    endpoint_obj = endpoint if isinstance(endpoint, dict) else {}
    runtime = serving.get("runtime") or {}
    scheduler = serving.get("scheduler") or {}
    proxy = serving.get("proxy") or {}
    provider = _active_provider(serving)
    provider_name = provider.get("name") or provider.get("provider")
    provider_api_key_env = provider.get("api_key_env")
    backend = serving.get("backend")
    if backend in {"provider", "api"}:
        print(
            f"serving backend {backend!r} is no longer supported; "
            "use backend 'litellm' for provider-backed models",
            file=sys.stderr,
        )
        return 1
    proxy_api_key = proxy.get("api_key", "sk-agent-bench-local")
    proxy_agent_provider = proxy.get("agent_provider")

    values = {
        "SERVING_CONFIG_ID": serving.get("config_id"),
        "SERVING_BACKEND": backend,
        "SERVING_PRESET": serving.get("preset", "default"),
        "BIND_HOST": endpoint_obj.get("bind_host", endpoint_obj.get("host", "0.0.0.0")),
        "CLIENT_HOST": endpoint_obj.get("client_host", "127.0.0.1"),
        "PORT": endpoint_obj.get("port", runtime.get("port", 8000)),
        "MODEL_ENDPOINT": _endpoint(serving),
        "GPU_MEM_UTIL": runtime.get("gpu_mem_util", 0.90),
        "MAX_NUM_SEQS": scheduler.get("max_num_seqs", 128),
        "ENABLE_PREFIX_CACHING": "1" if scheduler.get("enable_prefix_caching", True) else "0",
        "SERVING_IMAGE": serving.get("image"),
        "SERVING_SIF": serving.get("sif"),
        "SERVING_PROVIDER": provider_name,
        "SERVING_PROVIDER_NAME": provider_name,
        "SERVING_PROVIDER_BASE_URL": provider.get("base_url") or provider.get("url"),
        "SERVING_PROVIDER_INTERNAL_ENDPOINT": _endpoint(serving),
        "SERVING_PROVIDER_API_KEY_ENV": provider_api_key_env,
        "LITELLM_PROXY_API_KEY": proxy_api_key if backend == "litellm" else None,
        "LITELLM_PROXY_NUM_WORKERS": proxy.get("num_workers"),
        "LITELLM_UPSTREAM_MODEL": provider.get("model"),
        "LITELLM_AGENT_PROVIDER": proxy_agent_provider,
    }

    for var, val in values.items():
        if val is None or val == "":
            continue
        _emit_default(var, val)
    if backend == "litellm":
        _emit_default("MODEL_API_KEY", proxy_api_key)
        _emit_default("OPENAI_API_KEY", proxy_api_key)
        _emit_default("OC_VLLM_API_KEY", proxy_api_key)
        if proxy_agent_provider:
            print(f"MODEL_PROVIDER={_shell_quote(proxy_agent_provider)}")
            print("export MODEL_PROVIDER")
            print(f"HERMES_INFERENCE_PROVIDER={_shell_quote(proxy_agent_provider)}")
            print("export HERMES_INFERENCE_PROVIDER")
    _emit_default("OPENAI_BASE_URL", _endpoint(serving))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
