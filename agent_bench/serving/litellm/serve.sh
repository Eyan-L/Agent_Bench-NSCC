#!/usr/bin/env bash
# Start a local LiteLLM OpenAI-compatible proxy for provider-backed serving.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

MODEL_CFG="${AB_MODEL_CONFIG:-}"
if [[ -z "$MODEL_CFG" && -n "${MODEL_PROFILE:-}" ]]; then
    MODEL_CFG="$KIT_ROOT/profiles/models/${MODEL_PROFILE}.json"
fi
if [[ -n "$MODEL_CFG" && -f "$MODEL_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_model_env.py" "$MODEL_CFG")"
elif [[ -n "$MODEL_CFG" ]]; then
    echo "WARN: model config not found: $MODEL_CFG - relying on env/fallbacks" >&2
fi

SERVING_CFG="${AB_SERVING_CONFIG:-}"
if [[ -z "$SERVING_CFG" && -n "${SERVING:-}" ]]; then
    SERVING_CFG="$KIT_ROOT/profiles/serving/${SERVING}"
    [[ -f "$SERVING_CFG" ]] || SERVING_CFG="$KIT_ROOT/profiles/serving/${SERVING}.json"
    [[ -f "$SERVING_CFG" ]] || SERVING_CFG="$KIT_ROOT/profiles/serving/${SERVING}.toml"
fi
if [[ -n "$SERVING_CFG" && -f "$SERVING_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_serving_env.py" "$SERVING_CFG")"
elif [[ -n "$SERVING_CFG" ]]; then
    echo "WARN: serving config not found: $SERVING_CFG - relying on env/fallbacks" >&2
fi

if ! command -v litellm >/dev/null 2>&1; then
    echo "ERROR: 'litellm' not on PATH. Install the LiteLLM proxy package in this environment." >&2
    exit 1
fi

export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-${LITELLM_PROXY_API_KEY:-sk-agent-bench-local}}"
export LITELLM_MODEL_ALIAS="${LITELLM_MODEL_ALIAS:-${MODEL:-}}"
export LITELLM_UPSTREAM_MODEL="${LITELLM_UPSTREAM_MODEL:-${LITELLM_MODEL:-${MODEL:-}}}"
export LITELLM_UPSTREAM_BASE_URL="${LITELLM_UPSTREAM_BASE_URL:-${SERVING_PROVIDER_BASE_URL:-}}"
export LITELLM_PORT="${LITELLM_PORT:-${PORT:-8000}}"
export LITELLM_NUM_WORKERS="${LITELLM_NUM_WORKERS:-${LITELLM_PROXY_NUM_WORKERS:-4}}"

UPSTREAM_KEY_ENV="${SERVING_PROVIDER_API_KEY_ENV:-}"
if [[ -z "$UPSTREAM_KEY_ENV" ]]; then
    echo "ERROR: SERVING_PROVIDER_API_KEY_ENV is not set by the serving profile." >&2
    exit 2
fi
if [[ -z "${!UPSTREAM_KEY_ENV:-}" ]]; then
    echo "ERROR: upstream API key env ${UPSTREAM_KEY_ENV} is empty." >&2
    exit 2
fi
export LITELLM_UPSTREAM_API_KEY="${!UPSTREAM_KEY_ENV}"

CONFIG_DIR="${LITELLM_CONFIG_DIR:-$KIT_ROOT/data/litellm}"
CONFIG_PATH="${LITELLM_CONFIG_PATH:-$CONFIG_DIR/litellm_config.yaml}"
python3 -m agent_bench.serving.render_litellm_config --output "$CONFIG_PATH"

export MODEL_ENDPOINT="http://127.0.0.1:$LITELLM_PORT/v1"
export OPENAI_BASE_URL="$MODEL_ENDPOINT"
export MODEL_API_KEY="$LITELLM_MASTER_KEY"
export OPENAI_API_KEY="$LITELLM_MASTER_KEY"
export OC_VLLM_API_KEY="$LITELLM_MASTER_KEY"
export HERMES_INFERENCE_PROVIDER="${LITELLM_AGENT_PROVIDER:-custom}"
export NO_TELEMETRY="${NO_TELEMETRY:-1}"

exec litellm --config "$CONFIG_PATH" \
    --host "${BIND_HOST:-0.0.0.0}" \
    --port "$LITELLM_PORT" \
    --num_workers "$LITELLM_NUM_WORKERS"
