#!/usr/bin/env bash
# backend/vllm/serve.sh — start a vLLM endpoint for the OpenClaw study.
#
# Reads its defaults from two JSON config axes:
#   MODEL config   (model identity + model-level limits)
#     - AB_MODEL_CONFIG / VLLM_MODEL_CONFIG explicit path; or
#     - MODEL_PROFILE resolved to $KIT_ROOT/config/model/$MODEL_PROFILE.json.
#   SERVING config (engine/port/scheduler/preset)
#     - AB_SERVING_CONFIG / VLLM_SERVING_CONFIG explicit path; or
#     - SERVING resolved to $KIT_ROOT/config/serving/$SERVING.json.
# Any value can still be overridden with a matching env var:
#   MODEL TP PORT MAX_MODEL_LEN GPU_MEM_UTIL CUDA_VISIBLE_DEVICES
# Env override wins over the config; config wins over the hardcoded fallbacks.
#
# Usage:
#   serve_vllm.sh                         # preset from config (or "default")
#   serve_vllm.sh --preset latency        # explicit preset
#   serve_vllm.sh --preset throughput
#   serve_vllm.sh --preset context
#
# Other env:
#   HF_HOME               (default: ~/.cache/huggingface)
#   KIT_ROOT              (default: repo root inferred from this script)

set -euo pipefail

# Resolve repo root so we can find the config files even when invoked from
# somewhere unusual.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# 1) Model config: MODEL + parser hints.
MODEL_CFG="${VLLM_MODEL_CONFIG:-${AB_MODEL_CONFIG:-}}"
if [[ -z "$MODEL_CFG" && -n "${MODEL_PROFILE:-}" ]]; then
    MODEL_CFG="$KIT_ROOT/config/model/${MODEL_PROFILE}.json"
fi
if [[ -n "$MODEL_CFG" && -f "$MODEL_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_model_env.py" "$MODEL_CFG")"
elif [[ -n "$MODEL_CFG" ]]; then
    echo "WARN: model config not found: $MODEL_CFG — relying on env/fallbacks" >&2
fi

# 2) Serving config: engine runtime knobs + serving preset.
SERVING_CFG="${VLLM_SERVING_CONFIG:-${AB_SERVING_CONFIG:-}}"
if [[ -z "$SERVING_CFG" && -n "${SERVING:-}" ]]; then
    SERVING_CFG="$KIT_ROOT/config/serving/${SERVING}.json"
fi
if [[ -n "$SERVING_CFG" && -f "$SERVING_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_serving_env.py" "$SERVING_CFG")"
elif [[ -n "$SERVING_CFG" ]]; then
    echo "WARN: serving config not found: $SERVING_CFG — using built-in defaults" >&2
fi

PRESET="${SERVING_PRESET:-default}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MODEL_CONTEXT_WINDOW:-131072}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)         PRESET="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---- Pre-flight checks ---------------------------------------------
# Auto-activate a conda env if vllm isn't on PATH and VLLM_CONDA_ENV is set.
# Lets `make run_phase*` work without the user pre-activating the env.
if ! command -v vllm >/dev/null 2>&1 && [[ -n "${VLLM_CONDA_ENV:-}" ]]; then
    for conda_sh in \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            conda activate "$VLLM_CONDA_ENV" 2>/dev/null && break
        fi
    done
fi
if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: 'vllm' not on PATH. Install with: pip install 'vllm>=0.19.0'" >&2
    echo "  (or set VLLM_CONDA_ENV=<env-name> if vllm lives in a conda env)" >&2
    exit 1
fi

VLLM_VER="$(vllm --version 2>&1 | head -n1 | awk '{print $NF}' || echo 0.0.0)"
echo "[serve_vllm] vLLM version: $VLLM_VER (need ≥0.19.0 for Qwen3.6)"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
    echo "[serve_vllm] Detected $GPU_COUNT GPU(s); using TP=$TP"
    if [[ "$GPU_COUNT" -lt "$TP" ]]; then
        echo "ERROR: TP=$TP but only $GPU_COUNT GPU(s) visible." >&2
        exit 1
    fi
else
    echo "WARN: nvidia-smi not found — proceeding but vLLM will fail if no GPU." >&2
fi

# ---- Base flags (shared across presets) ----------------------------
# Tool calling + reasoning parser are mandatory for OpenClaw agent use —
# without them, tool calls won't parse and thinking tokens bleed into tool args.
BASE_FLAGS=(
    "$MODEL"
    --host "$BIND_HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enable-auto-tool-choice
    --trust-remote-code
    # Fix for Mamba cache block count < default max_num_seqs on Qwen3.6 GDN layers.
    # Each preset overrides --max-num-seqs below; 512 is a safe ceiling.
)
if [[ -n "$REASONING_PARSER" ]]; then
    BASE_FLAGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
    BASE_FLAGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi

# ---- Preset-specific flags -----------------------------------------
# Non-default presets map to serving-experiment sweep points.
# Document them so config_id in manifests matches what was actually served.

declare -a PRESET_FLAGS
CONFIG_ID=""

case "$PRESET" in
    default)
        # Phase 0/1 baseline — balanced, before RQ3 sweep.
        CONFIG_ID="${SERVING_CONFIG_ID:-vllm-default}"
        PRESET_FLAGS=(
            --max-num-seqs "${MAX_NUM_SEQS:-128}"
        )
        [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]] && PRESET_FLAGS+=(--enable-prefix-caching)
        ;;

    latency)
        # RQ3 point 1: minimize TTFT/TPOT per request. Low concurrency, small batch.
        # Prefix caching still on — agent reuses system prompt across steps.
        CONFIG_ID="vllm-latency"
        PRESET_FLAGS=(
            --max-num-seqs 8
            --enable-prefix-caching
            --scheduling-policy priority
        )
        ;;

    throughput)
        # RQ3 point 2: maximize aggregate tokens/sec. High concurrency.
        # Expert parallelism on top of tensor parallelism for MoE.
        CONFIG_ID="vllm-throughput"
        PRESET_FLAGS=(
            --max-num-seqs 256
            --enable-prefix-caching
            --enable-expert-parallel
        )
        ;;

    context)
        # RQ3 point 3: optimize for repeated prefixes and long context.
        # FP8 KV cache roughly doubles tokens-per-byte, trading a small quality
        # cost for much more cache. If you want a zero-confound sweep, use
        # 'auto' instead; fp8 is the practical choice on 2×80GB.
        CONFIG_ID="vllm-context"
        PRESET_FLAGS=(
            --max-num-seqs 32
            --enable-prefix-caching
            --kv-cache-dtype fp8
        )
        ;;

    *)
        echo "Unknown preset: $PRESET (use one of: default, latency, throughput, context)" >&2
        exit 2 ;;
esac

echo "[serve_vllm] Preset: $PRESET"
echo "[serve_vllm] config_id (use this in manifests): $CONFIG_ID"
echo "[serve_vllm] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (TP=$TP)"
echo "[serve_vllm] Endpoint: http://localhost:$PORT/v1"
echo "[serve_vllm] Full command:"
printf "  vllm serve"
printf " %q" "${BASE_FLAGS[@]}" "${PRESET_FLAGS[@]}"
printf "\n\n"

# Persist the config_id so the experiment driver can read it.
mkdir -p "${KIT_ROOT:-$(pwd)}/data"
echo "$CONFIG_ID" > "${KIT_ROOT:-$(pwd)}/data/serving_config_id"

# If VLLM_LOG_FILE is set, redirect engine output into the experiment's
# results folder. Callers (e.g. scripts/e2e_smoke.sh) set this to
# <sweep_output_dir>/serving/vllm.log so per-sweep telemetry lands next to
# the runs.
if [[ -n "${VLLM_LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname "$VLLM_LOG_FILE")"
    echo "[serve_vllm] log file: $VLLM_LOG_FILE"
    exec vllm serve "${BASE_FLAGS[@]}" "${PRESET_FLAGS[@]}" >>"$VLLM_LOG_FILE" 2>&1
fi

exec vllm serve "${BASE_FLAGS[@]}" "${PRESET_FLAGS[@]}"
