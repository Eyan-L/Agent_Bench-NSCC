#!/usr/bin/env bash
# backend/vllm/serve.sh — start a vLLM endpoint for the OpenClaw study.
#
# Reads its defaults from two JSON configs (composed-config model):
#   HARNESS config (model selection)
#     - HARNESS env var, resolved to $KIT_ROOT/config/harness/$HARNESS.json; or
#     - VLLM_HARNESS_YAML for an explicit path (var name kept for back-compat).
#   BACKEND config (engine knobs: TP/port/gpu_mem/max_len/cuda_devices/presets)
#     - BACKEND env var, resolved to $KIT_ROOT/config/backend/$BACKEND.json; or
#     - VLLM_BACKEND_YAML for an explicit path (var name kept for back-compat).
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

# 1) Harness config: MODEL + MODEL_ENDPOINT (single source of truth).
HARNESS_CFG="${VLLM_HARNESS_YAML:-}"
if [[ -z "$HARNESS_CFG" && -n "${HARNESS:-}" ]]; then
    HARNESS_CFG="$KIT_ROOT/config/harness/${HARNESS}.json"
fi
if [[ -n "$HARNESS_CFG" && -f "$HARNESS_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_harness_env.py" "$HARNESS_CFG")"
elif [[ -n "$HARNESS_CFG" ]]; then
    echo "WARN: harness config not found: $HARNESS_CFG — relying on env/fallbacks" >&2
fi

# 2) Backend config: engine runtime knobs + first serving preset.
BACKEND_CFG="${VLLM_BACKEND_YAML:-}"
if [[ -z "$BACKEND_CFG" && -n "${BACKEND:-}" ]]; then
    BACKEND_CFG="$KIT_ROOT/config/backend/${BACKEND}.json"
fi
if [[ -n "$BACKEND_CFG" && -f "$BACKEND_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_backend_env.py" "$BACKEND_CFG")"
elif [[ -n "$BACKEND_CFG" ]]; then
    echo "WARN: backend config not found: $BACKEND_CFG — using built-in defaults" >&2
fi

PRESET="${SERVING_PRESET:-default}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

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
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --trust-remote-code
    # Fix for Mamba cache block count < default max_num_seqs on Qwen3.6 GDN layers.
    # Each preset overrides --max-num-seqs below; 512 is a safe ceiling.
)

# ---- Preset-specific flags -----------------------------------------
# These three map directly to the RQ3 sweep points in config/backend/vllm/tp2_rq3.json.
# Document them so config_id in manifests matches what was actually served.

declare -a PRESET_FLAGS
CONFIG_ID=""

case "$PRESET" in
    default)
        # Phase 0/1 baseline — balanced, before RQ3 sweep.
        CONFIG_ID="vllm-default"
        PRESET_FLAGS=(
            --max-num-seqs 128
            --enable-prefix-caching
        )
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
