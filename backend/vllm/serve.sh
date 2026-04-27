#!/usr/bin/env bash
# backend/vllm/serve.sh — start a vLLM endpoint for the OpenClaw + SWE-bench study.
#
# Designed for Qwen3.6-35B-A3B on 2× H100 80GB (tensor parallel = 2).
# Four presets. The latency / throughput / context presets are the three points
# of the RQ3 serving-config sweep — run all three and compare.
#
# Usage:
#   serve_vllm.sh                         # default preset
#   serve_vllm.sh --preset latency
#   serve_vllm.sh --preset throughput
#   serve_vllm.sh --preset context
#
# Env overrides:
#   MODEL            (default: Qwen/Qwen3.6-35B-A3B)
#   TP               (default: 2)
#   PORT             (default: 8000)
#   MAX_MODEL_LEN    (default: 131072 — Qwen3.6 docs recommend ≥128K)
#   GPU_MEM_UTIL     (default: 0.90)
#   HF_HOME          (default: ~/.cache/huggingface)

set -euo pipefail

PRESET="default"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

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
if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: 'vllm' not on PATH. Install with: pip install 'vllm>=0.19.0'" >&2
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
# These three map directly to the RQ3 sweep points in sweeps/phase2_serving.yaml.
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
echo "[serve_vllm] Endpoint: http://localhost:$PORT/v1"
echo "[serve_vllm] Full command:"
printf "  vllm serve"
printf " %q" "${BASE_FLAGS[@]}" "${PRESET_FLAGS[@]}"
printf "\n\n"

# Persist the config_id so the experiment driver can read it.
mkdir -p "${KIT_ROOT:-$(pwd)}/data"
echo "$CONFIG_ID" > "${KIT_ROOT:-$(pwd)}/data/serving_config_id"

exec vllm serve "${BASE_FLAGS[@]}" "${PRESET_FLAGS[@]}"
