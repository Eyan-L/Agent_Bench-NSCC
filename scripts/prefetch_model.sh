#!/usr/bin/env bash
# scripts/prefetch_model.sh — pre-download model weights.
#
# vllm serve will download on first run, but that makes the first 'make run_phase0'
# take 10-20 minutes of silent 'waiting' before anything happens. Prefetch once.
#
# Resolves $MODEL from the active harness config (HARNESS env or
# VLLM_HARNESS_YAML) — the single source of truth for "which model this
# run uses", same as serve.sh. Pass MODEL=foo to override.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

HARNESS_CFG="${VLLM_HARNESS_YAML:-}"
if [[ -z "$HARNESS_CFG" && -n "${HARNESS:-}" ]]; then
    HARNESS_CFG="$KIT_ROOT/config/harness/${HARNESS}.json"
fi
if [[ -z "$HARNESS_CFG" ]]; then
    HARNESS_CFG="$KIT_ROOT/config/harness/openclaw_27b.json"
fi
if [[ -f "$HARNESS_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_harness_env.py" "$HARNESS_CFG")"
fi

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[prefetch] Model: $MODEL"
echo "[prefetch] Cache: $HF_HOME"
echo "[prefetch] Expected size: ~70 GB (bf16 weights)"
echo

if ! command -v huggingface-cli >/dev/null 2>&1 && ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Install with: pip install 'huggingface_hub[cli]'" >&2
    exit 1
fi

# HF stores weights at $cache_dir/models--... so pass $HF_HOME/hub explicitly —
# without /hub, snapshot_download lands at $HF_HOME/models--... which vLLM
# (which honors HF_HOME and looks under hub/) won't see.
HF_HUB="${HF_HOME%/}/hub"
mkdir -p "$HF_HUB"
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL" --cache-dir "$HF_HUB"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL', cache_dir='$HF_HUB')
"
fi

echo
echo "[prefetch] Done. vllm serve will now load from cache instantly."
