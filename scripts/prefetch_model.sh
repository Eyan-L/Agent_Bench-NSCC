#!/usr/bin/env bash
# scripts/prefetch_model.sh — pre-download model weights.
#
# vllm serve will download on first run, but that makes the first 'make run_phase0'
# take 10-20 minutes of silent 'waiting' before anything happens. Prefetch once.
#
# Resolves AB_MODEL_NAME from AB_MODEL_CONFIG, matching the serving launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

MODEL_CFG="${AB_MODEL_CONFIG:-}"
if [[ -z "$MODEL_CFG" && -n "${AB_MODEL_CONFIG:-}" ]]; then
    MODEL_CFG="$AB_REPO_ROOT/profiles/models/${AB_MODEL_CONFIG}.json"
fi
if [[ -z "$MODEL_CFG" ]]; then
    MODEL_CFG="$AB_REPO_ROOT/profiles/models/qwen36_27b.json"
fi
if [[ -f "$MODEL_CFG" ]]; then
    eval "$(python3 "$AB_REPO_ROOT/scripts/load_model_env.py" "$MODEL_CFG")"
fi

AB_MODEL_NAME="${AB_MODEL_NAME:-Qwen/Qwen3.6-27B}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[prefetch] Model: $AB_MODEL_NAME"
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
    huggingface-cli download "$AB_MODEL_NAME" --cache-dir "$HF_HUB"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$AB_MODEL_NAME', cache_dir='$HF_HUB')
"
fi

echo
echo "[prefetch] Done. vllm serve will now load from cache instantly."
