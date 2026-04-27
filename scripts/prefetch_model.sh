#!/usr/bin/env bash
# scripts/prefetch_model.sh — pre-download model weights.
#
# vllm serve will download on first run, but that makes the first 'make run-phase0'
# take 10-20 minutes of silent 'waiting' before anything happens. Prefetch once.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[prefetch] Model: $MODEL"
echo "[prefetch] Cache: $HF_HOME"
echo "[prefetch] Expected size: ~70 GB (bf16 weights)"
echo

if ! command -v huggingface-cli >/dev/null 2>&1 && ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Install with: pip install 'huggingface_hub[cli]'" >&2
    exit 1
fi

# Prefer the CLI if available, fall back to python module.
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL" --cache-dir "$HF_HOME"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL', cache_dir='$HF_HOME')
"
fi

echo
echo "[prefetch] Done. vllm serve will now load from cache instantly."
