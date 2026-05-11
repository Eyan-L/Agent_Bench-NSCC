#!/usr/bin/env bash
# scripts/mig/mig_destroy_all.sh — destroy every MIG slice on a GPU.
#
# Convenience wrapper around `nvidia-smi mig -dci` then `-dgi` for one GPU.
# Useful before mig_disable.sh, or to "reset" a GPU's slice layout.
#
# Usage: sudo bash scripts/mig/mig_destroy_all.sh <gpu_index>

set -uo pipefail

GPU="${1:?usage: $0 <gpu_index>}"

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must run as root." >&2
    exit 1
fi

CURRENT="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i "$GPU" | tr -d ' ')"
if [[ "$CURRENT" != "Enabled" ]]; then
    echo "[mig-destroy-all] GPU $GPU not in MIG mode — nothing to do."
    exit 0
fi

echo "[mig-destroy-all] destroying CIs ..."
nvidia-smi mig -dci -i "$GPU" 2>&1 | sed 's/^/  /' || true
echo "[mig-destroy-all] destroying GIs ..."
nvidia-smi mig -dgi -i "$GPU" 2>&1 | sed 's/^/  /' || true
echo "[mig-destroy-all] done."
