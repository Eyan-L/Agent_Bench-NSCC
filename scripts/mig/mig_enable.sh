#!/usr/bin/env bash
# scripts/mig/mig_enable.sh — switch a GPU into MIG mode.
#
# MIG mode can ONLY be changed when no CUDA processes are using the GPU,
# and the driver may require a full GPU reset (`nvidia-smi --gpu-reset`)
# after toggling. Requires sudo / root.
#
# Usage: sudo bash scripts/mig/mig_enable.sh <gpu_index>

set -uo pipefail

GPU="${1:?usage: $0 <gpu_index>}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not on PATH" >&2; exit 1
fi
if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must run as root (try: sudo $0 $GPU)" >&2
    exit 1
fi

# Refuse if anything is using the GPU.
USERS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d '[:space:]')"
if [[ -n "$USERS" ]]; then
    echo "ERROR: GPU $GPU has running CUDA processes:" >&2
    nvidia-smi --query-compute-apps=pid,process_name --format=csv -i "$GPU" >&2
    echo "Stop them before enabling MIG." >&2
    exit 2
fi

CURRENT="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i "$GPU" | tr -d ' ')"
if [[ "$CURRENT" == "Enabled" ]]; then
    echo "[mig-enable] GPU $GPU already in MIG mode — nothing to do."
    exit 0
fi

echo "[mig-enable] enabling MIG on GPU $GPU ..."
if ! nvidia-smi -i "$GPU" -mig 1; then
    echo "ERROR: nvidia-smi -mig 1 failed." >&2
    echo "       If the message says 'reset required', run: sudo nvidia-smi --gpu-reset -i $GPU" >&2
    exit 3
fi

# Verify the switch took effect.
CURRENT="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i "$GPU" | tr -d ' ')"
echo "[mig-enable] GPU $GPU mode now: $CURRENT"
[[ "$CURRENT" == "Enabled" ]] || exit 4
