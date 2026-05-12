#!/usr/bin/env bash
# scripts/mig/mig_disable.sh — switch a GPU out of MIG mode.
#
# Destroys every compute instance and GPU instance on the target GPU first
# (in that order — CIs are leaves of GIs). Then `-mig 0`.
#
# Usage: sudo bash scripts/mig/mig_disable.sh <gpu_index>

set -uo pipefail

GPU="${1:?usage: $0 <gpu_index>}"

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must run as root (try: sudo $0 $GPU)" >&2
    exit 1
fi

CURRENT="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i "$GPU" | tr -d ' ')"
if [[ "$CURRENT" != "Enabled" ]]; then
    echo "[mig-disable] GPU $GPU is not in MIG mode (current=$CURRENT)."
    exit 0
fi

echo "[mig-disable] destroying compute instances on GPU $GPU ..."
nvidia-smi mig -dci -i "$GPU" 2>&1 | sed 's/^/  /' || true

echo "[mig-disable] destroying GPU instances on GPU $GPU ..."
nvidia-smi mig -dgi -i "$GPU" 2>&1 | sed 's/^/  /' || true

echo "[mig-disable] disabling MIG mode on GPU $GPU ..."
if ! nvidia-smi -i "$GPU" -mig 0; then
    echo "ERROR: nvidia-smi -mig 0 failed." >&2
    echo "       A GPU reset may be required: sudo nvidia-smi --gpu-reset -i $GPU" >&2
    exit 3
fi

CURRENT="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i "$GPU" | tr -d ' ')"
echo "[mig-disable] GPU $GPU mode now: $CURRENT"
