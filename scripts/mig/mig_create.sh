#!/usr/bin/env bash
# scripts/mig/mig_create.sh — create one MIG instance (GI + CI) on a GPU.
#
# Wraps `nvidia-smi mig -cgi <profile> -C` which creates a GPU instance and
# a compute instance with the default CI profile in one shot. Prints the
# new slice's MIG-UUID to stdout (suitable for piping into VLLM_MIG_UUID).
#
# Usage: sudo bash scripts/mig/mig_create.sh <gpu_index> <profile>
#   <profile> can be a name like "1g.24gb" / "1g.12gb" / "2g.24gb" / "4g.47gb"
#   or the numeric profile id (look up with `nvidia-smi mig -lgip`).

set -uo pipefail

GPU="${1:?usage: $0 <gpu_index> <profile>}"
PROFILE="${2:?usage: $0 <gpu_index> <profile>}"

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must run as root (try: sudo $0 $GPU $PROFILE)" >&2
    exit 1
fi

CURRENT="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i "$GPU" | tr -d ' ')"
if [[ "$CURRENT" != "Enabled" ]]; then
    echo "ERROR: GPU $GPU is not in MIG mode. Enable first:" >&2
    echo "  sudo bash scripts/mig/mig_enable.sh $GPU" >&2
    exit 2
fi

# Snapshot UUIDs before creation so we can find the new one afterwards.
SNAP_BEFORE="$(nvidia-smi -L 2>/dev/null | grep -oE 'MIG-[0-9a-fA-F-]+' | sort -u || true)"

echo "[mig-create] creating $PROFILE on GPU $GPU ..."
if ! nvidia-smi mig -cgi "$PROFILE" -C -i "$GPU" >&2; then
    echo "ERROR: nvidia-smi mig -cgi failed." >&2
    echo "       List available profiles: nvidia-smi mig -lgip -i $GPU" >&2
    exit 3
fi

SNAP_AFTER="$(nvidia-smi -L 2>/dev/null | grep -oE 'MIG-[0-9a-fA-F-]+' | sort -u || true)"
NEW_UUID="$(comm -13 <(echo "$SNAP_BEFORE") <(echo "$SNAP_AFTER") | head -1)"

if [[ -z "$NEW_UUID" ]]; then
    echo "WARN: could not pinpoint new slice UUID. Run 'make mig-status' to list." >&2
    exit 0
fi

echo "[mig-create] new slice UUID: $NEW_UUID" >&2
echo "$NEW_UUID"
