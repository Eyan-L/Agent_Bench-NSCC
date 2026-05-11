#!/usr/bin/env bash
# scripts/mig/mig_destroy.sh — destroy one MIG slice by UUID.
#
# Reverse-looks-up the GI id that owns the given slice UUID and destroys it
# (which cascades to its compute instances). Idempotent: a missing UUID
# exits 0 with a note.
#
# Usage: sudo bash scripts/mig/mig_destroy.sh <slice_uuid>
#   slice_uuid is the full "MIG-..." string from `mig_status.sh` or
#   `mig_create.sh`'s stdout. Bare UUIDs (no MIG- prefix) are accepted too.

set -uo pipefail

UUID_IN="${1:?usage: $0 <slice_uuid>}"

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must run as root (try: sudo $0 $UUID_IN)" >&2
    exit 1
fi

# Normalize.
UUID="${UUID_IN#MIG-}"
FULL="MIG-${UUID}"

if ! nvidia-smi -L | grep -qF "$FULL"; then
    echo "[mig-destroy] $FULL not found — already destroyed?"
    exit 0
fi

# Find which GPU index this slice belongs to. `nvidia-smi -L` groups by GPU.
GPU="$(nvidia-smi -L | awk -v u="$FULL" '
    /^GPU [0-9]+:/ { match($0, /GPU ([0-9]+)/, m); gpu=m[1]; next }
    $0 ~ u { print gpu; exit }
')"

# Find the GI id by parsing `nvidia-smi mig -lgi`. The output layout is
# GPU column + Name + Profile id + Instance id (the "GI id" we need).
GI_ID="$(nvidia-smi -L | awk -v u="$FULL" '
    /MIG/ && /UUID: '"$UUID"'/ {
        # Line shape: MIG 1g.24gb Device <local>: (UUID: MIG-...)
        match($0, /Device +([0-9]+)/, m); print m[1]; exit
    }
')"

if [[ -z "$GI_ID" ]]; then
    echo "ERROR: could not extract GI id for $FULL" >&2
    exit 3
fi

echo "[mig-destroy] GPU $GPU, GI $GI_ID -> $FULL"

# Destroy compute instances first, then the GI itself.
nvidia-smi mig -dci -gi "$GI_ID" -i "$GPU" 2>&1 | sed 's/^/  /' || true
nvidia-smi mig -dgi -gi "$GI_ID" -i "$GPU" 2>&1 | sed 's/^/  /' || true

if nvidia-smi -L | grep -qF "$FULL"; then
    echo "ERROR: $FULL still present after destroy attempt." >&2
    exit 4
fi
echo "[mig-destroy] done."
