#!/usr/bin/env bash
# scripts/mig/mig_status.sh — pretty-print current MIG state.
#
# Combines `nvidia-smi -L` + `nvidia-smi mig -lgi -lci` so you can see
# which slices exist + their UUIDs + which compute-instance lives under
# which GPU-instance, all in one go.
#
# Usage: bash scripts/mig/mig_status.sh

set -uo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not on PATH" >&2
    exit 1
fi

echo "=== GPUs (mig mode) ==="
nvidia-smi --query-gpu=index,name,mig.mode.current --format=csv

echo
echo "=== Slice UUIDs (for VLLM_MIG_UUID) ==="
# Filter the `nvidia-smi -L` output for lines describing MIG instances.
# Each MIG line looks like:
#   MIG 1g.24gb     Device  0: (UUID: MIG-abcd...)
nvidia-smi -L 2>/dev/null | awk '
    /^GPU [0-9]+:/ { gpu=$0; next }
    /MIG/ {
        # Extract profile (e.g. "1g.24gb") and UUID (MIG-...)
        prof = $0
        sub(/.*MIG /, "", prof); sub(/ +Device.*/, "", prof)
        uuid = $0
        sub(/.*UUID: /, "", uuid); sub(/\)$/, "", uuid)
        # Indent under the parent GPU line for readability.
        if (gpu) { print "  on " gpu; gpu="" }
        printf "    %-12s %s\n", prof, uuid
    }
' || true

if ! nvidia-smi -L 2>/dev/null | grep -q "MIG-"; then
    echo "  (no MIG slices — MIG is disabled or no instances created)"
fi

echo
echo "=== GPU instances (-lgi) ==="
nvidia-smi mig -lgi 2>/dev/null | grep -v "^$" || echo "  (none)"

echo
echo "=== Compute instances (-lci) ==="
nvidia-smi mig -lci 2>/dev/null | grep -v "^$" || echo "  (none)"

echo
echo "Supported profiles on this hardware (-lgip):"
nvidia-smi mig -lgip 2>/dev/null | awk '/MIG/ && /Name/' || nvidia-smi mig -lgip 2>/dev/null | grep -E "^\|.*[0-9]g\." | head -8
