#!/usr/bin/env bash
# Pull the immutable OpenClaw v2026.9.2 container into a local Singularity SIF
# and immediately run the read-only migration audit.

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER_NAME}"
IMAGE="docker://ghcr.io/openclaw/openclaw:2026.9.2"
OUT="$SCRATCH/sif_images/openclaw_2026.9.2_base.sif"
FORCE=0
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --repo-root) AB_REPO_ROOT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
command -v singularity >/dev/null 2>&1 || {
    echo "ERROR: singularity not on PATH" >&2
    exit 2
}

AUDIT="$AB_REPO_ROOT/agent_bench/deployments/pbs/audit_oc_v2026_9_2.sh"
[[ -x "$AUDIT" || -f "$AUDIT" ]] || {
    echo "ERROR: audit script not found: $AUDIT" >&2
    exit 2
}

mkdir -p "$(dirname "$OUT")"

if [[ -f "$OUT" && "$FORCE" -eq 0 ]]; then
    echo "[prepare-oc-9.2] reuse existing base: $OUT"
else
    TMP="${OUT}.new"
    rm -f "$TMP"
    echo "[prepare-oc-9.2] pulling $IMAGE"
    singularity pull "$TMP" "$IMAGE"
    [[ -s "$TMP" ]] || {
        echo "ERROR: singularity pull produced no SIF: $TMP" >&2
        exit 1
    }
    mv "$TMP" "$OUT"
fi

echo "[prepare-oc-9.2] base ready"
ls -lh "$OUT"
sha256sum "$OUT"

echo "[prepare-oc-9.2] running migration audit"
bash "$AUDIT" "$OUT"

echo "[prepare-oc-9.2] PASS"
