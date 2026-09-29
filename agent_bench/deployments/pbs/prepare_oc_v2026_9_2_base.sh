#!/usr/bin/env bash
# Pull the immutable OpenClaw v2026.9.2 container into a LOCAL Singularity SIF
# and immediately run the read-only migration audit.
#
# Intended workflow:
#   1. Run this script on the user's WSL/local workstation, where GHCR is fast.
#   2. Copy the resulting SIF to NSCC $SCRATCH/sif_images/.
#   3. Run audit_oc_v2026_9_2.sh on NSCC against the uploaded SIF.
#
# The script refuses to pull from an NSCC login/compute node by default because
# outbound OCI downloads there are too slow/unreliable for this workflow.

set -euo pipefail

OPENCLAW_PREPARE_VERSION=2026.9.2
IMAGE=""
OUT=""
FORCE=0
ALLOW_NSCC_PULL=0
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) OPENCLAW_PREPARE_VERSION="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --repo-root) AB_REPO_ROOT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --allow-nscc-pull) ALLOW_NSCC_PULL=1; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$OPENCLAW_PREPARE_VERSION" in
    2026.9.2|2026.9.6) ;;
    *) echo "ERROR: unsupported OpenClaw version: $OPENCLAW_PREPARE_VERSION" >&2; exit 2 ;;
esac
IMAGE="${IMAGE:-docker://ghcr.io/openclaw/openclaw:${OPENCLAW_PREPARE_VERSION}}"
OUT="${OUT:-${HOME}/agent-bench-sif-build/openclaw_${OPENCLAW_PREPARE_VERSION}_base.sif}"

HOST_FQDN="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
HOST_SHORT="$(hostname 2>/dev/null || true)"
if [[ "$ALLOW_NSCC_PULL" -eq 0 ]]; then
    case "${HOST_FQDN} ${HOST_SHORT}" in
        *asp2a*|*aspire2a*|*pbs*|*x1000*|*x1001*)
            cat >&2 <<'EOF'
ERROR: refusing to pull the OpenClaw OCI image on NSCC.
Run this script on the local WSL workstation instead, then copy the SIF to NSCC.
Use --allow-nscc-pull only for an intentional override.
EOF
            exit 2
            ;;
    esac
fi

# Local WSL installations may expose either `singularity` or `apptainer`.
if command -v singularity >/dev/null 2>&1; then
    CONTAINER_CLI="singularity"
elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CLI="apptainer"
else
    # Module loading is useful on managed hosts but harmless locally when absent.
    module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_CLI="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_CLI="apptainer"
    else
        echo "ERROR: neither singularity nor apptainer is on PATH" >&2
        exit 2
    fi
fi

AUDIT="$AB_REPO_ROOT/agent_bench/deployments/pbs/audit_oc_v2026_9_2.sh"
[[ -x "$AUDIT" || -f "$AUDIT" ]] || {
    echo "ERROR: audit script not found: $AUDIT" >&2
    exit 2
}

mkdir -p "$(dirname "$OUT")"

if [[ -f "$OUT" && "$FORCE" -eq 0 ]]; then
    echo "[prepare-oc-9.2] reuse existing local base: $OUT"
else
    TMP="${OUT}.new"
    rm -f "$TMP"
    echo "[prepare-oc-9.2] pulling locally with $CONTAINER_CLI: $IMAGE"
    "$CONTAINER_CLI" pull "$TMP" "$IMAGE"
    [[ -s "$TMP" ]] || {
        echo "ERROR: container pull produced no SIF: $TMP" >&2
        exit 1
    }
    mv "$TMP" "$OUT"
fi

echo "[prepare-oc-9.2] local base ready"
ls -lh "$OUT"
sha256sum "$OUT"

echo "[prepare-oc-9.2] running local migration audit"
bash "$AUDIT" "$OUT" "$OPENCLAW_PREPARE_VERSION"

echo "[prepare-oc-9.2] PASS"
echo "[prepare-oc] version: $OPENCLAW_PREPARE_VERSION; build the Agent_Bench candidate locally before uploading."
