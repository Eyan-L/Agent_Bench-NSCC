#!/usr/bin/env bash
# Build a patched Hermes Agent runtime SIF on NSCC without Docker.
#
# This mirrors build_oc_sif.sh for Hermes:
#
#   1. pull/build an upstream Hermes SIF separately
#   2. `singularity build --sandbox sb/ <base.sif>`
#   3. copy agent_bench/harnesses/hermes/ into the sandbox
#   4. run apply_patch.py against the upstream /opt/hermes tree
#   5. clean staging files and seal a fresh runtime SIF
#
# Usage:
#   bash agent_bench/deployments/pbs/build_hm_sif.sh \
#       --base /scratch/users/ntu/$USER/sif_images/hermes_v2026.5.16.sif \
#       --out  /scratch/users/ntu/$USER/sif_images/hermes_runtime.sif

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER_NAME}"
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

HERMES_VERSION_DEFAULT="v2026.5.16"
VERSION_FILE="$AB_REPO_ROOT/agent_bench/harnesses/hermes/patch/VERSION"
if [[ -f "$VERSION_FILE" ]]; then
    HERMES_VERSION_DEFAULT="$(awk -F': *' '$1 == "upstream_version" { print $2; exit }' "$VERSION_FILE")"
fi

HERMES_VERSION="${AB_AGENT_RUNTIME_VERSION:-${HERMES_VERSION_DEFAULT:-v2026.5.16}}"
BASE_SIF="$SCRATCH/sif_images/hermes_${HERMES_VERSION}.sif"
OUT_SIF="$SCRATCH/sif_images/hermes_runtime.sif"
INSIDE_ROOT="/opt/hermes"
SANDBOX_DIR=""
KEEP_SANDBOX=0
BASE_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)         BASE_SIF="$2"; BASE_SET=1; shift 2 ;;
        --out)          OUT_SIF="$2"; shift 2 ;;
        --version)      HERMES_VERSION="$2"; shift 2 ;;
        --inside-root)  INSIDE_ROOT="$2"; shift 2 ;;
        --sandbox-dir)  SANDBOX_DIR="$2"; shift 2 ;;
        --repo-root)     AB_REPO_ROOT="$2"; shift 2 ;;
        --keep-sandbox) KEEP_SANDBOX=1; shift ;;
        -h|--help)      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$BASE_SET" == "0" ]]; then
    BASE_SIF="$SCRATCH/sif_images/hermes_${HERMES_VERSION}.sif"
fi

[[ -f "$BASE_SIF" ]] || { echo "ERROR: --base sif not found: $BASE_SIF" >&2; exit 1; }
[[ -f "$AB_REPO_ROOT/agent_bench/harnesses/hermes/apply_patch.py" ]] || { echo "ERROR: Hermes apply_patch.py not under $AB_REPO_ROOT" >&2; exit 1; }
[[ -d "$AB_REPO_ROOT/agent_bench/harnesses/hermes/patch" ]] || { echo "ERROR: Hermes patch dir not under $AB_REPO_ROOT" >&2; exit 1; }

if [[ -z "$SANDBOX_DIR" ]]; then
    SANDBOX_DIR="$(mktemp -d "$SCRATCH/.hm_sb_build_XXXXXX")"
fi
trap '[[ $KEEP_SANDBOX -eq 1 ]] || rm -rf "$SANDBOX_DIR"' EXIT

module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
command -v singularity >/dev/null 2>&1 || { echo "ERROR: singularity not on PATH" >&2; exit 1; }

echo "[build-hm-sif] base        = $BASE_SIF"
echo "[build-hm-sif] out         = $OUT_SIF"
echo "[build-hm-sif] version     = $HERMES_VERSION"
echo "[build-hm-sif] inside-root = $INSIDE_ROOT"
echo "[build-hm-sif] sandbox     = $SANDBOX_DIR"
echo "[build-hm-sif] repo-root    = $AB_REPO_ROOT"
echo "[build-hm-sif] started     = $(date -u +%FT%TZ)"

rm -rf "$SANDBOX_DIR"
singularity build --sandbox "$SANDBOX_DIR" "$BASE_SIF"

mkdir -p \
    "$SANDBOX_DIR/workspace" \
    "$SANDBOX_DIR/opt/data" \
    "$SANDBOX_DIR/.singularity.d/env"
cp -r "$AB_REPO_ROOT/agent_bench/harnesses/hermes" "$SANDBOX_DIR/hm-harness"

cat > "$SANDBOX_DIR/.singularity.d/env/90-agent-bench-hermes.sh" <<EOF
export HERMES_VERSION=${HERMES_VERSION}
export HERMES_PATCH_MODE=baked
export HERMES_HOME=/opt/data
export PATH=${INSIDE_ROOT}/.venv/bin:${INSIDE_ROOT}:\${PATH}
export PYTHONPATH=${INSIDE_ROOT}
EOF
chmod 0644 "$SANDBOX_DIR/.singularity.d/env/90-agent-bench-hermes.sh"

singularity exec --writable "$SANDBOX_DIR" bash -c "
    set -euo pipefail
    for bin in python3; do
        if ! command -v \"\$bin\" >/dev/null 2>&1; then
            echo \"ERROR: \$bin is missing in the Hermes base image.\"
            echo 'Use an upstream/base image that includes python3.'
            exit 1
        fi
    done
    export HERMES_AGENT_ROOT='$INSIDE_ROOT'
    export HERMES_PATCH_MODE=runtime
    python3 /hm-harness/apply_patch.py apply
    HERMES_PATCH_MODE=baked python3 /hm-harness/apply_patch.py status
"

singularity exec --writable "$SANDBOX_DIR" bash -c "
    set -e
    rm -rf /hm-harness
"

OUT_NEW="${OUT_SIF}.new"
rm -f "$OUT_NEW"
mkdir -p "$(dirname "$OUT_SIF")"
singularity build "$OUT_NEW" "$SANDBOX_DIR"

if [[ -f "$OUT_SIF" ]]; then
    mv "$OUT_SIF" "${OUT_SIF}.bak"
fi
mv "$OUT_NEW" "$OUT_SIF"

echo "[build-hm-sif] finished    = $(date -u +%FT%TZ)"
ls -lh "$OUT_SIF"
echo "[build-hm-sif] applied patches (status from new sif):"
singularity exec --bind "$AB_REPO_ROOT/agent_bench/harnesses/hermes:/hm-harness:ro" "$OUT_SIF" \
    bash -c "HERMES_AGENT_ROOT='$INSIDE_ROOT' HERMES_PATCH_MODE=baked python3 /hm-harness/apply_patch.py status" \
    2>&1 | sed 's/^/    /'
echo "[build-hm-sif] smoke:"
singularity exec "$OUT_SIF" bash -lc "command -v hermes >/dev/null && python3 - <<'PY'
import importlib.util
assert importlib.util.find_spec('hermes_cli.oneshot') is not None
print('    hermes CLI + hermes_cli.oneshot OK')
PY"
echo "[build-hm-sif] done; backup at ${OUT_SIF}.bak (if pre-existing)"
