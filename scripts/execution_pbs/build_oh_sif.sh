#!/usr/bin/env bash
# Build an OpenHands runtime SIF on NSCC without Docker.
#
# The local Docker image under harnesses/openhands/docker/ is built from
# python:3.12-slim + openhands-sdk/tools + /opt/openhands_driver.py. On NSCC we
# cannot rely on Docker, so this script follows the same sandbox -> seal flow as
# build_oc_sif.sh:
#
#   1. `singularity build --sandbox sb/ <base.sif>`  - writable rootfs copy
#   2. install openhands-sdk + openhands-tools into the sandbox Python
#   3. copy harnesses/openhands/driver.py to /opt/openhands_driver.py
#   4. create the mount targets the app binds at runtime
#   5. `singularity build out.sif sb/` and atomically swap into place
#
# Default base is python:3.12 rather than python:3.12-slim. The full image is
# larger, but includes git/ca-certificates, which the harness needs for patch
# collection and avoids needing privileged apt inside a Singularity sandbox.
#
# Usage:
#   bash scripts/execution_pbs/build_oh_sif.sh \
#       --base /scratch/users/ntu/$USER/sif_images/openhands_python3.12.sif \
#       --out  /scratch/users/ntu/$USER/sif_images/openhands_runtime.sif

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER_NAME}"
KIT_ROOT="${KIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

OPENHANDS_VERSION_DEFAULT="1.21.0"
VERSION_FILE="$KIT_ROOT/harnesses/openhands/VERSION"
if [[ -f "$VERSION_FILE" ]]; then
    OPENHANDS_VERSION_DEFAULT="$(awk -F': *' '$1 == "upstream_version" { print $2; exit }' "$VERSION_FILE")"
fi

OPENHANDS_VERSION="${OH_RUNTIME_VERSION:-${OPENHANDS_VERSION_DEFAULT:-1.21.0}}"
BASE_SIF="$SCRATCH/sif_images/openhands_python3.12.sif"
OUT_SIF="$SCRATCH/sif_images/openhands_runtime.sif"
SANDBOX_DIR=""
KEEP_SANDBOX=0
BASE_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)         BASE_SIF="$2"; BASE_SET=1; shift 2 ;;
        --out)          OUT_SIF="$2"; shift 2 ;;
        --version)      OPENHANDS_VERSION="$2"; shift 2 ;;
        --sandbox-dir)  SANDBOX_DIR="$2"; shift 2 ;;
        --kit-root)     KIT_ROOT="$2"; shift 2 ;;
        --keep-sandbox) KEEP_SANDBOX=1; shift ;;
        -h|--help)      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$BASE_SET" == "0" ]]; then
    BASE_SIF="$SCRATCH/sif_images/openhands_python3.12.sif"
fi

DRIVER="$KIT_ROOT/harnesses/openhands/driver.py"
[[ -f "$BASE_SIF" ]] || { echo "ERROR: --base sif not found: $BASE_SIF" >&2; exit 1; }
[[ -f "$DRIVER" ]] || { echo "ERROR: OpenHands driver not found: $DRIVER" >&2; exit 1; }

if [[ -z "$SANDBOX_DIR" ]]; then
    SANDBOX_DIR="$(mktemp -d "$SCRATCH/.oh_sb_build_XXXXXX")"
fi
trap '[[ $KEEP_SANDBOX -eq 1 ]] || rm -rf "$SANDBOX_DIR"' EXIT

module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
command -v singularity >/dev/null 2>&1 || { echo "ERROR: singularity not on PATH" >&2; exit 1; }

echo "[build-oh-sif] base       = $BASE_SIF"
echo "[build-oh-sif] out        = $OUT_SIF"
echo "[build-oh-sif] version    = $OPENHANDS_VERSION"
echo "[build-oh-sif] sandbox    = $SANDBOX_DIR"
echo "[build-oh-sif] kit-root   = $KIT_ROOT"
echo "[build-oh-sif] started    = $(date -u +%FT%TZ)"

rm -rf "$SANDBOX_DIR"
singularity build --sandbox "$SANDBOX_DIR" "$BASE_SIF"

mkdir -p \
    "$SANDBOX_DIR/opt" \
    "$SANDBOX_DIR/workspace" \
    "$SANDBOX_DIR/etc/agent-bench/state" \
    "$SANDBOX_DIR/var/openhands/events" \
    "$SANDBOX_DIR/.singularity.d/env"
install -m 0755 "$DRIVER" "$SANDBOX_DIR/opt/openhands_driver.py"

cat > "$SANDBOX_DIR/.singularity.d/env/90-agent-bench-openhands.sh" <<EOF
export OPENHANDS_SUPPRESS_BANNER=1
export OPENHANDS_VERSION=${OPENHANDS_VERSION}
export RUNTIME=process
EOF
chmod 0644 "$SANDBOX_DIR/.singularity.d/env/90-agent-bench-openhands.sh"

singularity exec --writable "$SANDBOX_DIR" bash -c "
    set -euo pipefail
    if ! command -v git >/dev/null 2>&1; then
        echo 'ERROR: git is missing in the OpenHands base image.'
        echo 'Use the default docker://python:3.12 base, or another base that includes git.'
        exit 1
    fi
    python -m pip install --no-cache-dir uv
    UV_NO_CACHE=1 python -m uv pip install --system \
        'openhands-sdk==${OPENHANDS_VERSION}' \
        'openhands-tools==${OPENHANDS_VERSION}'
    python -c \"import openhands.sdk, openhands.tools; print('openhands sdk+tools OK')\"
"

OUT_NEW="${OUT_SIF}.new"
rm -f "$OUT_NEW"
mkdir -p "$(dirname "$OUT_SIF")"
singularity build "$OUT_NEW" "$SANDBOX_DIR"

if [[ -f "$OUT_SIF" ]]; then
    mv "$OUT_SIF" "${OUT_SIF}.bak"
fi
mv "$OUT_NEW" "$OUT_SIF"

echo "[build-oh-sif] finished   = $(date -u +%FT%TZ)"
ls -lh "$OUT_SIF"
echo "[build-oh-sif] smoke:"
singularity exec "$OUT_SIF" python -c "import openhands.sdk, openhands.tools; print('    openhands import OK')"
singularity exec "$OUT_SIF" python /opt/openhands_driver.py --help >/dev/null
echo "    /opt/openhands_driver.py --help OK"
echo "[build-oh-sif] done; backup at ${OUT_SIF}.bak (if pre-existing)"
