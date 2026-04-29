#!/usr/bin/env bash
# Bake the OpenClaw runtime patches (eval/frontend/openclaw/patch/) into a
# Singularity overlay file. The overlay is mounted read-only in subsequent
# `singularity exec` calls so the patched dist/*.js bundles are visible
# to the gateway/agent without ever modifying the upstream sif.
#
# Idempotent: re-running this script on an existing overlay is a no-op
# unless --force is given.
#
# Usage:
#   pbs/build_oc_overlay.sh \
#     --sif <oc_runtime.sif> \
#     --overlay <out_overlay.img> \
#     --kit-root <repo_root>
#
# Optional:
#   --inside-pkg <path>   path of the openclaw npm pkg inside the sif
#                         (default: /usr/local/lib/node_modules/openclaw)
#   --size MB             overlay size in MB (default: 512)
#   --force               rebuild even if the overlay already exists
#
# Notes:
# * Requires `singularity overlay create` (Singularity 3.6+).
# * No root / fakeroot needed: ext3 overlay files can be created by users.

set -euo pipefail

SIF=""
OVERLAY=""
KIT_ROOT=""
INSIDE_PKG="/usr/local/lib/node_modules/openclaw"
SIZE_MB=512
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sif)        SIF="$2"; shift 2 ;;
        --overlay)    OVERLAY="$2"; shift 2 ;;
        --kit-root)   KIT_ROOT="$2"; shift 2 ;;
        --inside-pkg) INSIDE_PKG="$2"; shift 2 ;;
        --size)       SIZE_MB="$2"; shift 2 ;;
        --force)      FORCE=1; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$SIF"      ]] || { echo "ERROR: --sif required" >&2; exit 1; }
[[ -n "$OVERLAY"  ]] || { echo "ERROR: --overlay required" >&2; exit 1; }
[[ -n "$KIT_ROOT" ]] || { echo "ERROR: --kit-root required" >&2; exit 1; }
[[ -f "$SIF"      ]] || { echo "ERROR: sif not found: $SIF" >&2; exit 1; }
[[ -d "$KIT_ROOT" ]] || { echo "ERROR: kit-root not found: $KIT_ROOT" >&2; exit 1; }
command -v singularity >/dev/null 2>&1 || { echo "ERROR: singularity not on PATH" >&2; exit 1; }

PATCH_DIR="$KIT_ROOT/eval/frontend/openclaw/patch"
APPLY_SCRIPT="$KIT_ROOT/eval/frontend/openclaw/apply_patch.py"
[[ -d "$PATCH_DIR"   ]] || { echo "ERROR: patch dir missing: $PATCH_DIR" >&2; exit 1; }
[[ -f "$APPLY_SCRIPT" ]] || { echo "ERROR: apply_patch.py missing: $APPLY_SCRIPT" >&2; exit 1; }

if [[ -f "$OVERLAY" && "$FORCE" -ne 1 ]]; then
    echo "[build-oc-overlay] overlay already exists at $OVERLAY (use --force to rebuild)"
    exit 0
fi
rm -f "$OVERLAY"

mkdir -p "$(dirname "$OVERLAY")"

echo "[build-oc-overlay] creating ${SIZE_MB}MB overlay at $OVERLAY"
singularity overlay create --size "$SIZE_MB" "$OVERLAY"

echo "[build-oc-overlay] applying patches inside container"
echo "  sif         : $SIF"
echo "  inside-pkg  : $INSIDE_PKG"
echo "  patch dir   : $PATCH_DIR"

# Run the patcher inside the container, against the sif's openclaw install,
# with the overlay mounted rw so changes persist into the overlay file.
singularity exec \
    --overlay "$OVERLAY:rw" \
    --bind "$PATCH_DIR:/oc-patches:ro" \
    --bind "$APPLY_SCRIPT:/oc-apply.py:ro" \
    "$SIF" \
    bash -c "
        set -e
        # apply_patch.py imports PyYAML; alpine/openclaw doesn't ship it.
        # Install into the overlay rw layer once so subsequent ro mounts
        # see it too.
        if ! python3 -c 'import yaml' 2>/dev/null; then
            python3 -m pip install --no-cache-dir --root-user-action=ignore PyYAML
        fi
        cd /oc-patches
        export OC_HOST_OPENCLAW='$INSIDE_PKG'
        export OC_PATCH_MODE=runtime
        python3 /oc-apply.py apply
        python3 /oc-apply.py status
    "

echo "[build-oc-overlay] done -> $OVERLAY"
