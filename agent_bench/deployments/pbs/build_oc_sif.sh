#!/usr/bin/env bash
# Standardised build of an OpenClaw runtime SIF with all Agent_Bench patches applied.
#
# Why this exists: the upstream sif (docker://ghcr.io/openclaw/openclaw:<tag>)
# ships without Agent_Bench's runtime patches. When we add a new patch under
# agent_bench/harnesses/openclaw/patch/, the robust NSCC path is to bake it into a fresh
# sif — singularity overlays on this cluster run via FUSE in unprivileged mode
# and cannot write to image-side directories (`Permission denied` on `/app/...`).
#
# Pipeline:
#   1. `singularity build --sandbox sb/ <base.sif>`  — writable rootfs copy
#   2. copy agent_bench/harnesses/openclaw/ into sb/oc-harness/   — apply_patch.py needs
#                                                       MANIFEST.json + .patch
#                                                       files at its
#                                                       relative location
#   3. `singularity exec --writable sb/ apply_patch.py apply`
#      idempotent: existing markers → skip; only new patches actually edit
#   4. clean up backups + harness copy from sandbox
#   5. `singularity build out.sif sb/`               — fresh sealed sif
#   6. atomic swap: out.sif → target path (keeps a .bak)
#
# Submit (login node):
#   bash agent_bench/deployments/pbs/build_oc_sif.sh \
#       --base /scratch/users/ntu/$USER/sif_images/openclaw_2026.4.22.sif \
#       --out  /scratch/users/ntu/$USER/sif_images/openclaw_2026.4.22_patched.sif
#
# Or override per-invocation:
#   bash agent_bench/deployments/pbs/build_oc_sif.sh \
#       --base $SCRATCH/sif_images/openclaw_2026.4.22.sif \
#       --out  $SCRATCH/sif_images/openclaw_2026.4.22_patched.sif \
#       --inside-pkg /app \
#       --sandbox-dir $SCRATCH/sif_image/_oc_sb_build
#
# Defaults (override via flags):
#   --base        $SCRATCH/sif_images/openclaw_2026.4.22.sif
#   --out         $SCRATCH/sif_images/openclaw_2026.4.22_patched.sif
#   --inside-pkg  /app                                (OC pkg root inside sif)
#   --sandbox-dir <tmpdir>/oc_sb_build                (deleted on success)
#   --keep-sandbox                                    (skip cleanup, for debug)

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER_NAME}"

BASE_SIF="$SCRATCH/sif_images/openclaw_2026.4.22.sif"
OUT_SIF="$SCRATCH/sif_images/openclaw_2026.4.22_patched.sif"
INSIDE_PKG="/app"
SANDBOX_DIR=""
KEEP_SANDBOX=0
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)         BASE_SIF="$2"; shift 2 ;;
        --out)          OUT_SIF="$2"; shift 2 ;;
        --inside-pkg)   INSIDE_PKG="$2"; shift 2 ;;
        --sandbox-dir)  SANDBOX_DIR="$2"; shift 2 ;;
        --repo-root)     AB_REPO_ROOT="$2"; shift 2 ;;
        --keep-sandbox) KEEP_SANDBOX=1; shift ;;
        -h|--help)      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -f "$BASE_SIF" ]] || { echo "ERROR: --base sif not found: $BASE_SIF" >&2; exit 1; }
[[ -d "$AB_REPO_ROOT/agent_bench/harnesses/openclaw/patch" ]] || { echo "ERROR: patch dir not under $AB_REPO_ROOT" >&2; exit 1; }

# Always work in a unique sandbox dir to dodge stale state. Use a temp dir
# under SCRATCH (sandbox rebuild needs ~3 GB of space; /tmp is small).
if [[ -z "$SANDBOX_DIR" ]]; then
    SANDBOX_DIR="$(mktemp -d "$SCRATCH/.oc_sb_build_XXXXXX")"
fi
trap '[[ $KEEP_SANDBOX -eq 1 ]] || rm -rf "$SANDBOX_DIR"' EXIT

module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
command -v singularity >/dev/null 2>&1 || { echo "ERROR: singularity not on PATH" >&2; exit 1; }

echo "[build-oc-sif] base       = $BASE_SIF"
echo "[build-oc-sif] out        = $OUT_SIF"
echo "[build-oc-sif] inside-pkg = $INSIDE_PKG"
echo "[build-oc-sif] sandbox    = $SANDBOX_DIR"
echo "[build-oc-sif] repo-root   = $AB_REPO_ROOT"
echo "[build-oc-sif] started    = $(date -u +%FT%TZ)"

# 1. Extract sif → writable sandbox
rm -rf "$SANDBOX_DIR"
singularity build --sandbox "$SANDBOX_DIR" "$BASE_SIF"

# 2. Stage the harness tree where apply_patch.py expects (relative to itself).
#    Sandbox is writable, so just cp instead of bind.
cp -r "$AB_REPO_ROOT/agent_bench/harnesses/openclaw" "$SANDBOX_DIR/oc-harness"

# 3. Apply all patches. idempotent by design — markers skip already-baked ones.
singularity exec --writable "$SANDBOX_DIR" bash -c "
    set -e
    export AB_HARNESS_OPENCLAW_SOURCE_ROOT='$INSIDE_PKG'
    export AB_HARNESS_OPENCLAW_PATCH_MODE=runtime
    python3 /oc-harness/apply_patch.py apply
    python3 /oc-harness/apply_patch.py status
"

# 4. Clean per-target .oc-orig backups + the staged harness tree.
singularity exec --writable "$SANDBOX_DIR" bash -c "
    set -e
    find '$INSIDE_PKG' -name '*.oc-orig' -delete
    rm -rf /oc-harness
"

# 5. Rebuild sealed sif beside the target. `singularity build` refuses to
#    overwrite an in-use file; write to .new then swap.
OUT_NEW="${OUT_SIF}.new"
rm -f "$OUT_NEW"
mkdir -p "$(dirname "$OUT_SIF")"
singularity build "$OUT_NEW" "$SANDBOX_DIR"

# 6. Atomic swap with a .bak fallback (only if base/out paths match).
if [[ -f "$OUT_SIF" ]]; then
    mv "$OUT_SIF" "${OUT_SIF}.bak"
fi
mv "$OUT_NEW" "$OUT_SIF"

echo "[build-oc-sif] finished   = $(date -u +%FT%TZ)"
ls -lh "$OUT_SIF"
echo "[build-oc-sif] applied patches (status from new sif):"
singularity exec --bind "$AB_REPO_ROOT/agent_bench/harnesses/openclaw:/oc-harness:ro" "$OUT_SIF" \
    bash -c "AB_HARNESS_OPENCLAW_SOURCE_ROOT='$INSIDE_PKG' python3 /oc-harness/apply_patch.py status" \
    2>&1 | sed 's/^/    /' || true

echo "[build-oc-sif] done; backup at ${OUT_SIF}.bak (if pre-existing)"
