#!/usr/bin/env bash
# Bake the OpenClaw runtime patches (agent_bench/harnesses/openclaw/patch/) into a
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
#     --repo-root <repo_root>
#
# Optional:
#   --inside-pkg <path>   path of the openclaw npm pkg inside the sif
#                         (default: /app)
#   --size MB             overlay size in MB (default: 512)
#   --force               rebuild even if the overlay already exists
#
# Notes:
# * Requires `singularity overlay create` (Singularity 3.6+).
# * No root / fakeroot needed: ext3 overlay files can be created by users.

set -euo pipefail

SIF=""
OVERLAY=""
AB_REPO_ROOT=""
INSIDE_PKG="/app"
SIZE_MB=512
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sif)        SIF="$2"; shift 2 ;;
        --overlay)    OVERLAY="$2"; shift 2 ;;
        --repo-root)   AB_REPO_ROOT="$2"; shift 2 ;;
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
[[ -n "$AB_REPO_ROOT" ]] || { echo "ERROR: --repo-root required" >&2; exit 1; }
[[ -f "$SIF"      ]] || { echo "ERROR: sif not found: $SIF" >&2; exit 1; }
[[ -d "$AB_REPO_ROOT" ]] || { echo "ERROR: repo-root not found: $AB_REPO_ROOT" >&2; exit 1; }
command -v singularity >/dev/null 2>&1 || { echo "ERROR: singularity not on PATH" >&2; exit 1; }

PATCH_DIR="$AB_REPO_ROOT/agent_bench/harnesses/openclaw/patch"
APPLY_SCRIPT="$AB_REPO_ROOT/agent_bench/harnesses/openclaw/apply_patch.py"
RUNTIME_COMPAT_SCRIPT="$AB_REPO_ROOT/agent_bench/harnesses/openclaw/runtime_compat_patch.py"
[[ -d "$PATCH_DIR"   ]] || { echo "ERROR: patch dir missing: $PATCH_DIR" >&2; exit 1; }
[[ -f "$APPLY_SCRIPT" ]] || { echo "ERROR: apply_patch.py missing: $APPLY_SCRIPT" >&2; exit 1; }
[[ -f "$RUNTIME_COMPAT_SCRIPT" ]] || { echo "ERROR: runtime compat patch missing: $RUNTIME_COMPAT_SCRIPT" >&2; exit 1; }

if [[ -f "$OVERLAY" && "$FORCE" -ne 1 ]]; then
    echo "[build-oc-overlay] overlay already exists at $OVERLAY (use --force to rebuild)"
    exit 0
fi
mkdir -p "$(dirname "$OVERLAY")"
TMP_OVERLAY="${OVERLAY}.tmp.$$"
rm -f "$TMP_OVERLAY"
cleanup_tmp() {
    rm -f "$TMP_OVERLAY"
}
trap cleanup_tmp EXIT

echo "[build-oc-overlay] creating ${SIZE_MB}MB overlay at $TMP_OVERLAY"
singularity overlay create --size "$SIZE_MB" "$TMP_OVERLAY"

echo "[build-oc-overlay] applying patches inside container"
echo "  sif         : $SIF"
echo "  inside-pkg  : $INSIDE_PKG"
echo "  patch dir   : $PATCH_DIR"

# Run the patchers inside the container, against the sif's OpenClaw install,
# with the overlay mounted rw so changes persist into the overlay file.
#
# apply_patch.py handles the normal Agent_Bench OpenClaw patch set. The runtime
# compatibility pass then patches two pinned-v2026.4.22 behaviours that are not
# exposed as configuration: the 10s sessions.patch deadline used by
# sessions_spawn and public model-pricing catalogue URLs. Both passes fail
# closed when their expected runtime anchors are missing.
OPENCLAW_HARNESS_DIR="$(dirname "$APPLY_SCRIPT")"
SUBAGENT_PATCH_TIMEOUT_MS="${AB_OPENCLAW_SUBAGENT_SESSION_PATCH_TIMEOUT_MS:-45000}"
singularity exec \
    --overlay "$TMP_OVERLAY:rw" \
    --bind "$OPENCLAW_HARNESS_DIR:/oc-harness:ro" \
    "$SIF" \
    bash -c "
        set -e
        export AB_HARNESS_OPENCLAW_SOURCE_ROOT='$INSIDE_PKG'
        export AB_HARNESS_OPENCLAW_PATCH_MODE=runtime
        python3 /oc-harness/apply_patch.py apply
        python3 /oc-harness/apply_patch.py status
        python3 /oc-harness/runtime_compat_patch.py \
            --root '$INSIDE_PKG' \
            --session-patch-timeout-ms '$SUBAGENT_PATCH_TIMEOUT_MS' \
            --status-out '$INSIDE_PKG/.agent-bench-runtime-compat.json'
    "

mv -f "$TMP_OVERLAY" "$OVERLAY"
trap - EXIT
echo "[build-oc-overlay] done -> $OVERLAY"
