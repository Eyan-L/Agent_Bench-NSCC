#!/usr/bin/env bash
# Build the pinned OpenClaw v2026.9.2 Agent_Bench candidate SIF locally.
#
# This is intentionally separate from build_oc_sif.sh, which remains the
# legacy v2026.4.22 build path.  The v2026.9.2 image already contains native
# loop detection and native OTel instrumentation; only the Agent_Bench
# exec/sandbox/fs bridge integration and offline-pricing hardening are baked.

set -euo pipefail

AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
BASE_SIF="${HOME}/agent-bench-sif-build/openclaw_2026.9.2_base.sif"
OUT_SIF="${HOME}/agent-bench-sif-build/openclaw_2026.9.2_agentbench.sif"
SANDBOX_DIR=""
KEEP_SANDBOX=0
ALLOW_NSCC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base) BASE_SIF="$2"; shift 2 ;;
        --out) OUT_SIF="$2"; shift 2 ;;
        --sandbox-dir) SANDBOX_DIR="$2"; shift 2 ;;
        --repo-root) AB_REPO_ROOT="$2"; shift 2 ;;
        --keep-sandbox) KEEP_SANDBOX=1; shift ;;
        --allow-nscc) ALLOW_NSCC=1; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

HOST="$(hostname 2>/dev/null || true)"
if [[ "$ALLOW_NSCC" -ne 1 && "$HOST" == asp2* ]]; then
    echo "ERROR: v2026.9.2 candidate SIF build is local-first; do not build it on NSCC." >&2
    echo "Build in WSL, then copy the sealed SIF to NSCC scratch." >&2
    exit 2
fi

if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
else
    echo "ERROR: neither apptainer nor singularity is on PATH" >&2
    exit 2
fi

PATCHER="$AB_REPO_ROOT/agent_bench/harnesses/openclaw/v2026_9_2_patch.py"
AUDIT="$AB_REPO_ROOT/agent_bench/deployments/pbs/audit_oc_v2026_9_2.sh"
[[ -f "$BASE_SIF" ]] || { echo "ERROR: base SIF not found: $BASE_SIF" >&2; exit 2; }
[[ -f "$PATCHER" ]] || { echo "ERROR: patcher not found: $PATCHER" >&2; exit 2; }
[[ -f "$AUDIT" ]] || { echo "ERROR: audit not found: $AUDIT" >&2; exit 2; }

mkdir -p "$(dirname "$OUT_SIF")" "${HOME}/agent-bench-sif-build"
if [[ -z "$SANDBOX_DIR" ]]; then
    SANDBOX_DIR="$(mktemp -d "${HOME}/agent-bench-sif-build/.oc_9_2_sb_XXXXXX")"
fi
cleanup() {
    if [[ "$KEEP_SANDBOX" -ne 1 ]]; then
        rm -rf "$SANDBOX_DIR"
    fi
}
trap cleanup EXIT

printf '[build-oc-9.2] runtime = %s\n' "$RUNTIME"
printf '[build-oc-9.2] base    = %s\n' "$BASE_SIF"
printf '[build-oc-9.2] out     = %s\n' "$OUT_SIF"
printf '[build-oc-9.2] sandbox = %s\n' "$SANDBOX_DIR"

rm -rf "$SANDBOX_DIR"
"$RUNTIME" build --sandbox "$SANDBOX_DIR" "$BASE_SIF"

ROOT="$SANDBOX_DIR/app"
[[ -d "$ROOT" ]] || { echo "ERROR: extracted SIF has no /app" >&2; exit 1; }

STATUS="$ROOT/.agent-bench-v2026.9.2.json"
python3 "$PATCHER" --root "$ROOT" --status-out "$STATUS"
python3 "$PATCHER" --root "$ROOT" --verify-only

# Never seal legacy patches into the v2026.9.2 runtime.
if grep -R -q -E \
    'AB_HARNESS_OPENCLAW_PATCH_POST_COMPACTION_LOOP_GUARD|AB_HARNESS_OPENCLAW_PATCH_MODEL_USAGE_BEGIN|AB_HARNESS_OPENCLAW_PATCH_TOOL_CALL_BEGIN' \
    "$ROOT/dist" 2>/dev/null; then
    echo "ERROR: legacy v2026.4.22 patch marker detected in v2026.9.2 candidate" >&2
    exit 1
fi

OUT_NEW="${OUT_SIF}.new"
rm -f "$OUT_NEW"
"$RUNTIME" build "$OUT_NEW" "$SANDBOX_DIR"

# Verify the sealed image itself, not merely the writable sandbox.  Use Node,
# which is part of OpenClaw's runtime contract; do not require Python in-image.
VERSION="$($RUNTIME exec --cleanenv --no-home "$OUT_NEW" node -p 'require("/app/package.json").version')"
[[ "$VERSION" == "2026.9.2" ]] || {
    echo "ERROR: sealed candidate version is $VERSION, expected 2026.9.2" >&2
    exit 1
}

"$RUNTIME" exec --cleanenv --no-home "$OUT_NEW" node -e '
const fs = require("node:fs");
const p = "/app/.agent-bench-v2026.9.2.json";
if (!fs.existsSync(p)) throw new Error("missing " + p);
const s = JSON.parse(fs.readFileSync(p, "utf8"));
if (s.upstream_version !== "2026.9.2") throw new Error("unexpected upstream_version");
const verification = s.verification || {};
if (!verification.verified) throw new Error("semantic patch verification did not pass");
const markers = verification.markers || {};
for (const key of ["exec_bridge", "sandbox_bypass", "shared_workspace_spawn", "settled_continuation_policy", "settled_continuation_instruction", "settled_continuation_retry", "empty_stop_retry_limit", "fs_helper", "fs_run", "fs_write", "fs_read", "fs_check"]) {
  if (!Array.isArray(markers[key]) || markers[key].length === 0) throw new Error("missing sealed marker: " + key);
}
process.stdout.write(JSON.stringify(s) + "\n");
'

# Parse the status marker from the sealed image and syntax-check every bundle
# that received an Agent_Bench bridge transform.  This catches bad semantic
# insertion points before the candidate ever reaches NSCC.
PATCHED_RUNTIME_FILES="$($RUNTIME exec --cleanenv --no-home "$OUT_NEW" node -e '
const fs = require("node:fs");
const s = JSON.parse(fs.readFileSync("/app/.agent-bench-v2026.9.2.json", "utf8"));
const files = new Set([
  ...(s.exec_bridge_files_changed || []),
  ...(s.sandbox_bypass_files_changed || []),
  ...(s.shared_workspace_spawn_files_changed || []),
  ...((s.settled_tool_continuation || {}).files_changed || []),
  ...(s.fs_bridge_files_changed || []),
]);
for (const file of files) process.stdout.write(file + "\n");
')"
if [[ -z "$PATCHED_RUNTIME_FILES" ]]; then
    echo "ERROR: status marker contains no patched runtime bundles" >&2
    exit 1
fi
while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    printf '[build-oc-9.2] node --check /app/%s\n' "$rel"
    "$RUNTIME" exec --cleanenv --no-home "$OUT_NEW" node --check "/app/$rel"
done <<< "$PATCHED_RUNTIME_FILES"

bash "$AUDIT" "$OUT_NEW"

if [[ -f "$OUT_SIF" ]]; then
    rm -f "${OUT_SIF}.bak"
    mv "$OUT_SIF" "${OUT_SIF}.bak"
fi
mv "$OUT_NEW" "$OUT_SIF"

printf '[build-oc-9.2] PASS\n'
ls -lh "$OUT_SIF"
sha256sum "$OUT_SIF"
printf '[build-oc-9.2] status marker:\n'
"$RUNTIME" exec --cleanenv --no-home "$OUT_SIF" \
    cat /app/.agent-bench-v2026.9.2.json | sed 's/^/    /'
printf '[build-oc-9.2] next: copy %s to NSCC $SCRATCH/sif_images/openclaw_2026.9.2_agentbench.sif\n' "$OUT_SIF"
