#!/usr/bin/env bash
# Build the Agent_Bench OpenClaw runtime SIF pinned to v2026.9.2.
#
# The v2026.9.2 base already provides native rolling tool-loop detection,
# post-compaction loop protection, and OpenTelemetry diagnostics. Agent_Bench
# only bakes the remaining NSCC/container integration and raw-serving telemetry
# patches declared in patch/MANIFEST.json, plus the offline-pricing compatibility
# rewrite for firewalled NSCC nodes.
#
# The build is deliberately fail-closed:
#   1. verify the base image reports the expected OpenClaw version;
#   2. extract to a writable sandbox;
#   3. apply every required Agent_Bench patch;
#   4. rewrite public pricing URLs (historical sessions.patch timeout is optional);
#   5. rebuild a sealed SIF;
#   6. verify the sealed SIF version, every manifest marker, native loop-safety
#      anchors, and runtime-compatibility status;
#   7. only then atomically replace the target SIF.
#
# Typical use with an existing base SIF:
#   bash agent_bench/deployments/pbs/build_oc_sif.sh
#
# Pull the pinned official GHCR image when the base SIF is absent:
#   bash agent_bench/deployments/pbs/build_oc_sif.sh --pull-base
#
# Defaults:
#   version       2026.9.2
#   image         docker://ghcr.io/openclaw/openclaw:2026.9.2
#   base          $SCRATCH/sif_images/openclaw_2026.9.2.sif
#   out           $SCRATCH/sif_images/openclaw_2026.9.2_agentbench.sif
#   inside-pkg    /app
#   sandbox-dir   unique directory under $SCRATCH

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER_NAME}"
OPENCLAW_VERSION="${AB_OPENCLAW_VERSION:-2026.9.2}"
OPENCLAW_IMAGE="${AB_OPENCLAW_IMAGE:-docker://ghcr.io/openclaw/openclaw:${OPENCLAW_VERSION}}"

BASE_SIF="$SCRATCH/sif_images/openclaw_${OPENCLAW_VERSION}.sif"
OUT_SIF="$SCRATCH/sif_images/openclaw_${OPENCLAW_VERSION}_agentbench.sif"
INSIDE_PKG="/app"
SANDBOX_DIR=""
KEEP_SANDBOX=0
PULL_BASE=0
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SUBAGENT_PATCH_TIMEOUT_MS="${AB_OPENCLAW_SUBAGENT_SESSION_PATCH_TIMEOUT_MS:-45000}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)      OPENCLAW_VERSION="$2"; shift 2 ;;
        --image)        OPENCLAW_IMAGE="$2"; shift 2 ;;
        --base)         BASE_SIF="$2"; shift 2 ;;
        --out)          OUT_SIF="$2"; shift 2 ;;
        --inside-pkg)   INSIDE_PKG="$2"; shift 2 ;;
        --sandbox-dir)  SANDBOX_DIR="$2"; shift 2 ;;
        --repo-root)    AB_REPO_ROOT="$2"; shift 2 ;;
        --pull-base)    PULL_BASE=1; shift ;;
        --keep-sandbox) KEEP_SANDBOX=1; shift ;;
        -h|--help)      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

PATCH_DIR="$AB_REPO_ROOT/agent_bench/harnesses/openclaw/patch"
RUNTIME_COMPAT_SCRIPT="$AB_REPO_ROOT/agent_bench/harnesses/openclaw/runtime_compat_patch.py"
HARNESS_DIR="$AB_REPO_ROOT/agent_bench/harnesses/openclaw"

[[ -d "$PATCH_DIR" ]] || { echo "ERROR: patch dir not under $AB_REPO_ROOT" >&2; exit 1; }
[[ -f "$RUNTIME_COMPAT_SCRIPT" ]] || {
    echo "ERROR: runtime compatibility patch missing: $RUNTIME_COMPAT_SCRIPT" >&2
    exit 1
}

module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
command -v singularity >/dev/null 2>&1 || { echo "ERROR: singularity not on PATH" >&2; exit 1; }

if [[ ! -f "$BASE_SIF" ]]; then
    if [[ $PULL_BASE -ne 1 ]]; then
        echo "ERROR: base SIF not found: $BASE_SIF" >&2
        echo "       rerun with --pull-base or provide --base <path>" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$BASE_SIF")"
    echo "[build-oc-sif] pulling base image: $OPENCLAW_IMAGE"
    singularity pull "$BASE_SIF" "$OPENCLAW_IMAGE"
fi

verify_openclaw_version() {
    local image="$1"
    singularity exec "$image" node -e "
const fs = require('node:fs');
const p = JSON.parse(fs.readFileSync('$INSIDE_PKG/package.json', 'utf8'));
const expected = '$OPENCLAW_VERSION';
if (p.version !== expected) {
  console.error('OpenClaw version mismatch: ' + p.version + ' != ' + expected);
  process.exit(2);
}
console.log(p.version);
"
}

echo "[build-oc-sif] expected version = $OPENCLAW_VERSION"
echo "[build-oc-sif] base image       = $OPENCLAW_IMAGE"
echo "[build-oc-sif] base sif         = $BASE_SIF"
echo "[build-oc-sif] out sif          = $OUT_SIF"
echo "[build-oc-sif] repo root        = $AB_REPO_ROOT"
echo "[build-oc-sif] started          = $(date -u +%FT%TZ)"

echo "[build-oc-sif] verifying base OpenClaw version"
verify_openclaw_version "$BASE_SIF"

if [[ -z "$SANDBOX_DIR" ]]; then
    SANDBOX_DIR="$(mktemp -d "$SCRATCH/.oc_${OPENCLAW_VERSION}_build_XXXXXX")"
fi
trap '[[ $KEEP_SANDBOX -eq 1 ]] || rm -rf "$SANDBOX_DIR"' EXIT

# 1. Extract SIF -> writable sandbox.
rm -rf "$SANDBOX_DIR"
singularity build --sandbox "$SANDBOX_DIR" "$BASE_SIF"

# 2. Stage the patch driver inside the writable rootfs.
cp -r "$HARNESS_DIR" "$SANDBOX_DIR/oc-harness"

# 3. Apply only the patches declared for v2026.9.2. Every listed patch is
# required; moved anchors intentionally fail the build so they can be ported.
singularity exec --writable "$SANDBOX_DIR" bash -c "
    set -euo pipefail
    export AB_HARNESS_OPENCLAW_SOURCE_ROOT='$INSIDE_PKG'
    export AB_HARNESS_OPENCLAW_PATCH_MODE=runtime
    python3 /oc-harness/apply_patch.py apply
    python3 /oc-harness/apply_patch.py status
    python3 /oc-harness/runtime_compat_patch.py \
        --root '$INSIDE_PKG' \
        --session-patch-timeout-ms '$SUBAGENT_PATCH_TIMEOUT_MS' \
        --allow-missing-session-patch \
        --status-out '$INSIDE_PKG/.agent-bench-runtime-compat.json'
"

# 4. Remove build-only state. Keep the compatibility marker in /app.
singularity exec --writable "$SANDBOX_DIR" bash -c "
    set -euo pipefail
    find '$INSIDE_PKG' -name '*.oc-orig' -delete
    rm -rf /oc-harness
"

# 5. Seal beside the target path.
OUT_NEW="${OUT_SIF}.new"
rm -f "$OUT_NEW"
mkdir -p "$(dirname "$OUT_SIF")"
singularity build "$OUT_NEW" "$SANDBOX_DIR"

# 6a. Exact upstream version.
echo "[build-oc-sif] verifying sealed OpenClaw version"
verify_openclaw_version "$OUT_NEW"

# 6b. Every required Agent_Bench manifest marker. baked mode is a hard gate;
# unlike the old builder this is NOT diagnostic-only and has no `|| true`.
echo "[build-oc-sif] verifying sealed Agent_Bench patch markers"
singularity exec --bind "$HARNESS_DIR:/oc-harness:ro" "$OUT_NEW" bash -c "
    set -euo pipefail
    export AB_HARNESS_OPENCLAW_SOURCE_ROOT='$INSIDE_PKG'
    export AB_HARNESS_OPENCLAW_PATCH_MODE=baked
    python3 /oc-harness/apply_patch.py apply
"

# 6c. Native v2026.9.2 loop safety must be present in the sealed runtime. These
# strings come from upstream's rolling detector and post-compaction guard.
echo "[build-oc-sif] verifying native OpenClaw loop safety"
singularity exec "$OUT_NEW" sh -lc "
    grep -R -q 'compaction_loop_persisted' '$INSIDE_PKG/dist' &&
    grep -R -q 'global circuit breaker to prevent runaway loops' '$INSIDE_PKG/dist'
"

# 6d. Offline-pricing compatibility. sessions.patch replacement may be zero on
# v2026.9.2 and is explicitly recorded as non-required.
echo "[build-oc-sif] verifying runtime compatibility marker"
singularity exec "$OUT_NEW" python3 -c "
import json
from pathlib import Path
p = Path('$INSIDE_PKG/.agent-bench-runtime-compat.json')
if not p.is_file():
    raise SystemExit('missing runtime compatibility marker: ' + str(p))
s = json.loads(p.read_text())
if s.get('session_patch_required') is not False:
    raise SystemExit('v2026.9.2 build unexpectedly requires the historical sessions.patch anchor')
if int(s.get('openrouter_pricing_url_replacements') or 0) <= 0:
    raise SystemExit('OpenRouter pricing URL was not replaced')
if int(s.get('litellm_pricing_url_replacements') or 0) <= 0:
    raise SystemExit('LiteLLM pricing URL was not replaced')
if s.get('pricing_mode') != 'offline_data_urls':
    raise SystemExit('pricing mode is not offline_data_urls')
print(json.dumps(s, sort_keys=True))
"

# 7. Atomic swap only after all hard gates pass.
if [[ -f "$OUT_SIF" ]]; then
    rm -f "${OUT_SIF}.bak"
    mv "$OUT_SIF" "${OUT_SIF}.bak"
fi
mv "$OUT_NEW" "$OUT_SIF"

echo "[build-oc-sif] finished = $(date -u +%FT%TZ)"
ls -lh "$OUT_SIF"
sha256sum "$OUT_SIF"
echo "[build-oc-sif] runtime compatibility status:"
singularity exec "$OUT_SIF" cat "$INSIDE_PKG/.agent-bench-runtime-compat.json" \
    2>&1 | sed 's/^/    /'
echo "[build-oc-sif] done; backup at ${OUT_SIF}.bak (if pre-existing)"
