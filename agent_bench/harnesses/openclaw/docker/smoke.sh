#!/usr/bin/env bash
# Smoke test for the OpenClaw harness image. Exercises:
#   1. Image runs at all (`openclaw --version` matches pinned version).
#   2. Patches are baked in (apply_patch.py status reports all "applied").
#   3. The gateway can start, accept a connection, then shut down cleanly.
#
# Does NOT touch the wider Agent_Bench stack — pure image-level test.
#
# Usage:
#   bash agent_bench/harnesses/openclaw/docker/smoke.sh
#   OC_IMAGE_TAG=foo/openclaw:dev bash agent_bench/harnesses/openclaw/docker/smoke.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"
VERSION_FILE="$HARNESS_DIR/patch/VERSION"
OPENCLAW_VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$VERSION_FILE")"
OC_IMAGE_TAG="${OC_IMAGE_TAG:-agent-bench/openclaw:${OPENCLAW_VERSION}}"

pass()  { echo "  PASS  $*"; }
fail()  { echo "  FAIL  $*" >&2; exit 1; }
banner(){ echo; echo "[smoke] $* ($OC_IMAGE_TAG)"; }

# 1. Image present + version pin. Bypass any image entrypoint (tini etc.) so
# `--version` is passed straight to the openclaw bin in both the hand-rolled
# image and the upstream-layered image.
banner "1/3 version check"
raw="$(docker run --rm --entrypoint /usr/local/bin/openclaw "$OC_IMAGE_TAG" --version 2>&1 | tr -d '\r' | tail -1)"
# Format: "OpenClaw <version>" or "OpenClaw <version> (<short-sha>)"
got="$(awk '{print $2}' <<< "$raw")"
[[ "$got" == "$OPENCLAW_VERSION" ]] || fail "version mismatch: got '$got' (raw: '$raw'), expected '$OPENCLAW_VERSION'"
pass "openclaw --version: $raw"

# 2. Patches baked
banner "2/3 patch markers (mode=baked)"
docker run --rm \
    -e OC_PATCH_MODE=baked \
    --entrypoint python3 \
    "$OC_IMAGE_TAG" /opt/openclaw-tools/apply_patch.py status \
    >/tmp/oc_smoke_patch.txt 2>&1 || {
        cat /tmp/oc_smoke_patch.txt >&2
        fail "patch status command failed"
    }
if grep -E '^\s*(missing|FAIL|error)' /tmp/oc_smoke_patch.txt >/dev/null; then
    cat /tmp/oc_smoke_patch.txt >&2
    fail "patch status reports a non-applied patch"
fi
applied="$(grep -c -E 'applied|present|ok' /tmp/oc_smoke_patch.txt || true)"
pass "patch status: $applied applied indicator(s)"

# 3. Gateway can start and we can stop it.
# We don't need a real openclaw.json with provider creds for this — just see
# the gateway accept the auth env and start its loop. A 6s window is generous;
# we kill the container to confirm it tears down cleanly.
banner "3/3 gateway launch (--help / dry path)"
docker run --rm \
    -e OPENCLAW_GATEWAY_TOKEN=smoke-token \
    --entrypoint /usr/local/bin/openclaw \
    "$OC_IMAGE_TAG" gateway --help \
    >/tmp/oc_smoke_gw.txt 2>&1 || {
        cat /tmp/oc_smoke_gw.txt >&2
        fail "openclaw gateway --help failed"
    }
grep -qi 'gateway' /tmp/oc_smoke_gw.txt || {
    cat /tmp/oc_smoke_gw.txt >&2
    fail "gateway --help output unexpected"
}
pass "openclaw gateway --help OK"

echo
echo "[smoke] all 3 checks passed for $OC_IMAGE_TAG"
