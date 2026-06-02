#!/usr/bin/env bash
# Build the OpenClaw harness image.
#
# Default tag is agent-bench/openclaw:<version>, where <version> is read from
# agent_bench/harnesses/openclaw/patch/VERSION (upstream_version field). Override via
# AB_AGENT_RUNTIME_IMAGE.
#
# Usage:
#   bash agent_bench/harnesses/openclaw/docker/build.sh
#   AB_AGENT_RUNTIME_IMAGE=foo/openclaw:dev bash agent_bench/harnesses/openclaw/docker/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"           # agent_bench/harnesses/openclaw/
VERSION_FILE="$HARNESS_DIR/patch/VERSION"

if [[ ! -f "$VERSION_FILE" ]]; then
    echo "ERROR: $VERSION_FILE not found" >&2
    exit 1
fi

OPENCLAW_VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$VERSION_FILE")"
[[ -n "$OPENCLAW_VERSION" ]] || { echo "ERROR: could not parse upstream_version from $VERSION_FILE" >&2; exit 1; }

AB_AGENT_RUNTIME_IMAGE="${AB_AGENT_RUNTIME_IMAGE:-agent-bench/openclaw:${OPENCLAW_VERSION}}"

echo "[build] image  = $AB_AGENT_RUNTIME_IMAGE"
echo "[build] context= $HARNESS_DIR"
echo "[build] version= $OPENCLAW_VERSION"

docker build \
    --build-arg "OPENCLAW_VERSION=$OPENCLAW_VERSION" \
    -t "$AB_AGENT_RUNTIME_IMAGE" \
    -f "$HERE/Dockerfile" \
    "$HARNESS_DIR"

echo "[build] done -> $AB_AGENT_RUNTIME_IMAGE"
