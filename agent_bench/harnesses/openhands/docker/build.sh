#!/usr/bin/env bash
# Build the OpenHands harness image.
#
# Default tag is agent-bench/openhands:<version>, where <version> is read
# from agent_bench/harnesses/openhands/VERSION (upstream_version field).
# Override via AB_AGENT_RUNTIME_IMAGE.
#
# Usage:
#   bash agent_bench/harnesses/openhands/docker/build.sh
#   AB_AGENT_RUNTIME_IMAGE=foo/openhands:dev bash agent_bench/harnesses/openhands/docker/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"            # agent_bench/harnesses/openhands/
VERSION_FILE="$HARNESS_DIR/VERSION"

if [[ ! -f "$VERSION_FILE" ]]; then
    echo "ERROR: $VERSION_FILE not found" >&2
    exit 1
fi

OPENHANDS_VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$VERSION_FILE")"
[[ -n "$OPENHANDS_VERSION" ]] || {
    echo "ERROR: could not parse upstream_version from $VERSION_FILE" >&2
    exit 1
}

AB_AGENT_RUNTIME_IMAGE="${AB_AGENT_RUNTIME_IMAGE:-agent-bench/openhands:${OPENHANDS_VERSION}}"

echo "[build] image  = $AB_AGENT_RUNTIME_IMAGE"
echo "[build] context= $HARNESS_DIR"
echo "[build] version= $OPENHANDS_VERSION"

docker build \
    --build-arg "OPENHANDS_VERSION=$OPENHANDS_VERSION" \
    -t "$AB_AGENT_RUNTIME_IMAGE" \
    -f "$HERE/Dockerfile" \
    "$HARNESS_DIR"

echo "[build] done -> $AB_AGENT_RUNTIME_IMAGE"
