#!/usr/bin/env bash
# Build the Hermes harness image.
#
# Default tag is agent-bench/hermes:<version>, where <version> is read from
# agent_bench/harnesses/hermes/patch/VERSION (upstream_version field). Override via
# AB_AGENT_RUNTIME_IMAGE.
#
# Usage:
#   bash agent_bench/harnesses/hermes/docker/build.sh
#   AB_AGENT_RUNTIME_IMAGE=foo/hermes:dev bash agent_bench/harnesses/hermes/docker/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"
VERSION_FILE="$HARNESS_DIR/patch/VERSION"

if [[ ! -f "$VERSION_FILE" ]]; then
    echo "ERROR: $VERSION_FILE not found" >&2
    exit 1
fi

HERMES_VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$VERSION_FILE")"
[[ -n "$HERMES_VERSION" ]] || {
    echo "ERROR: could not parse upstream_version from $VERSION_FILE" >&2
    exit 1
}

AB_AGENT_RUNTIME_IMAGE="${AB_AGENT_RUNTIME_IMAGE:-agent-bench/hermes:${HERMES_VERSION}}"

echo "[build] image  = $AB_AGENT_RUNTIME_IMAGE"
echo "[build] context= $HARNESS_DIR"
echo "[build] version= $HERMES_VERSION"

docker build \
    --build-arg "HERMES_VERSION=$HERMES_VERSION" \
    -t "$AB_AGENT_RUNTIME_IMAGE" \
    -f "$HERE/Dockerfile" \
    "$HARNESS_DIR"

echo "[build] done -> $AB_AGENT_RUNTIME_IMAGE"
