#!/usr/bin/env bash
# Build the Hermes harness image.
#
# Default tag is agent-bench/hermes:<version>, where <version> is read from
# harnesses/hermes/patch/VERSION (upstream_version field). Override via
# HM_IMAGE_TAG.
#
# Usage:
#   bash harnesses/hermes/docker/build.sh
#   HM_IMAGE_TAG=foo/hermes:dev bash harnesses/hermes/docker/build.sh
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

HM_IMAGE_TAG="${HM_IMAGE_TAG:-agent-bench/hermes:${HERMES_VERSION}}"

echo "[build] image  = $HM_IMAGE_TAG"
echo "[build] context= $HARNESS_DIR"
echo "[build] version= $HERMES_VERSION"

docker build \
    --build-arg "HERMES_VERSION=$HERMES_VERSION" \
    -t "$HM_IMAGE_TAG" \
    -f "$HERE/Dockerfile" \
    "$HARNESS_DIR"

echo "[build] done -> $HM_IMAGE_TAG"
