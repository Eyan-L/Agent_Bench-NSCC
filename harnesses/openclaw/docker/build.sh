#!/usr/bin/env bash
# Build the OpenClaw harness image.
#
# Default tag is agent-bench/openclaw:<version>, where <version> is read from
# harnesses/openclaw/patch/VERSION (upstream_version field). Override via
# OC_IMAGE_TAG.
#
# Usage:
#   bash harnesses/openclaw/docker/build.sh
#   OC_IMAGE_TAG=foo/openclaw:dev bash harnesses/openclaw/docker/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"           # harnesses/openclaw/
VERSION_FILE="$HARNESS_DIR/patch/VERSION"

if [[ ! -f "$VERSION_FILE" ]]; then
    echo "ERROR: $VERSION_FILE not found" >&2
    exit 1
fi

OPENCLAW_VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$VERSION_FILE")"
[[ -n "$OPENCLAW_VERSION" ]] || { echo "ERROR: could not parse upstream_version from $VERSION_FILE" >&2; exit 1; }

OC_IMAGE_TAG="${OC_IMAGE_TAG:-agent-bench/openclaw:${OPENCLAW_VERSION}}"

echo "[build] image  = $OC_IMAGE_TAG"
echo "[build] context= $HARNESS_DIR"
echo "[build] version= $OPENCLAW_VERSION"

docker build \
    --build-arg "OPENCLAW_VERSION=$OPENCLAW_VERSION" \
    -t "$OC_IMAGE_TAG" \
    -f "$HERE/Dockerfile" \
    "$HARNESS_DIR"

echo "[build] done -> $OC_IMAGE_TAG"
