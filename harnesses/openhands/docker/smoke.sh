#!/usr/bin/env bash
# Smoke-test the OpenHands harness image: SDK imports, driver script present.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"
VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$HARNESS_DIR/VERSION")"
OH_IMAGE_TAG="${OH_IMAGE_TAG:-agent-bench/openhands:${VERSION}}"

echo "[smoke] image: $OH_IMAGE_TAG"

# 1. SDK import
docker run --rm "$OH_IMAGE_TAG" python -c \
    "import openhands.sdk, openhands.tools; \
print('sdk', openhands.sdk.version.__version__ if hasattr(openhands.sdk, 'version') else 'imported'); \
from openhands.tools import get_default_tools; \
print('tools:', [t.name for t in get_default_tools(enable_browser=False)])"

# 2. driver.py exists + parses
docker run --rm "$OH_IMAGE_TAG" python /opt/openhands_driver.py --help >/dev/null

echo "[smoke] ok"
