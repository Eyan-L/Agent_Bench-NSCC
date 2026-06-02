#!/usr/bin/env bash
# Smoke-test the OpenHands harness image: SDK imports, driver script present.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"
VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$HARNESS_DIR/VERSION")"
AB_AGENT_RUNTIME_IMAGE="${AB_AGENT_RUNTIME_IMAGE:-agent-bench/openhands:${VERSION}}"

echo "[smoke] image: $AB_AGENT_RUNTIME_IMAGE"

# 1. SDK import
docker run --rm "$AB_AGENT_RUNTIME_IMAGE" python -c \
    "import openhands.sdk, openhands.tools; \
print('sdk', openhands.sdk.version.__version__ if hasattr(openhands.sdk, 'version') else 'imported'); \
from openhands.tools import get_default_tools; \
print('tools:', [t.name for t in get_default_tools(enable_browser=False)])"

# 2. driver.py exists + parses
docker run --rm "$AB_AGENT_RUNTIME_IMAGE" python /opt/openhands_driver.py --help >/dev/null

echo "[smoke] ok"
