#!/usr/bin/env bash
# Stop the OpenClaw Gateway inside a SWE-bench run container. Idempotent.
#
# In normal operation the container is removed at run cleanup, which kills
# the gateway anyway. This script exists for when the harness crashes
# mid-run and we need to clean up an orphan gateway in a still-running
# container without nuking the workspace.
#
# Usage: frontend/openclaw/gateway_down.sh <container_id>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <container_id>" >&2
    exit 1
fi
CID="$1"

if ! docker exec "$CID" true 2>/dev/null; then
    echo "[gateway-down] container $CID not reachable; nothing to do" >&2
    exit 0
fi

# pkill returns non-zero when no matching process exists — that's fine.
docker exec "$CID" pkill -f 'openclaw-gateway' 2>/dev/null || true
docker exec "$CID" pkill -f 'openclaw gateway run' 2>/dev/null || true
echo "[gateway-down] sent SIGTERM to gateway in $CID" >&2
