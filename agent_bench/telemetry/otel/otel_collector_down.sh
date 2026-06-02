#!/usr/bin/env bash
# Stop and remove the OTel collector container. Idempotent.
set -euo pipefail
NAME="${AB_TELEMETRY_NAME:-oc-otel}"
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker rm -f "$NAME" >/dev/null
    echo "[otel-down] removed $NAME" >&2
else
    echo "[otel-down] $NAME not present" >&2
fi
