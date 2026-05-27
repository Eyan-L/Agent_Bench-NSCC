#!/usr/bin/env bash
# Start the OTel collector container that openclaw_agent runs export to.
#
# Idempotent: does nothing if a container named OC_OTEL_NAME already runs.
# Safe to re-run.
#
# Mounts the project-tracked collector config at otel/collector-config.yaml
# (read-only) and OC_OTEL_DATA_DIR (default /tmp/otel) for trace output.

set -euo pipefail

KIT_ROOT="${KIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NAME="${OC_OTEL_NAME:-oc-otel}"
IMAGE="${OC_OTEL_IMAGE:-otel/opentelemetry-collector-contrib:latest}"
CONFIG="$KIT_ROOT/otel/collector-config.yaml"
DATA_DIR="${OC_OTEL_DATA_DIR:-/tmp/otel}"

if [ ! -f "$CONFIG" ]; then
    echo "[otel-up] config missing: $CONFIG" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"
chmod 777 "$DATA_DIR" 2>/dev/null || true
# Ensure trace file exists with permissive mode so the file exporter can write.
touch "$DATA_DIR/otel-traces.jsonl"
chmod 666 "$DATA_DIR/otel-traces.jsonl" 2>/dev/null || true

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[otel-up] $NAME already running" >&2
    exit 0
fi
# If a stopped container with same name exists, remove it.
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker rm -f "$NAME" >/dev/null
fi

echo "[otel-up] starting $NAME (image=$IMAGE, data=$DATA_DIR)" >&2
docker run -d --name "$NAME" --network host \
    -v "$CONFIG:/etc/otelcol-contrib/config.yaml:ro" \
    -v "$DATA_DIR:/data" \
    "$IMAGE" >/dev/null

# Poll for readiness (HTTP receiver on :4318).
for i in $(seq 1 30); do
    if curl -fsS -o /dev/null -X POST -H 'Content-Type: application/json' \
        --data '{}' http://127.0.0.1:4318/v1/traces 2>/dev/null; then
        echo "[otel-up] ready after ${i}s" >&2
        exit 0
    fi
    sleep 1
done
echo "[otel-up] OTel collector did not become ready in 30s" >&2
docker logs --tail 30 "$NAME" >&2 || true
exit 2
