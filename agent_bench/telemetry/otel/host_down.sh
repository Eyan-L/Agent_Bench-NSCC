#!/usr/bin/env bash
# Stop the host otelcol-contrib started by host_up.sh.
set -euo pipefail
DATA_DIR="${AB_TELEMETRY_DATA_DIR:-/tmp/otel}"
PID_FILE="$DATA_DIR/otelcol.pid"
if [ ! -f "$PID_FILE" ]; then
    echo "[otel-down-host] no pid file at $PID_FILE; nothing to do" >&2
    exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID"
    for _ in $(seq 1 20); do
        sleep 0.2
        kill -0 "$PID" 2>/dev/null || break
    done
    kill -KILL "$PID" 2>/dev/null || true
    echo "[otel-down-host] stopped pid $PID" >&2
fi
rm -f "$PID_FILE"
