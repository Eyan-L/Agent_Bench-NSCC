#!/usr/bin/env bash
# Reset the OTel file exporter's trace output between runs.
#
# We can't just `> file` — the collector holds the fd open and continues
# writing at its prior offset, leaving a sparse file with leading null bytes.
# Instead we delete + recreate the file and restart the collector so it
# reopens the new inode.

set -euo pipefail

NAME="${OC_OTEL_NAME:-oc-otel}"
DATA_DIR="${OC_OTEL_DATA_DIR:-/tmp/otel}"
TRACE_FILE="$DATA_DIR/otel-traces.jsonl"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    docker stop -t 5 "$NAME" >/dev/null
    rm -f "$TRACE_FILE"
    touch "$TRACE_FILE"
    chmod 666 "$TRACE_FILE" 2>/dev/null || true
    docker start "$NAME" >/dev/null
    # Poll readiness so the next run isn't racing against the receiver.
    for i in $(seq 1 30); do
        if curl -fsS -o /dev/null -X POST -H 'Content-Type: application/json' \
            --data '{}' http://127.0.0.1:4318/v1/traces 2>/dev/null; then
            echo "[otel-reset] ready after ${i}s" >&2
            exit 0
        fi
        sleep 1
    done
    echo "[otel-reset] OTel collector did not come back in 30s" >&2
    exit 2
else
    rm -f "$TRACE_FILE"
    touch "$TRACE_FILE"
    chmod 666 "$TRACE_FILE" 2>/dev/null || true
    echo "[otel-reset] collector not running; trace file truncated" >&2
fi
