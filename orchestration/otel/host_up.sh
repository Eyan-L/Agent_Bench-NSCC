#!/usr/bin/env bash
# Start otelcol-contrib as a host process (no docker). Idempotent.
#
# Downloads the binary on first run into ~/.cache/oc-orchestration/otelcol/ if not
# already present. Logs to /tmp/otel/otelcol.log, PID to /tmp/otel/otelcol.pid.
#
# Env:
#   OC_OTEL_VERSION    collector release tag (default: v0.115.1)
#   OC_OTEL_DATA_DIR   trace output dir (default: /tmp/otel)
#   OC_OTEL_CONFIG     config file (default: <repo>/orchestration/otel/collector-config.yaml)

set -euo pipefail

KIT_ROOT="${KIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERSION="${OC_OTEL_VERSION:-v0.115.1}"
DATA_DIR="${OC_OTEL_DATA_DIR:-/tmp/otel}"
CONFIG="${OC_OTEL_CONFIG:-$KIT_ROOT/orchestration/otel/collector-config.yaml}"
CACHE="${OC_OTEL_CACHE:-$HOME/.cache/oc-orchestration/otelcol}"

mkdir -p "$DATA_DIR" "$CACHE"
touch "$DATA_DIR/otel-traces.jsonl"

if [ ! -f "$CONFIG" ]; then
    echo "[otel-up-host] config missing: $CONFIG" >&2
    exit 1
fi

PID_FILE="$DATA_DIR/otelcol.pid"
LOG_FILE="$DATA_DIR/otelcol.log"

# Already running?
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[otel-up-host] already running (pid $(cat "$PID_FILE"))" >&2
    exit 0
fi

# Detect platform.
case "$(uname -s)" in
    Linux)  OS=linux ;;
    Darwin) OS=darwin ;;
    *) echo "[otel-up-host] unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
    x86_64|amd64) ARCH=amd64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) echo "[otel-up-host] unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

BIN="$CACHE/otelcol-contrib-${VERSION}-${OS}-${ARCH}"
if [ ! -x "$BIN" ]; then
    URL="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/${VERSION}/otelcol-contrib_${VERSION#v}_${OS}_${ARCH}.tar.gz"
    echo "[otel-up-host] downloading $URL" >&2
    TMP=$(mktemp -d)
    curl -fsSL "$URL" -o "$TMP/otelcol.tar.gz"
    tar -xzf "$TMP/otelcol.tar.gz" -C "$TMP"
    mv "$TMP/otelcol-contrib" "$BIN"
    chmod +x "$BIN"
    rm -rf "$TMP"
fi

# Bail early if anything else already binds the OTLP ports — otherwise we'd
# silently lose all spans (collector dies, but our readiness probe would hit
# the squatter and report success).
for p in 4317 4318; do
    if ss -tln "( sport = :$p )" 2>/dev/null | grep -q LISTEN; then
        echo "[otel-up-host] port $p already in use (another collector running?)" >&2
        ss -tlnp 2>/dev/null | grep ":$p " >&2 || true
        exit 1
    fi
done

echo "[otel-up-host] starting $BIN (config=$CONFIG)" >&2
nohup "$BIN" --config "$CONFIG" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

# Poll readiness: collector logs "Everything is ready. Begin running and processing data."
# Also verify our PID is still alive — guards against bind failure.
for i in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[otel-up-host] collector exited early; tail of $LOG_FILE:" >&2
        tail -30 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 2
    fi
    if grep -q "Everything is ready" "$LOG_FILE" 2>/dev/null; then
        echo "[otel-up-host] ready (pid $PID) after ${i}s" >&2
        exit 0
    fi
    sleep 1
done

echo "[otel-up-host] collector did not become ready in 30s; tail of $LOG_FILE:" >&2
tail -30 "$LOG_FILE" >&2
exit 2
