#!/usr/bin/env bash
# Backend dispatcher. Set SERVING_BACKEND={vllm|sglang|litellm}, then forward args.
#
# Supports a generic `stop` action: forwards to the backend's own stop
# handler if it has one (sglang does), otherwise falls back to killing
# whatever's listening on $PORT (default 8000) — covers vllm which runs in
# the foreground and has no native stop subcommand.
set -euo pipefail

: "${SERVING_BACKEND:?must set SERVING_BACKEND to one of: vllm, sglang, litellm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/../${SERVING_BACKEND}/serve.sh"

if [[ ! -x "${TARGET}" ]]; then
  echo "[backend] no launcher for SERVING_BACKEND=${SERVING_BACKEND} (expected ${TARGET})" >&2
  exit 2
fi

ACTION="${1:-start}"

if [[ "$ACTION" == "stop" ]]; then
    # Backend-specific stop, when supported (e.g. sglang has its own case).
    if grep -qE "^[[:space:]]*stop\)" "$TARGET" 2>/dev/null; then
        exec "$TARGET" stop
    fi
    # Port-based fallback (covers vllm, which has no stop action).
    port="${PORT:-8000}"
    pid="$(ss -tlnp 2>/dev/null | awk -v p=":${port} " '$0 ~ p {match($0, /pid=([0-9]+)/, m); if (m[1]) {print m[1]; exit}}')"
    if [[ -n "$pid" ]]; then
        echo "[backend] stopping ${SERVING_BACKEND} (pid=${pid} on port ${port})"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 15); do
            sleep 1
            kill -0 "$pid" 2>/dev/null || break
        done
        kill -9 "$pid" 2>/dev/null || true
    else
        echo "[backend] no ${SERVING_BACKEND} listener on port ${port} — nothing to stop"
    fi
    exit 0
fi

exec "${TARGET}" "$@"
