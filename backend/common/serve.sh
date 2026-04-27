#!/usr/bin/env bash
# Backend dispatcher. Set SERVING_BACKEND={vllm|sglang}, then forward args.
set -euo pipefail

: "${SERVING_BACKEND:?must set SERVING_BACKEND to one of: vllm, sglang}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/../${SERVING_BACKEND}/serve.sh"

if [[ ! -x "${TARGET}" ]]; then
  echo "[backend] no launcher for SERVING_BACKEND=${SERVING_BACKEND} (expected ${TARGET})" >&2
  exit 2
fi

exec "${TARGET}" "$@"
