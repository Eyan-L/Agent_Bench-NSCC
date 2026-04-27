#!/usr/bin/env bash
# Apply sglang patch overlay to a target site-packages directory.
# Mode controlled by SGLANG_PATCH_MODE: bind | copy | baked | skip
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/defaults.env"

if [[ "${SGLANG_PATCH_MODE}" == "skip" || "${SGLANG_PATCH_MODE}" == "baked" ]]; then
  echo "[sglang patch] mode=${SGLANG_PATCH_MODE}, nothing to do" >&2
  exit 0
fi

if [[ -z "${SGLANG_SITE_PACKAGES}" ]]; then
  SGLANG_SITE_PACKAGES="$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')"
fi

if [[ ! -d "${SGLANG_SITE_PACKAGES}/sglang" ]]; then
  echo "[sglang patch] error: ${SGLANG_SITE_PACKAGES}/sglang not found" >&2
  exit 1
fi

case "${SGLANG_PATCH_MODE}" in
  bind)
    echo "[sglang patch] bind mode: caller must mount ${SGLANG_PATCH_DIR} -> ${SGLANG_SITE_PACKAGES}/sglang (ro)" >&2
    ;;
  copy)
    echo "[sglang patch] copy mode: cp ${SGLANG_PATCH_DIR}/ -> ${SGLANG_SITE_PACKAGES}/sglang/" >&2
    cp -rv "${SGLANG_PATCH_DIR}/." "${SGLANG_SITE_PACKAGES}/sglang/"
    ;;
  *)
    echo "[sglang patch] unknown mode: ${SGLANG_PATCH_MODE}" >&2
    exit 2
    ;;
esac
