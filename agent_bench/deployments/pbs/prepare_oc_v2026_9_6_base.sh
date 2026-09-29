#!/usr/bin/env bash
# Prepare the official OpenClaw 2026.9.6 base locally, using the v9 build workflow.
set -euo pipefail
AB_PREPARE_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$AB_PREPARE_SCRIPT_DIR/prepare_oc_v2026_9_2_base.sh" "$@" --version 2026.9.6
