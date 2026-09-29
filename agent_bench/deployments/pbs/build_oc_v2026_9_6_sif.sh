#!/usr/bin/env bash
# Build the pinned OpenClaw 2026.9.6 SIF using the existing v9 bridge family.
# Version gates inspect package.json and the sealed status record; bridge
# markers retain their v9.2 names for compatibility with existing collectors.
set -euo pipefail

AB_BUILD_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$AB_BUILD_SCRIPT_DIR/build_oc_v2026_9_2_sif.sh" "$@" --version 2026.9.6
