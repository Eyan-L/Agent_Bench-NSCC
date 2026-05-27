#!/usr/bin/env bash
# Compatibility wrapper. Python owns LiteLLM profile resolution and command planning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export KIT_ROOT

exec python3 -m agent_bench.serving.launch --backend litellm "$@"
