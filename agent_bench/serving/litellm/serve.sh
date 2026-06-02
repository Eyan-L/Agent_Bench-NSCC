#!/usr/bin/env bash
# Python owns LiteLLM config resolution and command planning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export AB_REPO_ROOT

exec python3 -m agent_bench.serving.launch --backend litellm "$@"
