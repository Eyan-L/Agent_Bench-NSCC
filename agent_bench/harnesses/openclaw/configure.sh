#!/usr/bin/env bash
# agent_bench/harnesses/openclaw/configure.sh
#
# Configure OpenClaw to use the active Agent Bench serving endpoint and register
# a coding agent. Idempotent and safe to re-run.
#
# Env:
#   AB_HARNESS_OPENCLAW_BIN       openclaw CLI to use (default: from PATH)
#   AB_MODEL_CONFIG               model profile ref/path (default: qwen36_27b)
#   AB_SERVING_CONFIG             serving profile ref/path (default: vllm/default)
#   AB_SERVING_BASE_URL           OpenAI-compatible base URL override
#   AB_MODEL_NAME                 model id override
#   AB_AGENT_RUNTIME_AGENT_ID     agent id to register (default: swebench)
#   AB_AGENT_WORKSPACE            agent workspace dir (default: /testbed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

resolve_profile_ref() {
    local base="$1"
    local ref="$2"
    shift 2
    if [[ "$ref" == */*.* || "$ref" == /* ]]; then
        echo "$ref"
        return 0
    fi
    local ext
    for ext in "$@"; do
        if [[ -f "$base/$ref.$ext" ]]; then
            echo "$base/$ref.$ext"
            return 0
        fi
    done
    echo "$base/$ref.$1"
}

MODEL_CFG="${AB_MODEL_CONFIG:-qwen36_27b}"
MODEL_CFG="$(resolve_profile_ref "$AB_REPO_ROOT/profiles/models" "$MODEL_CFG" json yaml yml)"
if [[ -f "$MODEL_CFG" ]]; then
    eval "$(python3 "$AB_REPO_ROOT/scripts/load_model_env.py" "$MODEL_CFG")"
else
    echo "WARN: model config not found: $MODEL_CFG; using built-in defaults" >&2
fi

SERVING_CFG="${AB_SERVING_CONFIG:-vllm/default}"
SERVING_CFG="$(resolve_profile_ref "$AB_REPO_ROOT/profiles/serving" "$SERVING_CFG" yaml yml json toml)"
if [[ -f "$SERVING_CFG" ]]; then
    eval "$(python3 "$AB_REPO_ROOT/scripts/load_serving_env.py" "$SERVING_CFG")"
else
    echo "WARN: serving config not found: $SERVING_CFG; using built-in endpoint defaults" >&2
fi

OPENCLAW_BIN="${AB_HARNESS_OPENCLAW_BIN:-$(command -v openclaw || true)}"
if [[ -z "$OPENCLAW_BIN" || ! -x "$OPENCLAW_BIN" ]]; then
    echo "ERROR: openclaw not found. Set AB_HARNESS_OPENCLAW_BIN=/path/to/openclaw." >&2
    exit 1
fi

SERVING_BASE="${AB_SERVING_BASE_URL:-http://localhost:8000/v1}"
MODEL_NAME="${AB_MODEL_NAME:-Qwen/Qwen3.6-27B}"
AGENT_ID="${AB_AGENT_RUNTIME_AGENT_ID:-swebench}"
AGENT_WORKSPACE="${AB_AGENT_WORKSPACE:-/testbed}"

printf '[openclaw_configure] binary   : %s\n' "$OPENCLAW_BIN"
printf '[openclaw_configure] endpoint : %s\n' "$SERVING_BASE"
printf '[openclaw_configure] model    : %s\n' "$MODEL_NAME"
printf '[openclaw_configure] agent    : %s  (workspace: %s)\n\n' "$AGENT_ID" "$AGENT_WORKSPACE"

AGENT_MODEL_PROVIDER_JSON=$(AB_SERVING_BASE_URL="$SERVING_BASE" AB_MODEL_NAME="$MODEL_NAME" AB_SERVING_API_KEY="${AB_SERVING_API_KEY:-sk-dummy}" python3 -c '
import json, os
print(json.dumps({
    "baseUrl": os.environ["AB_SERVING_BASE_URL"],
    "api": "openai-completions",
    "apiKey": os.environ["AB_SERVING_API_KEY"],
    "models": [{
        "id": os.environ["AB_MODEL_NAME"],
        "name": os.environ["AB_MODEL_NAME"],
        "api": "openai-completions",
    }],
}))
')
"$OPENCLAW_BIN" config set --strict-json 'models.providers.agent_model' "$AGENT_MODEL_PROVIDER_JSON"

AGENT_JSON=$(AB_AGENT_RUNTIME_AGENT_ID="$AGENT_ID" AB_AGENT_WORKSPACE="$AGENT_WORKSPACE" python3 -c '
import json, os
print(json.dumps({
    "id": os.environ["AB_AGENT_RUNTIME_AGENT_ID"],
    "default": True,
    "name": os.environ["AB_AGENT_RUNTIME_AGENT_ID"] + "-coder",
    "workspace": os.environ["AB_AGENT_WORKSPACE"],
    "tools": {"profile": "coding", "exec": {"host": "node", "security": "full"}},
}))
')
"$OPENCLAW_BIN" config set --strict-json 'agents.list[0]' "$AGENT_JSON"
"$OPENCLAW_BIN" config set agents.defaults.model.primary "agent_model/$MODEL_NAME" || true
"$OPENCLAW_BIN" config validate

echo
echo "---- resulting models.providers ----"
"$OPENCLAW_BIN" config get models.providers || true
echo
echo "---- resulting agents.list ----"
"$OPENCLAW_BIN" config get agents.list || true
echo
echo "---- resulting agents.defaults ----"
"$OPENCLAW_BIN" config get agents.defaults || true

echo
echo "[openclaw_configure] done."
echo "Next step: test with"
echo "  $OPENCLAW_BIN agent --local --agent $AGENT_ID --message 'What is 2+2?' --json | jq ."
