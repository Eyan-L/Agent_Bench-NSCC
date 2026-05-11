#!/usr/bin/env bash
# harnesses/openclaw/configure.sh
#
# Configure OpenClaw to use vLLM as the model provider and register a coding
# agent. Idempotent — safe to re-run.
#
# Sources model + endpoint from the active harness config (HARNESS env var
# or explicit VLLM_HARNESS_YAML), keeping OC and serve.sh in lockstep. Env
# overrides still win.
#
# Env:
#   OC_BIN              openclaw CLI to use (default: from PATH)
#   HARNESS             config/harness/$HARNESS.json (default: openclaw_27b)
#   VLLM_HARNESS_YAML   explicit harness config path (skips $HARNESS lookup;
#                       var name kept for back-compat)
#   VLLM_BASE           vLLM OpenAI-compat URL (overrides config model.endpoint)
#   VLLM_MODEL          model id (overrides config model.name)
#   OC_AGENT_ID         agent id to register (default: swebench)
#   OC_WORKSPACE        agent workspace dir (default: /testbed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Source MODEL/endpoint from the active harness config (single source of
# truth for model selection). Env override still wins.
HARNESS_CFG="${VLLM_HARNESS_YAML:-}"
if [[ -z "$HARNESS_CFG" ]]; then
    HARNESS_CFG="$KIT_ROOT/config/harness/${HARNESS:-openclaw_27b}.json"
fi
if [[ -f "$HARNESS_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_harness_env.py" "$HARNESS_CFG")"
else
    echo "WARN: harness config not found: $HARNESS_CFG — using built-in defaults" >&2
fi

OC="${OC_BIN:-$(command -v openclaw || true)}"
if [[ -z "$OC" ]]; then
    echo "ERROR: openclaw not found on PATH. Set OC_BIN=/path/to/openclaw." >&2
    exit 1
fi
VLLM_BASE="${VLLM_BASE:-${MODEL_ENDPOINT:-http://localhost:8000/v1}}"
VLLM_MODEL="${VLLM_MODEL:-${MODEL:-Qwen/Qwen3.6-27B}}"
OC_AGENT_ID="${OC_AGENT_ID:-swebench}"
OC_WORKSPACE="${OC_WORKSPACE:-/testbed}"

if [[ ! -x "$OC" ]]; then
    echo "ERROR: openclaw not found at $OC (set OC_BIN to override)" >&2
    exit 1
fi

echo "[openclaw_configure] binary   : $OC"
echo "[openclaw_configure] vllm     : $VLLM_BASE"
echo "[openclaw_configure] model    : $VLLM_MODEL"
echo "[openclaw_configure] agent    : $OC_AGENT_ID  (workspace: $OC_WORKSPACE)"
echo

# -----------------------------------------------------------------------------
# 1. Register a "vllm" model provider pointing at the local OpenAI-compat endpoint
# -----------------------------------------------------------------------------
# Per schema: models.providers is a dict keyed by provider name.
# Each provider has { baseUrl, headers, ... }.
# We register a provider named "vllm".
# OpenClaw's OpenAI-compat adapter will use baseUrl + any auth headers to reach it.

"$OC" config set models.providers.vllm.baseUrl "$VLLM_BASE"

# vLLM ignores the API key but OpenClaw's provider adapter may demand one.
# Set a dummy — any non-empty string works for the HTTP handshake.
# Compose JSON via python to avoid shell-escape headaches with model ids
# that contain slashes or quotes.
VLLM_PROVIDER_JSON=$(VLLM_BASE="$VLLM_BASE" VLLM_MODEL="$VLLM_MODEL" python3 -c '
import json, os
print(json.dumps({
    "baseUrl": os.environ["VLLM_BASE"],
    "apiKey": "sk-dummy",
    "models": [{"id": os.environ["VLLM_MODEL"], "name": os.environ["VLLM_MODEL"]}],
}))
')
"$OC" config set --strict-json 'models.providers.vllm' "$VLLM_PROVIDER_JSON"
# -----------------------------------------------------------------------------
# 2. Register a coding agent whose default model points at the vllm provider
# -----------------------------------------------------------------------------
# Per schema: agents.list is an array of { id, workspace, tools, ... }.
# We set the first entry to our swebench agent.
# Using strict JSON batch so we don't fight escaping.

$OC config set --strict-json 'agents.list[0]' '{
  "id": "swebench",
  "default": true,
  "name": "swebench-coder",
  "workspace": "./testbed",
  "tools": {
    "profile": "coding",
    "exec": {"host": "node", "security": "full"}
  }
}'

$OC config validate

# -----------------------------------------------------------------------------
# 3. Point agents.defaults at the vllm provider + model
# -----------------------------------------------------------------------------
# The exact key name for default provider/model isn't obvious from the schema
# excerpt — try both likely spellings. Failures are tolerated; whichever one
# the installed OpenClaw version accepts will stick.


# -----------------------------------------------------------------------------
# 4. Validate the config so a broken schema fails here instead of at run time
# -----------------------------------------------------------------------------
"$OC" config validate

# -----------------------------------------------------------------------------
# 5. Show the resulting model/agent sections for confirmation
# -----------------------------------------------------------------------------
echo
echo "---- resulting models.providers ----"
"$OC" config get models.providers || true
echo
echo "---- resulting agents.list ----"
"$OC" config get agents.list || true
echo
echo "---- resulting agents.defaults ----"
"$OC" config get agents.defaults || true

echo
echo "[openclaw_configure] done."
echo "Next step: test with"
echo "  $OC agent --local --agent $OC_AGENT_ID --message 'What is 2+2?' --json | jq ."