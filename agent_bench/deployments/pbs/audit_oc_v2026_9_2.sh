#!/usr/bin/env bash
# Read-only structural audit for the OpenClaw v2026.9.2 base/candidate SIF.
#
# Base images are checked for the pinned upstream version and native runtime
# anchors. Candidate images that contain /app/.agent-bench-v2026.9.2.json are
# additionally checked for the complete Agent_Bench bridge patch set, absence
# of legacy v2026.4.22 loop/telemetry patches, and removal of public pricing
# endpoints.
#
# IMPORTANT: disable host HOME and inherited PATH. Singularity/Apptainer
# otherwise can report a host npm OpenClaw instead of the copy sealed in the SIF.

set -euo pipefail

SIF="${1:-}"
if [[ -z "$SIF" ]]; then
    echo "usage: $0 /path/to/openclaw_2026.9.2.sif" >&2
    exit 2
fi
[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF" >&2; exit 2; }

if command -v apptainer >/dev/null 2>&1; then
    OC_RUNTIME=apptainer
else
    module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
    command -v singularity >/dev/null 2>&1 || {
        echo "ERROR: neither apptainer nor singularity is on PATH" >&2
        exit 2
    }
    OC_RUNTIME=singularity
fi

run() {
    "$OC_RUNTIME" exec --cleanenv --no-home \
        --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        "$SIF" sh -lc "$1"
}

printf '== OpenClaw candidate audit ==\n'
printf 'runtime=%s\n' "$OC_RUNTIME"
printf 'sif=%s\n' "$SIF"
sha256sum "$SIF"

printf '\n== version ==\n'
PACKAGE_VERSION="$(run 'node -e '\''const p=require("/app/package.json"); process.stdout.write(String(p.version||""))'\''' 2>/dev/null || true)"
CLI_VERSION="$(run 'node /app/openclaw.mjs --version' 2>/dev/null || true)"
printf 'package.json: %s\n' "${PACKAGE_VERSION:-MISSING}"
printf 'openclaw.mjs: %s\n' "${CLI_VERSION:-MISSING}"

printf '\n== native loop detection ==\n'
NATIVE_LOOP_FILES="$(run 'grep -R -l -E "compaction_loop_persisted|post-compaction guard armed|post-compaction-guard" /app/dist 2>/dev/null | sort -u' || true)"
if [[ -n "$NATIVE_LOOP_FILES" ]]; then
    printf '%s\n' "$NATIVE_LOOP_FILES"
else
    echo 'MISSING: native post-compaction loop guard strings'
fi

ROLLING_LOOP_FILES="$(run 'grep -R -l -E "loopDetection|tool loop|loop detection" /app/dist 2>/dev/null | sort -u | head -40' || true)"
if [[ -n "$ROLLING_LOOP_FILES" ]]; then
    printf '%s\n' "$ROLLING_LOOP_FILES"
else
    echo 'MISSING: rolling loop-detection strings'
fi

printf '\n== exec backend anchors ==\n'
EXEC_FILES="$(run 'grep -R -l -E "sandbox backend does not provide buildExecSpec|buildExecSpec" /app/dist 2>/dev/null | sort -u' || true)"
if [[ -n "$EXEC_FILES" ]]; then
    printf '%s\n' "$EXEC_FILES"
else
    echo 'MISSING: buildExecSpec anchor'
fi

printf '\n== sandbox container anchors ==\n'
SANDBOX_FILES="$(run 'grep -R -l "ensureSandboxContainer" /app/dist 2>/dev/null | sort -u' || true)"
if [[ -n "$SANDBOX_FILES" ]]; then
    printf '%s\n' "$SANDBOX_FILES"
else
    echo 'MISSING: ensureSandboxContainer anchor'
fi

printf '\n== sandbox filesystem anchors ==\n'
FS_FILES="$(run 'grep -R -l -E "readPinnedFile|runCheckedCommand" /app/dist 2>/dev/null | sort -u' || true)"
if [[ -n "$FS_FILES" ]]; then
    printf '%s\n' "$FS_FILES"
else
    echo 'MISSING: readPinnedFile/runCheckedCommand anchors'
fi

printf '\n== native model/tool telemetry ==\n'
MODEL_OTEL_FILES="$(run 'grep -R -l -E "openclaw\.model\.usage|openclaw\.model\.call" /app/dist/extensions/diagnostics-otel /app/dist 2>/dev/null | sort -u | head -30' || true)"
TOOL_OTEL_FILES="$(run 'grep -R -l "openclaw.tool.execution" /app/dist/extensions/diagnostics-otel /app/dist 2>/dev/null | sort -u | head -30' || true)"
[[ -n "$MODEL_OTEL_FILES" ]] && printf 'model:\n%s\n' "$MODEL_OTEL_FILES" || echo 'MISSING: native model telemetry strings'
[[ -n "$TOOL_OTEL_FILES" ]] && printf 'tool:\n%s\n' "$TOOL_OTEL_FILES" || echo 'MISSING: native tool telemetry strings'

printf '\n== pi-ai OpenAI provider ==\n'
PI_AI_FILES="$(run 'find /app -type f -path "*/@mariozechner/pi-ai/dist/providers/openai-completions.js" -print 2>/dev/null | sort -u' || true)"
if [[ -n "$PI_AI_FILES" ]]; then
    printf '%s\n' "$PI_AI_FILES"
else
    echo 'INFO: legacy pi-ai openai-completions.js path absent; v2026.9.2 uses native OTel measurement path'
fi

CANDIDATE=0
if run 'test -f /app/.agent-bench-v2026.9.2.json' >/dev/null 2>&1; then
    CANDIDATE=1
fi

printf '\n== Agent_Bench v2026.9.2 candidate markers ==\n'
if [[ "$CANDIDATE" -eq 1 ]]; then
    run 'cat /app/.agent-bench-v2026.9.2.json'
    run 'grep -R -l "AB_HARNESS_OPENCLAW_V2026_9_2_" /app/dist 2>/dev/null | sort -u || true'
else
    echo 'base image: no candidate status marker (expected before patching)'
fi

printf '\n== legacy Agent_Bench markers ==\n'
LEGACY_MARKERS="$(run 'grep -R -l -E "AB_HARNESS_OPENCLAW_PATCH_POST_COMPACTION_LOOP_GUARD|AB_HARNESS_OPENCLAW_PATCH_MODEL_USAGE_BEGIN|AB_HARNESS_OPENCLAW_PATCH_GATEWAY_AGENT_USAGE_BEGIN|AB_HARNESS_OPENCLAW_PATCH_TOOL_CALL_BEGIN|AB_HARNESS_OPENCLAW_PATCH_TOOL_CALL_HANDLER_BEGIN" /app/dist /app/node_modules 2>/dev/null | sort -u | head -80' || true)"
if [[ -n "$LEGACY_MARKERS" ]]; then
    printf '%s\n' "$LEGACY_MARKERS"
else
    echo 'none'
fi

printf '\n== public pricing endpoints ==\n'
PRICING_FILES="$(run 'grep -R -l -E "https://openrouter\.ai/api/v1/models|https://raw\.githubusercontent\.com/BerriAI/litellm/main/model_prices_and_context_window\.json" /app/dist /app/packages /app/node_modules 2>/dev/null | sort -u | head -80' || true)"
if [[ -n "$PRICING_FILES" ]]; then
    printf '%s\n' "$PRICING_FILES"
else
    echo 'none'
fi

printf '\n== migration gate ==\n'
FAIL=0
if [[ "$PACKAGE_VERSION" != "2026.9.2" ]]; then
    echo "FAIL image_version package.json=${PACKAGE_VERSION:-MISSING} expected=2026.9.2"
    FAIL=1
else
    echo 'PASS image_version'
fi
if [[ -z "$NATIVE_LOOP_FILES" ]]; then
    echo 'FAIL native_loop_guard'
    FAIL=1
else
    echo 'PASS native_loop_guard'
fi
if [[ -z "$ROLLING_LOOP_FILES" ]]; then
    echo 'FAIL rolling_loop_detection'
    FAIL=1
else
    echo 'PASS rolling_loop_detection'
fi
if [[ -z "$EXEC_FILES" ]]; then
    echo 'FAIL exec_backend_anchor'
    FAIL=1
else
    echo 'PASS exec_backend_anchor'
fi
if [[ -z "$SANDBOX_FILES" ]]; then
    echo 'FAIL sandbox_container_anchor'
    FAIL=1
else
    echo 'PASS sandbox_container_anchor'
fi
if [[ -z "$FS_FILES" ]]; then
    echo 'FAIL sandbox_fs_anchor'
    FAIL=1
else
    echo 'PASS sandbox_fs_anchor'
fi
if [[ -z "$MODEL_OTEL_FILES" ]]; then
    echo 'FAIL native_model_otel'
    FAIL=1
else
    echo 'PASS native_model_otel'
fi
if [[ -z "$TOOL_OTEL_FILES" ]]; then
    echo 'FAIL native_tool_otel'
    FAIL=1
else
    echo 'PASS native_tool_otel'
fi

if [[ "$CANDIDATE" -eq 1 ]]; then
    for marker in \
        AB_HARNESS_OPENCLAW_V2026_9_2_EXEC_BRIDGE_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SANDBOX_BYPASS_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SHARED_WORKSPACE_SPAWN_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_POLICY_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_INSTRUCTION_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_RETRY_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_LENGTH_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_NONEMPTY_LENGTH_ELIGIBILITY_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_ASSISTANT_EMPTY_STOP_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_TERMINAL_PAYLOAD_V2_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_UNBOUNDED_LENGTH_CONTINUATION_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V2_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_TOOL_ERROR_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_EMPTY_STOP_RETRY_LIMIT_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_LOOP_VETO_RECOVERY_PROBE_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_HELPER_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_RUN_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_WRITE_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_READ_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_CHECK_BEGIN
    do
        if run "grep -R -q '$marker' /app/dist"; then
            echo "PASS candidate_marker $marker"
        else
            echo "FAIL candidate_marker $marker"
            FAIL=1
        fi
    done
    if [[ -n "$LEGACY_MARKERS" ]]; then
        echo 'FAIL legacy_v2026_4_22_markers_present'
        FAIL=1
    else
        echo 'PASS no_legacy_v2026_4_22_markers'
    fi
    if [[ -n "$PRICING_FILES" ]]; then
        echo 'FAIL public_pricing_endpoints_present'
        FAIL=1
    else
        echo 'PASS public_pricing_endpoints_absent'
    fi
else
    echo 'INFO base_image_only candidate bridge markers not required yet'
fi

exit "$FAIL"
