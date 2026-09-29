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
EXPECTED_VERSION="${2:-2026.9.2}"
case "$EXPECTED_VERSION" in
    2026.9.2|2026.9.6) ;;
    *) echo "ERROR: unsupported OpenClaw audit version: $EXPECTED_VERSION" >&2; exit 2 ;;
esac
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

if [[ "$EXPECTED_VERSION" == "2026.9.6" ]]; then
    OC_AUDIT_DIR="$(cd "$(dirname "$0")" && pwd)"
    "$OC_RUNTIME" exec --cleanenv --no-home \
        --bind "$OC_AUDIT_DIR/audit_oc_v2026_9_6.mjs:/opt/agent-bench-audit.mjs:ro" \
        "$SIF" node /opt/agent-bench-audit.mjs
    exit $?
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
if [[ "$PACKAGE_VERSION" != "$EXPECTED_VERSION" ]]; then
    echo "FAIL image_version package.json=${PACKAGE_VERSION:-MISSING} expected=$EXPECTED_VERSION"
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
        AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_ASSISTANT_EMPTY_STOP_V2_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_CONTINUATION_DECISION_DIAGNOSTIC_V1_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V2_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_IMPLICIT_CHILD_HANDOFF_OWNERSHIP_V1_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_BOUNDED_NO_PROGRESS_LENGTH_V1_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_LENGTH_SEMANTIC_PROGRESS_V2_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V7_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V5_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_TRUNCATE_ONLY_NATIVE_RECOVERY_V1_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V3_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_TOOL_ERROR_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_EMPTY_STOP_RETRY_LIMIT_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_LOOP_VETO_RECOVERY_PROBE_BEGIN \
        AB_HARNESS_OPENCLAW_V2026_9_2_SESSIONS_YIELD_NO_PENDING_LOOP_V3_BEGIN \
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
    if run 'set -e
        error="accepted continuation children were not durably registered"
        files="$(grep -R -l "$error" /app/dist 2>/dev/null)"
        test -n "$files"
        for file in $files; do
            calls="$(grep -o "$error" "$file" | wc -l)"
            markers="$(grep -o "AB_HARNESS_OPENCLAW_V2026_9_2_IMPLICIT_CHILD_HANDOFF_OWNERSHIP_V1_BEGIN" "$file" | wc -l)"
            assignments="$(grep -o -E "[A-Za-z_$][A-Za-z0-9_$]*\.runtimeContinuationStarted[[:space:]]*=[[:space:]]*true;" "$file" | wc -l)"
            test "$calls" -gt 0
            test "$calls" -eq "$markers"
            test "$calls" -le "$assignments"
        done'; then
        echo 'PASS implicit_child_handoff_ownership_contract'
    else
        echo 'FAIL implicit_child_handoff_ownership_contract'
        FAIL=1
    fi
    if run 'set -e
        files="$(grep -R -l "function resolveEmptyResponseRetryInstruction(" /app/dist 2>/dev/null)"
        test -n "$files"
        for file in $files; do
            definitions="$(grep -o "function resolveEmptyResponseRetryInstruction(" "$file" | wc -l)"
            markers="$(grep -o "AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_ASSISTANT_EMPTY_STOP_V2_BEGIN" "$file" | wc -l)"
            owner_markers="$(grep -o "AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V2_BEGIN" "$file" | wc -l)"
            test "$definitions" -eq "$markers"
            test "$definitions" -eq "$owner_markers"
            ! grep -q "tools_unsettled" "$file"
            owner_contract="$(sed -n "/AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V2_BEGIN/,/AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V2_END/p" "$file")"
            printf "%s" "$owner_contract" | grep -q "runtimeContinuationStarted === true"
            printf "%s" "$owner_contract" | grep -q "&& !__abRuntimeContinuationOwner"
            printf "%s" "$owner_contract" | grep -q "runtime_continuation_owner"
            printf "%s" "$owner_contract" | grep -q "&& __abRoadmapContinuationEligible) return null;"
            ! printf "%s" "$owner_contract" | grep -q "acceptedSessionSpawns"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V1_BEGIN" "$file"
            for helper in classifyAssistantTurn resolveSettledToolBatchEvidence hasCompletedMessagingToolDeliveryEvidence hasAsyncActivity; do
                grep -q -E "function[[:space:]]+$helper[[:space:]]*\\(|(const|let|var)[[:space:]]+$helper[[:space:]]*=|as[[:space:]]+$helper([^A-Za-z0-9_$]|$)|import[[:space:]]+$helper([[:space:],]|$)" "$file"
            done
        done'; then
        echo 'PASS all_resolver_definitions_have_current_empty_logic_and_bindings'
    else
        echo 'FAIL resolver_definition_current_empty_coverage_or_bindings'
        FAIL=1
    fi
    if run 'set -e
        files="$(grep -R -l "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V3_BEGIN" /app/dist 2>/dev/null)"
        test -n "$files"
        for file in $files; do
            reset_contract="$(sed -n "/AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V3_BEGIN/,/AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V3_END/p" "$file")"
            printf "%s" "$reset_contract" | grep -q "runtimeContinuationStarted === true"
            ! printf "%s" "$reset_contract" | grep -q "acceptedSessionSpawns"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V2_BEGIN" "$file"
        done'; then
        echo 'PASS consecutive_empty_reset_runtime_ownership_contract'
    else
        echo 'FAIL consecutive_empty_reset_runtime_ownership_contract'
        FAIL=1
    fi
    if run 'set -e
        files="$(grep -R -l "function detectToolCallLoop" /app/dist 2>/dev/null)"
        test -n "$files"
        for file in $files; do
            definitions="$(grep -o "function detectToolCallLoop" "$file" | wc -l)"
            markers="$(grep -o "AB_HARNESS_OPENCLAW_V2026_9_2_SESSIONS_YIELD_NO_PENDING_LOOP_V3_BEGIN" "$file" | wc -l)"
            test "$definitions" -eq "$markers"
            grep -q "sessions-yield-no-pending" "$file"
            grep -q "sessions_yield_no_pending" "$file"
            grep -q "No pending child completion is owned by this turn" "$file"
            grep -q "yieldOwnershipEpoch" "$file"
            grep -q "scope?.runId" "$file"
            grep -q "byRunId" "$file"
            grep -q "params.toolCallId" "$file"
            grep -q "loopState.count < 10" "$file"
            grep -q "loopState.count < CRITICAL_THRESHOLD" "$file"
            ownership_contract="$(sed -n "/AB_HARNESS_OPENCLAW_V2026_9_2_SESSIONS_YIELD_NO_PENDING_LOOP_V3_BEGIN/,/AB_HARNESS_OPENCLAW_V2026_9_2_SESSIONS_YIELD_NO_PENDING_LOOP_V3_END/p" "$file")"
            ! printf "%s" "$ownership_contract" | grep -q "generation"
            ! printf "%s" "$ownership_contract" | grep -q "criticalEmitted"
            ! printf "%s" "$ownership_contract" | grep -q "probeConsumed"
            ! printf "%s" "$ownership_contract" | grep -q "probeOffered"
            ! printf "%s" "$ownership_contract" | grep -q "ProjectedToolLoopAdmission"
            printf "%s" "$ownership_contract" | grep -q "__abPeekSessionsYieldNoPendingRunState"
            printf "%s" "$ownership_contract" | grep -q "staleHistoryByRunId"
            printf "%s" "$ownership_contract" | grep -q "sessions_spawn"
            printf "%s" "$ownership_contract" | grep -q "expectsCompletionMessage"
            printf "%s" "$ownership_contract" | grep -q "__abSessionsYieldSeenOwnershipIdsKey"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_SESSIONS_YIELD_NO_PENDING_LOOP_V1_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_SESSIONS_YIELD_NO_PENDING_LOOP_V2_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_SESSIONS_YIELD_PROBE_PROJECTION_V1_BEGIN" "$file"
        done'; then
        echo 'PASS sessions_yield_no_pending_loop_contract'
    else
        echo 'FAIL sessions_yield_no_pending_loop_contract'
        FAIL=1
    fi
    if run 'set -e
        files="$(grep -R -l "function recoverEmbeddedRunOverflow(" /app/dist 2>/dev/null)"
        test -n "$files"
        for file in $files; do
            overflow_definitions="$(grep -o "function recoverEmbeddedRunOverflow(" "$file" | wc -l)"
            attempt_definitions="$(grep -o "function recoverEmbeddedRunAttempt(" "$file" | wc -l)"
            pressure_markers="$(grep -o "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V7_BEGIN" "$file" | wc -l)"
            reset_markers="$(grep -o "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V5_BEGIN" "$file" | wc -l)"
            test "$overflow_definitions" -eq 1
            test "$attempt_definitions" -eq 1
            test "$overflow_definitions" -eq "$pressure_markers"
            test "$attempt_definitions" -eq "$reset_markers"
            grep -q "function compactEmbeddedRunForRecovery(" "$file"
            grep -q "function recoverEmbeddedRunAttempt(" "$file"
            grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V7_BEGIN" "$file"
            grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V5_BEGIN" "$file"
            grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_TRUNCATE_ONLY_NATIVE_RECOVERY_V1_BEGIN" "$file"
            grep -q "attempt?.preflightRecovery ?? attempt?.contextBudgetStatus" "$file"
            grep -q "attempt?.assistantTexts ?? \[\]" "$file"
            grep -q "accumulated-empty" "$file"
            grep -q "hasNonToolTerminalState({" "$file"
            grep -q "acceptedSessionSpawns: \[\]" "$file"
            ! grep -q "hasCompletionOwningSpawn" "$file"
            grep -q "runtimeContinuationStarted === true" "$file"
            grep -q "currentReasoningOnly" "$file"
            grep -q "explicitDone" "$file"
            for helper in resolveSettledToolBatchEvidence hasNonToolTerminalState; do
                grep -q -E "function[[:space:]]+$helper[[:space:]]*\\(|(const|let|var)[[:space:]]+$helper[[:space:]]*=|as[[:space:]]+$helper([^A-Za-z0-9_$]|$)|import[[:space:]]+$helper([[:space:],]|$)" "$file"
            done
            if grep -q "preflightPressure" "$file"; then
                :
            else
                test "$(grep -o "agentBenchContextPressure" "$file" | wc -l)" -ge 2
                grep -q "Context window exceeded under preflight pressure" "$file"
            fi
            grep -q "agent-bench-terminal" "$file"
            grep -q "truncate_tool_results_only" "$file"
            grep -q -E "lengthNoProgressAttempts[[:space:]]*=[[:space:]]*0" "$file"
            grep -q -E "emptyResponseAttempts[[:space:]]*=[[:space:]]*0" "$file"
            ! grep -q "input.normalizedAttempt" "$file"
            ! grep -q "hasOutboundDeliveryEvidence(attempt)" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_LENGTH_RECOVERY_V1_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_COMPACTION_LENGTH_STREAK_RESET_V1_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V2_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V3_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V4_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V5_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V6_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_COMPACTION_RETRY_STREAK_RESET_V2_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V3_BEGIN" "$file"
            ! grep -q "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V4_BEGIN" "$file"
        done'; then
        echo 'PASS context_pressure_terminal_native_recovery_contract'
    else
        echo 'FAIL context_pressure_terminal_native_recovery_contract'
        FAIL=1
    fi
    if run "grep -R -q -E 'AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_ASSISTANT_EMPTY_STOP_BEGIN|AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_TERMINAL_PAYLOAD_V2_BEGIN|hasCurrentTerminalPayload' /app/dist"; then
        echo 'FAIL obsolete_cross_bundle_current_empty_contract_present'
        FAIL=1
    else
        echo 'PASS no_obsolete_cross_bundle_current_empty_contract'
    fi
    if run "grep -R -q 'AB_HARNESS_OPENCLAW_V2026_9_2_UNBOUNDED_LENGTH_CONTINUATION_BEGIN' /app/dist"; then
        echo 'FAIL stale_unbounded_length_continuation_present'
        FAIL=1
    else
        echo 'PASS no_stale_unbounded_length_continuation'
    fi
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
