#!/usr/bin/env bash
# Read-only structural audit for the OpenClaw v2026.9.2 base/candidate SIF.
#
# This intentionally does not modify the image. It answers the migration
# questions we need before porting Agent_Bench's NSCC bridge patches:
#   - is this really OpenClaw 2026.9.2?
#   - is upstream loop detection/post-compaction guard present?
#   - where are the compiled exec/sandbox/fs anchors in this image?
#   - where is pi-ai's OpenAI-compatible provider implementation?
#   - which legacy Agent_Bench markers are already present?

set -euo pipefail

SIF="${1:-}"
if [[ -z "$SIF" ]]; then
    echo "usage: $0 /path/to/openclaw_2026.9.2.sif" >&2
    exit 2
fi
[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF" >&2; exit 2; }

module load singularity/4.3.1 2>/dev/null || module load singularity 2>/dev/null || true
command -v singularity >/dev/null 2>&1 || {
    echo "ERROR: singularity not on PATH" >&2
    exit 2
}

run() {
    singularity exec "$SIF" sh -lc "$1"
}

printf '== OpenClaw candidate audit ==\n'
printf 'sif=%s\n' "$SIF"
sha256sum "$SIF"

printf '\n== version ==\n'
run 'openclaw --version 2>/dev/null || node /app/openclaw.mjs --version 2>/dev/null || true'

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

printf '\n== pi-ai OpenAI provider ==\n'
PI_AI_FILES="$(run 'find /app -type f -path "*/@mariozechner/pi-ai/dist/providers/openai-completions.js" -print 2>/dev/null | sort -u' || true)"
if [[ -n "$PI_AI_FILES" ]]; then
    printf '%s\n' "$PI_AI_FILES"
else
    echo 'MISSING: pi-ai openai-completions.js'
fi

printf '\n== existing Agent_Bench markers ==\n'
run 'grep -R -l "AB_HARNESS_OPENCLAW_PATCH_" /app/dist /app/node_modules 2>/dev/null | sort -u | head -80 || true'

printf '\n== migration gate ==\n'
FAIL=0
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
if [[ -z "$PI_AI_FILES" ]]; then
    echo 'WARN pi_ai_provider_missing (raw-response patch strategy may need to change)'
else
    echo 'PASS pi_ai_provider'
fi

exit "$FAIL"
