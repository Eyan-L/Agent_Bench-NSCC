#!/usr/bin/env bash
# Replay agent-trace replay plans against a running sglang server using the
# unified loadgen at scripts/go/loadgen (agent subcommand). This replaces
# scripts/python/trace_load_generator_v2.py as the primary entry point; the
# Python script is kept around for emergency fallback.
#
# Typical usage:
#
#   AGENT_DIRS=/path/to/agent_dirs.json run/replay_agent_trace.sh
#   MODE=attempt CONCURRENT_AGENTS=4 run/replay_agent_trace.sh
#
# The script:
#   1. ensures the loadgen binary is built (same binary used by the Alibaba
#      and Azure scripts);
#   2. starts sglang via run/sglang_launcher.sh (no-op if already up);
#   3. runs `loadgen agent` with the chosen settings;
#   4. leaves sglang running unless STOP_SGLANG=1 is set.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GO_BIN="${GO_BIN:-/mnt/vdb/hongrui/opt/go/bin/go}"
GOPATH_DIR="${GOPATH_DIR:-/mnt/vdb/hongrui/go}"
GOCACHE_DIR="${GOCACHE_DIR:-/mnt/vdb/hongrui/go-build}"

LOADGEN_DIR="${PROJECT_ROOT}/loadgen"
LOADGEN_BIN="${LOADGEN_DIR}/loadgen-bin"

AGENT_DIRS="${AGENT_DIRS:-${PROJECT_ROOT}/agent_dirs/example.json}"

# sglang knobs forwarded to run/sglang_launcher.sh.
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-50}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
export SGLANG_PORT SGLANG_HOST SGLANG_MODEL_PATH SGLANG_CUDA_VISIBLE_DEVICES \
       SGLANG_MEM_FRACTION_STATIC SGLANG_PAGE_SIZE SGLANG_HICACHE_SIZE \
       SGLANG_TOOL_CALL_PARSER SGLANG_EXTRA_ARGS

API_BASE="${API_BASE:-http://127.0.0.1:${SGLANG_PORT}/v1}"
MODE="${MODE:-sync}"                  # sync | attempt
CONCURRENT_AGENTS="${CONCURRENT_AGENTS:-2}"
MAX_AGENTS="${MAX_AGENTS:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-1024}"
RESPONSE_OUTPUT_DIR="${RESPONSE_OUTPUT_DIR:-cache}"
AGENT_START_STAGGER="${AGENT_START_STAGGER:-}"
STAGGER_SEED="${STAGGER_SEED:-0}"
TIMEOUT="${TIMEOUT:-3000}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/vdb/hongrui/replay_results}"
WAIT_SGLANG_SECS="${WAIT_SGLANG_SECS:-1200}"
STOP_SGLANG="${STOP_SGLANG:-0}"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

build_loadgen() {
    if [[ -x "${LOADGEN_BIN}" ]]; then
        local newest_src
        newest_src="$(find "${LOADGEN_DIR}" -name '*.go' -type f -newer "${LOADGEN_BIN}" -print -quit)"
        if [[ -z "${newest_src}" ]]; then
            return 0
        fi
    fi
    if [[ ! -x "${GO_BIN}" ]]; then
        echo "error: go binary not found at ${GO_BIN}. Set GO_BIN." >&2
        exit 1
    fi
    log "building loadgen with $(${GO_BIN} version)"
    (
        cd "${LOADGEN_DIR}"
        GOPATH="${GOPATH_DIR}" GOCACHE="${GOCACHE_DIR}" "${GO_BIN}" build -o "${LOADGEN_BIN}" ./cmd/loadgen
    )
}

ensure_sglang_up() {
    if curl -fsS "${API_BASE}/models" -o /dev/null 2>&1; then
        log "sglang already responding at ${API_BASE}"
        return 0
    fi
    log "starting sglang via run/sglang_launcher.sh"
    bash "${SCRIPT_DIR}/sglang_launcher.sh" start

    log "waiting up to ${WAIT_SGLANG_SECS}s for ${API_BASE}/models"
    local waited=0
    while (( waited < WAIT_SGLANG_SECS )); do
        if curl -fsS "${API_BASE}/models" -o /dev/null 2>&1; then
            log "sglang is up after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "error: sglang did not become ready within ${WAIT_SGLANG_SECS}s" >&2
    exit 1
}

run_loadgen() {
    if [[ ! -f "${AGENT_DIRS}" ]]; then
        echo "error: agent-dirs JSON not found: ${AGENT_DIRS}" >&2
        exit 1
    fi

    mkdir -p "${OUTPUT_DIR}"
    local dirs_base
    dirs_base="$(basename "${AGENT_DIRS}" .json)"
    local stamp
    stamp="$(date '+%Y%m%d_%H%M%S')"
    local output_path="${OUTPUT_DIR}/agent_${dirs_base}_${MODE}_${stamp}.jsonl"

    local args=(
        agent
        -agent-dirs-json "${AGENT_DIRS}"
        -api-base "${API_BASE}"
        -mode "${MODE}"
        -concurrent-agents "${CONCURRENT_AGENTS}"
        -max-agents "${MAX_AGENTS}"
        -max-requests "${MAX_REQUESTS}"
        -max-inflight "${MAX_INFLIGHT}"
        -response-output-dir "${RESPONSE_OUTPUT_DIR}"
        -timeout "${TIMEOUT}"
        -stagger-seed "${STAGGER_SEED}"
        -output "${output_path}"
    )
    if [[ -n "${AGENT_START_STAGGER}" ]]; then
        args+=(-agent-start-stagger "${AGENT_START_STAGGER}")
    fi

    log "agent-dirs=${AGENT_DIRS}"
    log "mode=${MODE} concurrent_agents=${CONCURRENT_AGENTS} max_agents=${MAX_AGENTS} stagger=${AGENT_START_STAGGER:-none}"
    log "output=${output_path}"
    "${LOADGEN_BIN}" "${args[@]}"
}

main() {
    build_loadgen
    ensure_sglang_up
    run_loadgen
    if [[ "${STOP_SGLANG}" == "1" ]]; then
        log "stopping sglang (STOP_SGLANG=1)"
        bash "${SCRIPT_DIR}/sglang_launcher.sh" stop
    fi
}

main "$@"
