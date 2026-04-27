#!/usr/bin/env bash
# Replay an Alibaba qwen-bailian token trace against a running sglang server
# using the unified loadgen at scripts/go/loadgen (alibaba subcommand).
#
# Defaults to a 15-minute wall-clock slice of the coder trace at real speed.
# Override via env vars, e.g.:
#
#   TRACE=.../qwen_traceB_tokens.jsonl DURATION=300 SPEEDUP=2 \
#     run/replay_alibaba_trace.sh
#
# The script:
#   1. makes sure Go 1.24 and the loadgen binary are available (builds if
#      missing);
#   2. starts sglang via run/sglang_launcher.sh (no-op if already up) and
#      waits for /v1/models to respond;
#   3. runs `loadgen alibaba` with the chosen window;
#   4. leaves sglang running unless STOP_SGLANG=1 is set.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GO_BIN="${GO_BIN:-/mnt/vdb/hongrui/opt/go/bin/go}"
GOPATH_DIR="${GOPATH_DIR:-/mnt/vdb/hongrui/go}"
GOCACHE_DIR="${GOCACHE_DIR:-/mnt/vdb/hongrui/go-build}"

LOADGEN_DIR="${PROJECT_ROOT}/loadgen"
LOADGEN_BIN="${LOADGEN_DIR}/loadgen-bin"

TRACE="${TRACE:-${PROJECT_ROOT}/alibaba/qwen-bailian-usagetraces-anon/qwen_traceA_tokens.jsonl}"

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
MODEL="${MODEL:-default}"
SPEEDUP="${SPEEDUP:-1.0}"
DURATION="${DURATION:-900}"            # trace seconds; 900 = 15min
START_OFFSET="${START_OFFSET:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-1024}"
TYPE_FILTER="${TYPE_FILTER:-}"
SESSION_SAMPLE_RATE="${SESSION_SAMPLE_RATE:-1.0}"
SESSION_SAMPLE_SEED="${SESSION_SAMPLE_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/vdb/hongrui/replay_results}"
WAIT_SGLANG_SECS="${WAIT_SGLANG_SECS:-1200}"
STOP_SGLANG="${STOP_SGLANG:-0}"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

build_loadgen() {
    # Rebuild if binary is missing or older than any .go file under the module.
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
    if [[ ! -f "${TRACE}" ]]; then
        echo "error: trace file not found: ${TRACE}" >&2
        exit 1
    fi

    mkdir -p "${OUTPUT_DIR}"
    local trace_base
    trace_base="$(basename "${TRACE}" .jsonl)"
    local stamp
    stamp="$(date '+%Y%m%d_%H%M%S')"
    local output_path="${OUTPUT_DIR}/${trace_base}_dur${DURATION}_sp${SPEEDUP}_${stamp}.json"

    local args=(
        alibaba
        -trace "${TRACE}"
        -api-base "${API_BASE}"
        -model "${MODEL}"
        -speedup "${SPEEDUP}"
        -start-offset "${START_OFFSET}"
        -duration "${DURATION}"
        -max-requests "${MAX_REQUESTS}"
        -max-inflight "${MAX_INFLIGHT}"
        -session-sample-rate "${SESSION_SAMPLE_RATE}"
        -session-sample-seed "${SESSION_SAMPLE_SEED}"
        -output "${output_path}"
    )
    if [[ -n "${TYPE_FILTER}" ]]; then
        args+=(-type-filter "${TYPE_FILTER}")
    fi

    log "replay trace=${TRACE}"
    log "speedup=${SPEEDUP} duration=${DURATION}s start_offset=${START_OFFSET}s max_inflight=${MAX_INFLIGHT}"
    log "session_sample_rate=${SESSION_SAMPLE_RATE} session_sample_seed=${SESSION_SAMPLE_SEED}"
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
