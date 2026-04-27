#!/usr/bin/env bash
# Run an Alibaba qwen-bailian trace replay experiment end-to-end:
#   1. create experiment/<date_time>/,
#   2. build the unified loadgen at /loadgen if needed,
#   3. start sglang via run/sglang_launcher.sh (or sglang_launch_multiple.sh
#      with --router), wait for /health,
#   4. run `loadgen alibaba`, capturing its stdout to replayer.log and response
#      bodies to responses.json,
#   5. stop sglang on exit.
#
# Environment variables (override defaults):
#   TRACE            default: alibaba/qwen-bailian-usagetraces-anon/qwen_coder_tokens.jsonl
#   MODEL            default: default
#   SPEEDUP          default: 1.0
#   DURATION         wall-clock seconds to run the replayer (default 900 = 15min).
#                    Implemented via `timeout`; 0 = no cap.
#   START_OFFSET     default: 0
#   MAX_REQUESTS     default: 0 (no cap)
#   MAX_INFLIGHT     default: 1024
#   TYPE_FILTER      default: "" (all)
#   SESSION_SAMPLE_RATE  keep fraction of sessions (default 1.0 = keep all)
#   SESSION_SAMPLE_SEED  RNG seed for deterministic session sampling (default 42)
#   API_BASE         default: http://127.0.0.1:30000/v1  (single)
#                            http://127.0.0.1:30000/v1  (router)
#   SGLANG_STARTUP_TIMEOUT_SECS  default 900
#   GO_BIN           default: /mnt/vdb/hongrui/opt/go/bin/go
#   EXPERIMENT_TZ    default: Asia/Singapore
#
# Flags:
#   --router                            start 2 workers + router via sglang_launch_multiple.sh
#   --trace-sm-utilization, --trace-sm  trace dcgmi fields into sm.log
#   --help
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_TZ="${EXPERIMENT_TZ:-Asia/Singapore}"
TIMESTAMP="$(TZ="${EXPERIMENT_TZ}" date '+%Y-%m-%d_%H-%M-%S')"
EXPERIMENT_ROOT_DIR="${EXPERIMENT_ROOT_DIR:-${PROJECT_ROOT}/experiment}"
RUN_LOG_DIR="${EXPERIMENT_ROOT_DIR}/${TIMESTAMP}_alibaba"

SGLANG_LOG_FILE="${RUN_LOG_DIR}/sglang.log"
SGLANG_PID_FILE="${RUN_LOG_DIR}/sglang.pid"
RUN_LOG_FILE="${RUN_LOG_DIR}/run_alibaba_trace_experiment.log"
REPLAYER_LOG_FILE="${RUN_LOG_DIR}/replayer.log"
RESPONSES_FILE="${RUN_LOG_DIR}/responses.json"
CONFIG_SNAPSHOT="${RUN_LOG_DIR}/run_config.json"
SM_LOG_FILE="${RUN_LOG_DIR}/sm.log"

SM_TRACE_FIELD_IDS="${SM_TRACE_FIELD_IDS:-1002,1003,1005,1009,1010}"
SM_TRACE_ENABLED=0
SM_TRACE_PID=""

SGLANG_LAUNCHER="${PROJECT_ROOT}/run/sglang_launcher.sh"
SGLANG_LAUNCHER_MULTIPLE="${PROJECT_ROOT}/run/sglang_launch_multiple.sh"

LOADGEN_DIR="${PROJECT_ROOT}/loadgen"
LOADGEN_BIN="${LOADGEN_DIR}/loadgen-bin"

GO_BIN="${GO_BIN:-/mnt/vdb/hongrui/opt/go/bin/go}"
GOPATH_DIR="${GOPATH_DIR:-/mnt/vdb/hongrui/go}"
GOCACHE_DIR="${GOCACHE_DIR:-/mnt/vdb/hongrui/go-build}"

TRACE="${TRACE:-${PROJECT_ROOT}/alibaba/qwen-bailian-usagetraces-anon/qwen_traceA_tokens.jsonl}"
MODEL="${MODEL:-default}"
SPEEDUP="${SPEEDUP:-1.0}"
DURATION="${DURATION:-900}"
START_OFFSET="${START_OFFSET:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-1024}"
TYPE_FILTER="${TYPE_FILTER:-}"
SESSION_SAMPLE_RATE="${SESSION_SAMPLE_RATE:-1.0}"
SESSION_SAMPLE_SEED="${SESSION_SAMPLE_SEED:-42}"

# sglang knobs — forwarded to run/sglang_launcher.sh. Defaults mirror the
# launcher's own defaults but can be overridden from the caller.
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

# Derive API_BASE / health URL from the configured port unless caller pinned them.
API_BASE="${API_BASE:-http://127.0.0.1:${SGLANG_PORT}/v1}"
export SGLANG_HEALTH_URL="${SGLANG_HEALTH_URL:-http://127.0.0.1:${SGLANG_PORT}/health}"

ROUTER_MODE=0

show_help() {
    sed -n '2,40p' "$0"
}

parse_args() {
    while (($# > 0)); do
        case "$1" in
            --router)
                ROUTER_MODE=1
                shift
                ;;
            --trace-sm-utilization|--trace-sm)
                SM_TRACE_ENABLED=1
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "unknown arg: $1" >&2
                exit 2
                ;;
        esac
    done
}

start_sm_trace() {
    if [[ "${SM_TRACE_ENABLED}" -ne 1 ]]; then
        return 0
    fi
    if ! command -v dcgmi >/dev/null 2>&1; then
        log "dcgmi not found, cannot trace SM activity."
        return 1
    fi
    log "starting dcgmi trace for fields ${SM_TRACE_FIELD_IDS}: ${SM_LOG_FILE}"
    : > "${SM_LOG_FILE}"
    dcgmi dmon -e "${SM_TRACE_FIELD_IDS}" -d 1000 >"${SM_LOG_FILE}" 2>&1 &
    SM_TRACE_PID=$!
    sleep 1
    if ! kill -0 "${SM_TRACE_PID}" >/dev/null 2>&1; then
        log "failed to start SM activity trace."
        SM_TRACE_PID=""
        return 1
    fi
}

stop_sm_trace() {
    if [[ -z "${SM_TRACE_PID}" ]]; then
        return 0
    fi
    if kill -0 "${SM_TRACE_PID}" >/dev/null 2>&1; then
        log "stopping SM activity trace..."
        kill "${SM_TRACE_PID}" >/dev/null 2>&1 || true
        wait "${SM_TRACE_PID}" 2>/dev/null || true
    fi
    SM_TRACE_PID=""
}

parse_args "$@"

mkdir -p "${RUN_LOG_DIR}"
touch "${RUN_LOG_FILE}"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${RUN_LOG_FILE}"
}

_active_launcher() {
    if [[ "${ROUTER_MODE}" -eq 1 ]]; then
        echo "${SGLANG_LAUNCHER_MULTIPLE}"
    else
        echo "${SGLANG_LAUNCHER}"
    fi
}

cleanup() {
    stop_sm_trace
    log "stopping sglang..."
    SGLANG_LOG_FILE="${SGLANG_LOG_FILE}" \
    SGLANG_PID_FILE="${SGLANG_PID_FILE}" \
    "$(_active_launcher)" stop >>"${RUN_LOG_FILE}" 2>&1 || true
}
trap cleanup EXIT INT TERM

_wait_for_url() {
    local url="$1"
    local timeout_secs="$2"
    local start_ts now
    start_ts="$(date +%s)"
    while true; do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            log "ready: ${url}"
            return 0
        fi
        now="$(date +%s)"
        if (( now - start_ts > timeout_secs )); then
            log "timeout waiting for: ${url}"
            return 1
        fi
        sleep 2
    done
}

wait_for_sglang() {
    local timeout_secs="${SGLANG_STARTUP_TIMEOUT_SECS:-900}"
    if [[ "${ROUTER_MODE}" -eq 1 ]]; then
        local node0_url="${SGLANG_NODE0_HEALTH_URL:-http://127.0.0.1:30001/health}"
        local node1_url="${SGLANG_NODE1_HEALTH_URL:-http://127.0.0.1:30002/health}"
        log "waiting for worker nodes to be ready..."
        _wait_for_url "${node0_url}" "${timeout_secs}"
        _wait_for_url "${node1_url}" "${timeout_secs}"
    else
        local health_url="${SGLANG_HEALTH_URL:-http://127.0.0.1:30000/health}"
        _wait_for_url "${health_url}" "${timeout_secs}"
    fi
}

build_loadgen_if_needed() {
    if [[ -x "${LOADGEN_BIN}" ]]; then
        local newest_src
        newest_src="$(find "${LOADGEN_DIR}" -name '*.go' -type f -newer "${LOADGEN_BIN}" -print -quit)"
        if [[ -z "${newest_src}" ]]; then
            log "loadgen binary up-to-date: ${LOADGEN_BIN}"
            return 0
        fi
    fi
    if [[ ! -x "${GO_BIN}" ]]; then
        log "error: go binary not found at ${GO_BIN}"
        exit 1
    fi
    log "building loadgen with $(${GO_BIN} version)"
    (
        cd "${LOADGEN_DIR}"
        GOPATH="${GOPATH_DIR}" GOCACHE="${GOCACHE_DIR}" "${GO_BIN}" build -o "${LOADGEN_BIN}" ./cmd/loadgen
    ) >>"${RUN_LOG_FILE}" 2>&1
}

write_config_snapshot() {
    python3 - "$CONFIG_SNAPSHOT" <<PY
import json, os, sys
out = sys.argv[1]
cfg = {
    "timestamp": "${TIMESTAMP}",
    "experiment_tz": "${EXPERIMENT_TZ}",
    "router_mode": bool(int("${ROUTER_MODE}")),
    "api_base": "${API_BASE}",
    "health_url": "${SGLANG_HEALTH_URL}",
    "trace": "${TRACE}",
    "model": "${MODEL}",
    "speedup": float("${SPEEDUP}"),
    "duration": float("${DURATION}"),
    "start_offset": float("${START_OFFSET}"),
    "max_requests": int("${MAX_REQUESTS}"),
    "max_inflight": int("${MAX_INFLIGHT}"),
    "type_filter": "${TYPE_FILTER}",
    "session_sample_rate": float("${SESSION_SAMPLE_RATE}"),
    "session_sample_seed": int("${SESSION_SAMPLE_SEED}"),
    "sglang": {
        "host": "${SGLANG_HOST}",
        "port": int("${SGLANG_PORT}"),
        "model_path": "${SGLANG_MODEL_PATH}",
        "cuda_visible_devices": "${SGLANG_CUDA_VISIBLE_DEVICES}",
        "mem_fraction_static": float("${SGLANG_MEM_FRACTION_STATIC}"),
        "page_size": int("${SGLANG_PAGE_SIZE}"),
        "hicache_size": int("${SGLANG_HICACHE_SIZE}"),
        "tool_call_parser": "${SGLANG_TOOL_CALL_PARSER}",
        "extra_args": "${SGLANG_EXTRA_ARGS}",
    },
}
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
PY
}

if [[ ! -f "${TRACE}" ]]; then
    log "error: trace file not found: ${TRACE}"
    exit 1
fi

log "run log dir: ${RUN_LOG_DIR}"
if [[ "${ROUTER_MODE}" -eq 1 ]]; then
    log "mode: router (workers on 30001/30002, router on 30000)"
else
    log "mode: single (port 30000)"
fi

log "trace: ${TRACE}"
log "model=${MODEL} speedup=${SPEEDUP} duration=${DURATION}s start_offset=${START_OFFSET}s max_inflight=${MAX_INFLIGHT} type_filter=${TYPE_FILTER:-ALL} session_sample_rate=${SESSION_SAMPLE_RATE} session_sample_seed=${SESSION_SAMPLE_SEED}"
log "sglang: port=${SGLANG_PORT} gpus=${SGLANG_CUDA_VISIBLE_DEVICES} model=${SGLANG_MODEL_PATH}"
log "api_base=${API_BASE} health=${SGLANG_HEALTH_URL}"

write_config_snapshot
build_loadgen_if_needed

log "starting sglang..."
log "DEBUG: launcher=$(_active_launcher) SGLANG_LOG_FILE=${SGLANG_LOG_FILE} SGLANG_PID_FILE=${SGLANG_PID_FILE}"
log "DEBUG: SGLANG_PORT=${SGLANG_PORT} SGLANG_HOST=${SGLANG_HOST} SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES} SGLANG_MODEL_PATH=${SGLANG_MODEL_PATH}"
log "DEBUG: SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC} SGLANG_EXTRA_ARGS=${SGLANG_EXTRA_ARGS}"
log "DEBUG: checking port ${SGLANG_PORT} before launch:"
ss -tlnp | grep ":${SGLANG_PORT} " >> "${RUN_LOG_FILE}" 2>&1 || log "DEBUG: port ${SGLANG_PORT} is free"
log "DEBUG: exec: SGLANG_LOG_FILE=${SGLANG_LOG_FILE} SGLANG_PID_FILE=${SGLANG_PID_FILE} $(_active_launcher) start"
SGLANG_LOG_FILE="${SGLANG_LOG_FILE}" \
SGLANG_PID_FILE="${SGLANG_PID_FILE}" \
"$(_active_launcher)" start >>"${RUN_LOG_FILE}" 2>&1

wait_for_sglang

loadgen_args=(
    alibaba
    -trace "${TRACE}"
    -api-base "${API_BASE}"
    -model "${MODEL}"
    -speedup "${SPEEDUP}"
    -start-offset "${START_OFFSET}"
    -max-requests "${MAX_REQUESTS}"
    -max-inflight "${MAX_INFLIGHT}"
    -session-sample-rate "${SESSION_SAMPLE_RATE}"
    -session-sample-seed "${SESSION_SAMPLE_SEED}"
    -output "${RESPONSES_FILE}"
    -record-body=true
)
if [[ -n "${TYPE_FILTER}" ]]; then
    loadgen_args+=(-type-filter "${TYPE_FILTER}")
fi

start_sm_trace
log "running loadgen alibaba..."
log "responses -> ${RESPONSES_FILE}"
log "replayer log -> ${REPLAYER_LOG_FILE}"
if [[ "${DURATION}" != "0" && -n "${DURATION}" ]]; then
    log "wall-clock cap: ${DURATION}s via timeout"
fi
replayer_exit=0
set +e
if [[ "${DURATION}" != "0" && -n "${DURATION}" ]]; then
    timeout --kill-after=30s "${DURATION}s" "${LOADGEN_BIN}" "${loadgen_args[@]}" 2>&1 | tee "${REPLAYER_LOG_FILE}"
else
    "${LOADGEN_BIN}" "${loadgen_args[@]}" 2>&1 | tee "${REPLAYER_LOG_FILE}"
fi
replayer_exit=${PIPESTATUS[0]}
set -e
stop_sm_trace

if [[ "${replayer_exit}" -eq 124 ]]; then
    log "loadgen hit wall-clock limit (${DURATION}s), terminated by timeout — treating as success"
    replayer_exit=0
elif [[ "${replayer_exit}" -ne 0 ]]; then
    log "loadgen exited with non-zero code: ${replayer_exit}"
fi

log "done. artifacts under ${RUN_LOG_DIR}"
exit "${replayer_exit}"
