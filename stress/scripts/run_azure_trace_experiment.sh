#!/usr/bin/env bash
# Run an Azure LLM inference trace replay experiment end-to-end via the
# unified loadgen (scripts/go/loadgen, azure subcommand):
#   1. create experiment/<date_time>_azure/,
#   2. build loadgen (Go) if needed,
#   3. start sglang via run/sglang_launcher.sh, wait for /health,
#   4. run `loadgen azure` against the preprocessed dataset JSON, capturing
#      output to loadgen.log and result json,
#   5. stop sglang on exit.
#
# The dataset JSON must be generated first with
# LLMLoadgen/offline/offline_liquid_tool_v2.py. See that script for details.
#
# Environment variables (override defaults):
#   DATASET_PATH     default: azure-trace/sample_0_code_all_input1_output1_ALL.json
#   DURATION         wall-clock seconds to run the loadgen (default 0 = whole trace).
#                    Implemented via `timeout`; 0 = no cap.
#   SAMPLING_RATE    float in [0,1]; 1.0 = keep all. Upsampling is done via SPEEDUP now.
#   RECUR_STEP       default: 10
#   SPEEDUP          default: 1.0  (>1 = compress inter-request delays; replaces old UPSCALE=ars/trace)
#   MAX_DRIFT        default: 100  (ms)
#   PROMPT_NAME      default: None  (None = random downsample; else Wasserstein on data[PROMPT_NAME])
#   SEED             default: 42
#   MAX_REQUESTS     default: 0 (unbounded)
#   MAX_INFLIGHT     default: 1024
#   API_URL          default: http://127.0.0.1:${SGLANG_PORT}/v1
#   SGLANG_STARTUP_TIMEOUT_SECS  default 900
#   GO_BIN           default: /mnt/vdb/hongrui/opt/go/bin/go
#   EXPERIMENT_TZ    default: Asia/Singapore
#
# Flags:
#   --trace-sm-utilization, --trace-sm  trace dcgmi fields into sm.log
#   --help
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_TZ="${EXPERIMENT_TZ:-Asia/Singapore}"
TIMESTAMP="$(TZ="${EXPERIMENT_TZ}" date '+%Y-%m-%d_%H-%M-%S')"
EXPERIMENT_ROOT_DIR="${EXPERIMENT_ROOT_DIR:-${PROJECT_ROOT}/experiment}"
RUN_LOG_DIR="${EXPERIMENT_ROOT_DIR}/${TIMESTAMP}_azure"

SGLANG_LOG_FILE="${RUN_LOG_DIR}/sglang.log"
SGLANG_PID_FILE="${RUN_LOG_DIR}/sglang.pid"
RUN_LOG_FILE="${RUN_LOG_DIR}/run_azure_trace_experiment.log"
LOADGEN_LOG_FILE="${RUN_LOG_DIR}/loadgen.log"
RESULT_FILE="${RUN_LOG_DIR}/result.json"
SAMPLED_FILE="${RUN_LOG_DIR}/sampled_trace.json"
CONFIG_SNAPSHOT="${RUN_LOG_DIR}/run_config.json"
SM_LOG_FILE="${RUN_LOG_DIR}/sm.log"

SM_TRACE_FIELD_IDS="${SM_TRACE_FIELD_IDS:-1002,1003,1005,1009,1010}"
SM_TRACE_ENABLED=0
SM_TRACE_PID=""

SGLANG_LAUNCHER="${PROJECT_ROOT}/run/sglang_launcher.sh"

LOADGEN_DIR="${PROJECT_ROOT}/loadgen"
LOADGEN_BIN="${LOADGEN_DIR}/loadgen-bin"

GO_BIN="${GO_BIN:-/mnt/vdb/hongrui/opt/go/bin/go}"
GOPATH_DIR="${GOPATH_DIR:-/mnt/vdb/hongrui/go}"
GOCACHE_DIR="${GOCACHE_DIR:-/mnt/vdb/hongrui/go-build}"

DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/azure-trace/sample_0_code_all_input1_output1_ALL.json}"
DURATION="${DURATION:-0}"
SAMPLING_RATE="${SAMPLING_RATE:-1.0}"
RECUR_STEP="${RECUR_STEP:-10}"
SPEEDUP="${SPEEDUP:-1.0}"
MAX_DRIFT="${MAX_DRIFT:-100}"
PROMPT_NAME="${PROMPT_NAME:-None}"
SEED="${SEED:-42}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-1024}"
SAVE_SAMPLED="${SAVE_SAMPLED:-0}"

# sglang knobs — forwarded to run/sglang_launcher.sh.
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

API_URL="${API_URL:-http://127.0.0.1:${SGLANG_PORT}/v1}"
export SGLANG_HEALTH_URL="${SGLANG_HEALTH_URL:-http://127.0.0.1:${SGLANG_PORT}/health}"

show_help() {
    sed -n '2,32p' "$0"
}

parse_args() {
    while (($# > 0)); do
        case "$1" in
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

cleanup() {
    stop_sm_trace
    log "stopping sglang..."
    SGLANG_LOG_FILE="${SGLANG_LOG_FILE}" \
    SGLANG_PID_FILE="${SGLANG_PID_FILE}" \
    "${SGLANG_LAUNCHER}" stop >>"${RUN_LOG_FILE}" 2>&1 || true
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
    _wait_for_url "${SGLANG_HEALTH_URL}" "${timeout_secs}"
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
import json, sys
out = sys.argv[1]
cfg = {
    "timestamp": "${TIMESTAMP}",
    "experiment_tz": "${EXPERIMENT_TZ}",
    "api_url": "${API_URL}",
    "health_url": "${SGLANG_HEALTH_URL}",
    "dataset_path": "${DATASET_PATH}",
    "duration": float("${DURATION}"),
    "sampling_rate": float("${SAMPLING_RATE}"),
    "recur_step": int("${RECUR_STEP}"),
    "speedup": float("${SPEEDUP}"),
    "max_drift_ms": int("${MAX_DRIFT}"),
    "prompt_name": "${PROMPT_NAME}",
    "seed": int("${SEED}"),
    "max_requests": int("${MAX_REQUESTS}"),
    "max_inflight": int("${MAX_INFLIGHT}"),
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

if [[ ! -f "${DATASET_PATH}" ]]; then
    log "error: dataset file not found: ${DATASET_PATH}"
    log "hint: generate it first with LLMLoadgen/offline/offline_liquid_tool_v2.py"
    exit 1
fi

EFFECTIVE_DATASET_PATH="${DATASET_PATH}"

log "run log dir: ${RUN_LOG_DIR}"
log "dataset: ${EFFECTIVE_DATASET_PATH} (wall-clock cap=${DURATION}s)"
log "sampling_rate=${SAMPLING_RATE} recur_step=${RECUR_STEP} speedup=${SPEEDUP} max_drift=${MAX_DRIFT}ms prompt_name=${PROMPT_NAME}"
log "sglang: port=${SGLANG_PORT} gpus=${SGLANG_CUDA_VISIBLE_DEVICES} model=${SGLANG_MODEL_PATH}"
log "api_url=${API_URL} health=${SGLANG_HEALTH_URL}"

write_config_snapshot
build_loadgen_if_needed

log "starting sglang..."
SGLANG_LOG_FILE="${SGLANG_LOG_FILE}" \
SGLANG_PID_FILE="${SGLANG_PID_FILE}" \
"${SGLANG_LAUNCHER}" start >>"${RUN_LOG_FILE}" 2>&1

wait_for_sglang

loadgen_args=(
    azure
    -trace "${EFFECTIVE_DATASET_PATH}"
    -sampling-rate "${SAMPLING_RATE}"
    -recur-step "${RECUR_STEP}"
    -speedup "${SPEEDUP}"
    -api-base "${API_URL}"
    -seed "${SEED}"
    -prompt-name "${PROMPT_NAME}"
    -max-drift-ms "${MAX_DRIFT}"
    -max-requests "${MAX_REQUESTS}"
    -max-inflight "${MAX_INFLIGHT}"
    -output "${RESULT_FILE}"
)
# SAVE_SAMPLED=1 writes the sampled trace and exits without sending requests.
if [[ "${SAVE_SAMPLED}" == "1" ]]; then
    loadgen_args+=(-save-sampled "${SAMPLED_FILE}")
    log "SAVE_SAMPLED=1: will only dump sampled trace, will NOT send requests"
fi

start_sm_trace
log "running loadgen azure..."
log "result -> ${RESULT_FILE}"
log "loadgen log -> ${LOADGEN_LOG_FILE}"
if [[ "${DURATION}" != "0" && -n "${DURATION}" ]]; then
    log "wall-clock cap: ${DURATION}s via timeout"
fi
loadgen_exit=0
set +e
if [[ "${DURATION}" != "0" && -n "${DURATION}" ]]; then
    (
        cd "${LOADGEN_DIR}"
        timeout --kill-after=30s "${DURATION}s" "${LOADGEN_BIN}" "${loadgen_args[@]}"
    ) 2>&1 | tee "${LOADGEN_LOG_FILE}"
else
    (
        cd "${LOADGEN_DIR}"
        "${LOADGEN_BIN}" "${loadgen_args[@]}"
    ) 2>&1 | tee "${LOADGEN_LOG_FILE}"
fi
loadgen_exit=${PIPESTATUS[0]}
set -e
stop_sm_trace

if [[ "${loadgen_exit}" -eq 124 ]]; then
    log "loadgen hit wall-clock limit (${DURATION}s), terminated by timeout — treating as success"
    loadgen_exit=0
elif [[ "${loadgen_exit}" -ne 0 ]]; then
    log "loadgen exited with non-zero code: ${loadgen_exit}"
fi

log "done. artifacts under ${RUN_LOG_DIR}"
exit "${loadgen_exit}"
