#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_TZ="${EXPERIMENT_TZ:-Asia/Singapore}"
TIMESTAMP="$(TZ="${EXPERIMENT_TZ}" date '+%Y-%m-%d_%H-%M-%S')"
EXPERIMENT_ROOT_DIR="${PROJECT_ROOT}/experiment"
RUN_LOG_DIR="${EXPERIMENT_ROOT_DIR}/${TIMESTAMP}"

SGLANG_LOG_FILE="${RUN_LOG_DIR}/sglang.log"
SGLANG_PID_FILE="${RUN_LOG_DIR}/sglang.pid"
RUN_LOG_FILE="${RUN_LOG_DIR}/run_agents.log"
TRACE_LOG_FILE="${RUN_LOG_DIR}/trace_load_generator.log"
SM_LOG_FILE="${RUN_LOG_DIR}/sm.log"
AGENT_DIR_SNAPSHOT_DIR="${RUN_LOG_DIR}/agent_dirs"
RESPONSES_OUTPUT_DIR="${RUN_LOG_DIR}/responses"

TRACE_SCRIPT="${TRACE_SCRIPT:-${PROJECT_ROOT}/scripts/python/trace_load_generator.py}"
SGLANG_LAUNCHER="${PROJECT_ROOT}/run/sglang_launcher.sh"
SGLANG_LAUNCHER_MULTIPLE="${PROJECT_ROOT}/run/sglang_launch_multiple.sh"
SM_TRACE_FIELD_IDS="1002,1003,1005,1009,1010"
SM_TRACE_ENABLED=0
SM_TRACE_PID=""
ROUTER_MODE=0

# sglang knobs forwarded to run/sglang_launcher.sh. Override from the env.
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

# Derived endpoints (single mode). Router mode uses its own health URLs below.
API_BASE="${API_BASE:-http://127.0.0.1:${SGLANG_PORT}/v1}"
export SGLANG_HEALTH_URL="${SGLANG_HEALTH_URL:-http://127.0.0.1:${SGLANG_PORT}/health}"

should_show_help() {
    local arg
    for arg in "$@"; do
        case "${arg}" in
            --help|-h)
                return 0
                ;;
            --)
                break
                ;;
        esac
    done

    return 1
}

if should_show_help "$@"; then
    cat <<EOF
Usage: $(basename "$0") [options] [trace_load_generator args]

Script options:
  --router                     Use two sglang instances + router (port 30000)
                               instead of a single instance (port 30001)
  --trace-sm-utilization, --trace-sm  Trace dcgmi fields ${SM_TRACE_FIELD_IDS} into experiment/<date_time>/sm.log

This script will:
1. create experiment/<date_time>/,
2. start sglang (single: sglang_launcher.sh / router: sglang_launch_multiple.sh),
3. run scripts/python/trace_load_generator.py,
4. copy --agent-dirs-json into experiment/<date_time>/agent_dirs/agent_dirs.json,
5. set response output dir to experiment/<date_time>/responses/,
6. stop sglang on exit.

Environment variables:
  SGLANG_HEALTH_URL            default (single): http://127.0.0.1:30001/health
                               default (router): http://127.0.0.1:30000/health
  SGLANG_STARTUP_TIMEOUT_SECS  default: 900
EOF
    exit 0
fi

mkdir -p "${RUN_LOG_DIR}"
mkdir -p "${AGENT_DIR_SNAPSHOT_DIR}"
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

parse_args() {
    TRACE_ARGS=()
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
            --)
                TRACE_ARGS+=("$@")
                break
                ;;
            *)
                TRACE_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

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
        # Wait for both worker nodes first — router /health is up instantly
        # but does not reflect whether workers have finished loading the model.
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

get_agent_dirs_json_arg() {
    local agent_dirs_json=""
    local i arg
    for ((i=1; i<=$#; i++)); do
        arg="${!i}"
        if [[ "${arg}" == "--agent-dirs-json" ]]; then
            local next_idx=$((i + 1))
            if (( next_idx <= $# )); then
                agent_dirs_json="${!next_idx}"
            fi
        elif [[ "${arg}" == --agent-dirs-json=* ]]; then
            agent_dirs_json="${arg#*=}"
        fi
    done

    echo "${agent_dirs_json}"
}

copy_agent_dirs_json() {
    local agent_dirs_json="${1:-}"
    if [[ -z "${agent_dirs_json}" ]]; then
        log "no --agent-dirs-json provided, skip json snapshot."
        return 0
    fi

    local src_json
    src_json="$(python3 -c 'import os,sys,pathlib; p=pathlib.Path(sys.argv[1]).expanduser().resolve(); print(p)' "${agent_dirs_json}")"
    if [[ ! -f "${src_json}" ]]; then
        log "agent dirs json not found: ${src_json}"
        return 1
    fi

    cp -a "${src_json}" "${AGENT_DIR_SNAPSHOT_DIR}/agent_dirs.json"
    log "copied agent dirs json: ${src_json} -> ${AGENT_DIR_SNAPSHOT_DIR}/agent_dirs.json"
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
log "run log dir: ${RUN_LOG_DIR}"
if [[ "${ROUTER_MODE}" -eq 1 ]]; then
    log "mode: router (2 nodes, port 30001+30002, router port 30000)"
else
    log "mode: single (port ${SGLANG_PORT})"
fi
log "sglang: port=${SGLANG_PORT} gpus=${SGLANG_CUDA_VISIBLE_DEVICES} model=${SGLANG_MODEL_PATH}"
log "api_base=${API_BASE} health=${SGLANG_HEALTH_URL}"
log "starting sglang..."
SGLANG_LOG_FILE="${SGLANG_LOG_FILE}" \
SGLANG_PID_FILE="${SGLANG_PID_FILE}" \
"$(_active_launcher)" start >>"${RUN_LOG_FILE}" 2>&1

wait_for_sglang

AGENT_DIRS_JSON_ARG="$(get_agent_dirs_json_arg "${TRACE_ARGS[@]}")"
TRACE_ARGS+=("--response-output-dir" "${RESPONSES_OUTPUT_DIR}")

# Auto-inject --api-base if the caller didn't already pass one so the
# load generator talks to whichever port sglang was started on.
_has_api_base=0
for _a in "${TRACE_ARGS[@]}"; do
    if [[ "${_a}" == "--api-base" || "${_a}" == --api-base=* ]]; then
        _has_api_base=1
        break
    fi
done
if [[ "${_has_api_base}" -eq 0 ]]; then
    TRACE_ARGS+=("--api-base" "${API_BASE}")
fi

start_sm_trace
log "running trace_load_generator..."
trace_exit=0
set +e
python3 "${TRACE_SCRIPT}" "${TRACE_ARGS[@]}" 2>&1 | tee "${TRACE_LOG_FILE}"
trace_exit=${PIPESTATUS[0]}
set -e
stop_sm_trace
if [[ "${trace_exit}" -ne 0 ]]; then
    log "trace_load_generator exited with non-zero code: ${trace_exit}"
fi

log "copying agent dirs json..."
copy_agent_dirs_json "${AGENT_DIRS_JSON_ARG}" || true
log "done. artifacts under ${RUN_LOG_DIR}"

exit "${trace_exit}"
