#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ACTION="${1:-start}"

# Accept the same env vars as sglang_launcher.sh; derive per-process paths
SGLANG_LOG_FILE="${SGLANG_LOG_FILE:-${PROJECT_ROOT}/sglang.log}"
SGLANG_PID_FILE="${SGLANG_PID_FILE:-${PROJECT_ROOT}/.sglang.pid}"

# sglang knobs — same interface as sglang_launcher.sh so Go config values
# flow through the same env vars regardless of which launcher is picked.
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-30}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
SGLANG_ENABLE_HIERARCHICAL_CACHE="${SGLANG_ENABLE_HIERARCHICAL_CACHE:-0}"

_log_base="${SGLANG_LOG_FILE%.log}"
_pid_base="${SGLANG_PID_FILE%.pid}"

NODE0_LOG="${_log_base}_node0.log"
NODE1_LOG="${_log_base}_node1.log"
ROUTER_LOG="${_log_base}_router.log"
NODE0_PID="${_pid_base}_node0.pid"
NODE1_PID="${_pid_base}_node1.pid"
ROUTER_PID="${_pid_base}_router.pid"

NODE0_PORT=30001
NODE1_PORT=30002
ROUTER_PORT=30000

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_start_one_node() {
    local gpu="$1"
    local port="$2"
    local pid_file="$3"
    local log_file="$4"

    if [[ -f "${pid_file}" ]]; then
        local existing_pid
        existing_pid="$(cat "${pid_file}" || true)"
        if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
            echo "sglang node (port=${port}) already running, pid=${existing_pid}"
            return 0
        fi
        rm -f "${pid_file}"
    fi

    # shellcheck disable=SC2206
    local extra_args=(${SGLANG_EXTRA_ARGS})

    local hier_args=()
    if [[ "${SGLANG_ENABLE_HIERARCHICAL_CACHE}" == "1" ]]; then
        hier_args=(--enable-hierarchical-cache --hicache-size "${SGLANG_HICACHE_SIZE}")
    fi

    (
        SGLANG_KV_MEMTRACE=0 \
        SGLANG_KV_MEMTRACE_STRICT=0 \
        CUDA_VISIBLE_DEVICES="${gpu}" python3 -m sglang.launch_server \
            --model-path "${SGLANG_MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "${port}" \
            --enable-metrics \
            --enable-cache-report \
            --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
            --page-size "${SGLANG_PAGE_SIZE}" \
            "${hier_args[@]}" \
            --tool-call-parser "${SGLANG_TOOL_CALL_PARSER}" \
            "${extra_args[@]}"
    ) >>"${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" > "${pid_file}"
    echo "sglang node started (gpu=${gpu}, port=${port}), pid=${pid}, log=${log_file}"
}

_stop_one() {
    local pid_file="$1"
    local label="$2"

    if [[ ! -f "${pid_file}" ]]; then
        echo "${label}: pid file not found, nothing to stop."
        return 0
    fi

    local pid
    pid="$(cat "${pid_file}" || true)"
    if [[ -z "${pid}" ]]; then
        rm -f "${pid_file}"
        return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f "${pid_file}"
        echo "${label}: process ${pid} not running, cleaned pid file."
        return 0
    fi

    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${pid_file}"
            echo "${label}: stopped."
            return 0
        fi
        sleep 1
    done

    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${pid_file}"
    echo "${label}: force-stopped."
}

_status_one() {
    local pid_file="$1"
    local label="$2"

    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}" || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "${label}: running, pid=${pid}"
            return 0
        fi
    fi
    echo "${label}: not running."
    return 1
}

# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

start_all() {
    _start_one_node 0 "${NODE0_PORT}" "${NODE0_PID}" "${NODE0_LOG}"
    _start_one_node 1 "${NODE1_PORT}" "${NODE1_PID}" "${NODE1_LOG}"

    if [[ -f "${ROUTER_PID}" ]]; then
        local existing_pid
        existing_pid="$(cat "${ROUTER_PID}" || true)"
        if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
            echo "sglang router already running, pid=${existing_pid}"
            return 0
        fi
        rm -f "${ROUTER_PID}"
    fi

    (
        python3 -m sglang_router.launch_router \
            --host 0.0.0.0 \
            --port "${ROUTER_PORT}" \
            --worker-urls \
                "http://127.0.0.1:${NODE0_PORT}" \
                "http://127.0.0.1:${NODE1_PORT}" \
            --policy cache_aware
    ) >>"${ROUTER_LOG}" 2>&1 &

    local router_pid=$!
    echo "${router_pid}" > "${ROUTER_PID}"
    echo "sglang router started (port=${ROUTER_PORT}), pid=${router_pid}, log=${ROUTER_LOG}"
}

stop_all() {
    _stop_one "${ROUTER_PID}" "sglang router"
    _stop_one "${NODE1_PID}" "sglang node1"
    _stop_one "${NODE0_PID}" "sglang node0"
}

status_all() {
    local rc=0
    _status_one "${NODE0_PID}" "sglang node0 (port=${NODE0_PORT})" || rc=1
    _status_one "${NODE1_PID}" "sglang node1 (port=${NODE1_PORT})" || rc=1
    _status_one "${ROUTER_PID}" "sglang router  (port=${ROUTER_PORT})" || rc=1
    return "${rc}"
}

case "${ACTION}" in
    start)   start_all ;;
    stop)    stop_all ;;
    restart) stop_all; start_all ;;
    status)  status_all ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 2
        ;;
esac
