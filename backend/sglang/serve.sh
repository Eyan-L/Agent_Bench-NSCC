#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ACTION="${1:-start}"

SGLANG_PID_FILE="${SGLANG_PID_FILE:-${PROJECT_ROOT}/.sglang.pid}"
SGLANG_LOG_FILE="${SGLANG_LOG_FILE:-${PROJECT_ROOT}/sglang.log}"

# Runtime-configurable knobs. Top-level scripts should export these before
# invoking sglang_launcher.sh so port / GPU / model can vary between runs.
SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-60}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen}"
# Extra flags appended verbatim to the launch command (space-separated).
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
# "1" (default) = pass --enable-hierarchical-cache; anything else = omit it.
SGLANG_ENABLE_HIERARCHICAL_CACHE="${SGLANG_ENABLE_HIERARCHICAL_CACHE:-1}"

_gpu_count() {
    local devs="$1"
    if [[ -z "${devs}" ]]; then
        echo 0
        return
    fi
    # Count comma-separated, non-empty entries.
    local count=0
    local IFS=','
    # shellcheck disable=SC2206
    local parts=(${devs})
    for p in "${parts[@]}"; do
        [[ -n "${p}" ]] && count=$((count + 1))
    done
    echo "${count}"
}

start_sglang() {
    if [[ -f "${SGLANG_PID_FILE}" ]]; then
        existing_pid="$(cat "${SGLANG_PID_FILE}" || true)"
        if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
            echo "sglang already running, pid=${existing_pid}"
            return 0
        fi
        rm -f "${SGLANG_PID_FILE}"
    fi

    mkdir -p "$(dirname "${SGLANG_LOG_FILE}")"
    mkdir -p "$(dirname "${SGLANG_PID_FILE}")"

    local gpu_count
    gpu_count="$(_gpu_count "${SGLANG_CUDA_VISIBLE_DEVICES}")"
    if [[ "${gpu_count}" -lt 1 ]]; then
        echo "error: SGLANG_CUDA_VISIBLE_DEVICES is empty; set it to the GPU ids to use" >&2
        return 1
    fi

    local tp_args=()
    if [[ "${gpu_count}" -gt 1 ]]; then
        tp_args=(--tp "${gpu_count}")
    fi

    # shellcheck disable=SC2206
    local extra_args=(${SGLANG_EXTRA_ARGS})

    echo "sglang launch: port=${SGLANG_PORT} gpus=${SGLANG_CUDA_VISIBLE_DEVICES} (tp=${gpu_count}) model=${SGLANG_MODEL_PATH}"

    (
        # Save all values before unsetting, so they don't leak into sglang's
        # child processes (e.g. scheduler) and cause port conflicts.
        _cuda_devs="${SGLANG_CUDA_VISIBLE_DEVICES}"
        _model="${SGLANG_MODEL_PATH}"
        _host="${SGLANG_HOST}"
        _port="${SGLANG_PORT}"
        _mem="${SGLANG_MEM_FRACTION_STATIC}"
        _pgsz="${SGLANG_PAGE_SIZE}"
        _hicache="${SGLANG_HICACHE_SIZE}"
        _tool="${SGLANG_TOOL_CALL_PARSER}"
        _hier="${SGLANG_ENABLE_HIERARCHICAL_CACHE}"
        unset SGLANG_PORT SGLANG_HOST SGLANG_MODEL_PATH SGLANG_CUDA_VISIBLE_DEVICES \
              SGLANG_MEM_FRACTION_STATIC SGLANG_PAGE_SIZE SGLANG_HICACHE_SIZE \
              SGLANG_TOOL_CALL_PARSER SGLANG_EXTRA_ARGS SGLANG_ENABLE_HIERARCHICAL_CACHE

        hier_args=()
        if [[ "${_hier}" == "1" ]]; then
            hier_args=(--enable-hierarchical-cache --hicache-size "${_hicache}")
        fi

        SGLANG_KV_MEMTRACE=0 \
        SGLANG_KV_MEMTRACE_STRICT=0 \
        CUDA_VISIBLE_DEVICES="${_cuda_devs}" python3 -m sglang.launch_server \
            --model-path "${_model}" \
            --host "${_host}" \
            --port "${_port}" \
            "${tp_args[@]}" \
            --enable-metrics \
            --enable-cache-report \
            --mem-fraction-static "${_mem}" \
            "${hier_args[@]}" \
            --page-size "${_pgsz}" \
            --tool-call-parser "${_tool}" \
            "${extra_args[@]}"
    ) >>"${SGLANG_LOG_FILE}" 2>&1 &

    sglang_pid=$!
    echo "${sglang_pid}" > "${SGLANG_PID_FILE}"
    echo "sglang started, pid=${sglang_pid}"
    echo "sglang log=${SGLANG_LOG_FILE}"
}

stop_sglang() {
    if [[ ! -f "${SGLANG_PID_FILE}" ]]; then
        echo "sglang pid file not found, nothing to stop."
        return 0
    fi

    sglang_pid="$(cat "${SGLANG_PID_FILE}" || true)"
    if [[ -z "${sglang_pid}" ]]; then
        rm -f "${SGLANG_PID_FILE}"
        echo "sglang pid file empty, cleaned."
        return 0
    fi

    if ! kill -0 "${sglang_pid}" 2>/dev/null; then
        rm -f "${SGLANG_PID_FILE}"
        echo "sglang process ${sglang_pid} not running, cleaned pid file."
        return 0
    fi

    # Kill the entire process group so TP worker children are also stopped.
    kill -- -"${sglang_pid}" 2>/dev/null || kill "${sglang_pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
        if ! kill -0 "${sglang_pid}" 2>/dev/null; then
            rm -f "${SGLANG_PID_FILE}"
            echo "sglang stopped."
            return 0
        fi
        sleep 1
    done

    kill -9 -- -"${sglang_pid}" 2>/dev/null || kill -9 "${sglang_pid}" 2>/dev/null || true
    rm -f "${SGLANG_PID_FILE}"
    echo "sglang force-stopped."
}

status_sglang() {
    if [[ -f "${SGLANG_PID_FILE}" ]]; then
        sglang_pid="$(cat "${SGLANG_PID_FILE}" || true)"
        if [[ -n "${sglang_pid}" ]] && kill -0 "${sglang_pid}" 2>/dev/null; then
            echo "sglang running, pid=${sglang_pid}"
            return 0
        fi
    fi
    echo "sglang not running."
    return 1
}

case "${ACTION}" in
    start)
        start_sglang
        ;;
    stop)
        stop_sglang
        ;;
    restart)
        stop_sglang
        start_sglang
        ;;
    status)
        status_sglang
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 2
        ;;
esac
