#!/usr/bin/env bash
# Python owns SGLang config resolution and command planning.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
export AB_REPO_ROOT

SGLANG_PID_FILE="${SGLANG_PID_FILE:-${PROJECT_ROOT}/.sglang.pid}"
export SGLANG_PID_FILE

ACTION="${1:-start}"
if [[ "${ACTION}" == -* ]]; then
    ACTION="start"
elif [[ $# -gt 0 ]]; then
    shift
fi

stop_sglang() {
    if [[ ! -f "${SGLANG_PID_FILE}" ]]; then
        echo "sglang pid file not found, nothing to stop."
        return 0
    fi

    local sglang_pid
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
        local sglang_pid
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
        exec python3 -m agent_bench.serving.launch --backend sglang "$@"
        ;;
    stop)
        stop_sglang
        ;;
    restart)
        stop_sglang
        exec python3 -m agent_bench.serving.launch --backend sglang "$@"
        ;;
    status)
        status_sglang
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [launch args]"
        exit 2
        ;;
esac
