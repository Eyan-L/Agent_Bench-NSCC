#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"

ACTION="${1:-start}"

# 1) MODEL config — model identity and model-level limits.
MODEL_CFG="${SGLANG_MODEL_CONFIG:-${AB_MODEL_CONFIG:-}}"
if [[ -z "$MODEL_CFG" && -n "${MODEL_PROFILE:-}" ]]; then
    MODEL_CFG="$KIT_ROOT/profiles/models/${MODEL_PROFILE}.json"
fi
if [[ -n "$MODEL_CFG" && -f "$MODEL_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_model_env.py" "$MODEL_CFG")"
fi

# 2) SERVING config — engine endpoint/scheduler knobs.
SERVING_CFG="${SGLANG_SERVING_CONFIG:-${AB_SERVING_CONFIG:-}}"
if [[ -z "$SERVING_CFG" && -n "${SERVING:-}" ]]; then
    SERVING_CFG="$KIT_ROOT/profiles/serving/${SERVING}.json"
fi
if [[ -n "$SERVING_CFG" && -f "$SERVING_CFG" ]]; then
    eval "$(python3 "$KIT_ROOT/scripts/load_serving_env.py" "$SERVING_CFG")"
fi

SGLANG_PID_FILE="${SGLANG_PID_FILE:-${PROJECT_ROOT}/.sglang.pid}"
# Empty default — when unset, engine output is inherited from the parent's
# stdout/stderr (Makefile's _ensure_serve redirects those to the per-sweep
# SERVE_LOG path). Set explicitly only if you want a fixed standalone path.
SGLANG_LOG_FILE="${SGLANG_LOG_FILE:-}"

# Runtime-configurable knobs. Top-level scripts should export these before
# invoking sglang_launcher.sh so port / GPU / model can vary between runs.
# Model/serving config values (MODEL, PORT, GPU_MEM_UTIL, MODEL_CONTEXT_WINDOW)
# are mapped onto the SGLANG_* knobs below when the matching
# SGLANG_* env var is not already set.
SGLANG_HOST="${SGLANG_HOST:-${BIND_HOST:-0.0.0.0}}"
SGLANG_PORT="${SGLANG_PORT:-${PORT:-30000}}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-${GPU_MEM_UTIL:-0.85}}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-${MAX_MODEL_LEN:-${MODEL_CONTEXT_WINDOW:-}}}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-60}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-${TOOL_CALL_PARSER:-qwen}}"
# Optional. When set (e.g. "qwen3", "deepseekv3"), pass --reasoning-parser so
# sglang extracts <think>...</think> spans into `reasoning_content` instead of
# leaving them embedded in `content`. Empty = omit the flag.
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-${REASONING_PARSER:-}}"
# Extra flags appended verbatim to the launch command (space-separated).
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
# "1" (default) = pass --enable-hierarchical-cache; anything else = omit it.
SGLANG_ENABLE_HIERARCHICAL_CACHE="${SGLANG_ENABLE_HIERARCHICAL_CACHE:-1}"
# "1" (default) = block until sglang dies (matches vllm/serve.sh, required by
# the Makefile's _ensure_serve poll loop). "0" = fire-and-forget daemon mode
# (writes PID file then exits, useful for `bash serve.sh start` from a TTY).
SGLANG_FOREGROUND="${SGLANG_FOREGROUND:-1}"

# Auto-activate a conda env if sglang isn't importable on the default python
# and SGLANG_CONDA_ENV is set. We can't rely on `conda activate` here — under
# non-interactive shells it inserts the env bin in the *middle* of PATH (after
# /usr/bin), so `python3` still resolves to the system one. Prepend the env's
# bin dir directly to PATH for a deterministic resolve.
if ! python3 -c "import sglang" >/dev/null 2>&1 && [[ -n "${SGLANG_CONDA_ENV:-}" ]]; then
    for env_bin in \
        "$HOME/anaconda3/envs/${SGLANG_CONDA_ENV}/bin" \
        "$HOME/miniconda3/envs/${SGLANG_CONDA_ENV}/bin" \
        "/opt/conda/envs/${SGLANG_CONDA_ENV}/bin"; do
        if [[ -x "${env_bin}/python3" ]]; then
            export PATH="${env_bin}:${PATH}"
            export CONDA_PREFIX="$(dirname "${env_bin}")"
            export CONDA_DEFAULT_ENV="${SGLANG_CONDA_ENV}"
            break
        fi
    done
fi

# Wire the env's bundled NVIDIA CUDA libs into LD_LIBRARY_PATH. sgl_kernel's
# SM90 binary is linked against CUDA 13 (libnvrtc.so.13), shipped via the
# nvidia-* pip wheels under site-packages/nvidia/*/lib. Without this the
# launcher dies with "libnvrtc.so.13: cannot open shared object file". `torch`
# does this in its __init__ for its own libs, but `python -m sglang.launch_server`
# imports sgl_kernel before any torch path setup.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    _nvidia_libs=""
    for _lib_dir in "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/*/lib; do
        [[ -d "${_lib_dir}" ]] || continue
        _nvidia_libs+="${_lib_dir}:"
    done
    if [[ -n "${_nvidia_libs}" ]]; then
        export LD_LIBRARY_PATH="${_nvidia_libs}${LD_LIBRARY_PATH:-}"
    fi
    unset _nvidia_libs _lib_dir
fi

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

    if [[ -n "${SGLANG_LOG_FILE}" ]]; then
        mkdir -p "$(dirname "${SGLANG_LOG_FILE}")"
    fi
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

    echo "sglang launch: port=${SGLANG_PORT} gpus=${SGLANG_CUDA_VISIBLE_DEVICES} (tp=${gpu_count}) model=${SGLANG_MODEL_PATH}${SGLANG_CONTEXT_LENGTH:+ ctx=${SGLANG_CONTEXT_LENGTH}} fg=${SGLANG_FOREGROUND}"

    (
        # Save all values before unsetting, so they don't leak into sglang's
        # child processes (e.g. scheduler) and cause port conflicts.
        _cuda_devs="${SGLANG_CUDA_VISIBLE_DEVICES}"
        _model="${SGLANG_MODEL_PATH}"
        _host="${SGLANG_HOST}"
        _port="${SGLANG_PORT}"
        _mem="${SGLANG_MEM_FRACTION_STATIC}"
        _ctx="${SGLANG_CONTEXT_LENGTH}"
        _pgsz="${SGLANG_PAGE_SIZE}"
        _hicache="${SGLANG_HICACHE_SIZE}"
        _tool="${SGLANG_TOOL_CALL_PARSER}"
        _reason="${SGLANG_REASONING_PARSER}"
        _hier="${SGLANG_ENABLE_HIERARCHICAL_CACHE}"
        unset SGLANG_PORT SGLANG_HOST SGLANG_MODEL_PATH SGLANG_CUDA_VISIBLE_DEVICES \
              SGLANG_MEM_FRACTION_STATIC SGLANG_CONTEXT_LENGTH SGLANG_PAGE_SIZE \
              SGLANG_HICACHE_SIZE SGLANG_TOOL_CALL_PARSER SGLANG_REASONING_PARSER \
              SGLANG_EXTRA_ARGS SGLANG_ENABLE_HIERARCHICAL_CACHE SGLANG_FOREGROUND

        hier_args=()
        if [[ "${_hier}" == "1" ]]; then
            hier_args=(--enable-hierarchical-cache --hicache-size "${_hicache}")
        fi

        ctx_args=()
        if [[ -n "${_ctx}" ]]; then
            ctx_args=(--context-length "${_ctx}")
        fi

        reason_args=()
        if [[ -n "${_reason}" ]]; then
            reason_args=(--reasoning-parser "${_reason}")
        fi

        SGLANG_KV_MEMTRACE=0 \
        SGLANG_KV_MEMTRACE_STRICT=0 \
        CUDA_VISIBLE_DEVICES="${_cuda_devs}" python3 -m sglang.launch_server \
            --model-path "${_model}" \
            --host "${_host}" \
            --port "${_port}" \
            "${tp_args[@]}" \
            "${ctx_args[@]}" \
            --enable-metrics \
            --enable-cache-report \
            --mem-fraction-static "${_mem}" \
            "${hier_args[@]}" \
            --page-size "${_pgsz}" \
            --tool-call-parser "${_tool}" \
            "${reason_args[@]}" \
            "${extra_args[@]}"
    ) >>"${SGLANG_LOG_FILE:-/dev/stdout}" 2>&1 &

    sglang_pid=$!
    echo "${sglang_pid}" > "${SGLANG_PID_FILE}"
    echo "sglang started, pid=${sglang_pid}"
    if [[ -n "${SGLANG_LOG_FILE}" ]]; then
        echo "sglang log=${SGLANG_LOG_FILE}"
    else
        echo "sglang log=<inherits parent stdout>"
    fi

    # When invoked through Makefile's _ensure_serve (the default), block on
    # the child so `kill -0 $bash_pid` stays true while sglang is up. Manual
    # daemon mode keeps the fire-and-forget behavior via SGLANG_FOREGROUND=0.
    if [[ "${SGLANG_FOREGROUND}" == "1" ]]; then
        wait "${sglang_pid}"
    fi
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
