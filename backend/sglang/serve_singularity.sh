#!/usr/bin/env bash
# sglang_launcher_singularity.sh — single-instance NSCC 2A launcher that runs
# sglang inside a singularity container. Mirrors sglang_launcher.sh in its
# env-var interface; adds these extras:
#
#   SGLANG_SIF_IMAGE        absolute path to the .sif image (required)
#   SGLANG_MOUNTS           whitespace-separated 'host:container' pairs
#   SGLANG_MODULES          whitespace-separated list of `module load` names
#                           (default: "singularity")
#   SGLANG_PATCH_DIR        absolute host path to the sglang_patch/ tree
#                           (must contain 'sglang/...' inside). When set, the
#                           launcher probes the SIF for sglang's site-packages
#                           path and overlays every *.py file under
#                           $SGLANG_PATCH_DIR/sglang via additional --bind
#                           flags. Leave unset to skip patch overlaying.
#
# Typical invocation (from experiment-bin or by hand):
#
#   SGLANG_SIF_IMAGE=/scratch/.../sglang.sif \
#   SGLANG_MOUNTS="/scratch/.../models:/models" \
#   SGLANG_MODULES="singularity" \
#   SGLANG_MODEL_PATH=/models/Qwen3-30B-A3B-Instruct-2507 \
#   SGLANG_CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   SGLANG_PORT=30000 \
#   ./sglang_launcher_singularity.sh start
#
# Note: SGLANG_MODEL_PATH should be the in-container path (e.g. /models/...)
# once you've added the matching host-path bind in SGLANG_MOUNTS. All other
# SGLANG_* knobs have the same meaning as in sglang_launcher.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ACTION="${1:-start}"

SGLANG_PID_FILE="${SGLANG_PID_FILE:-${PROJECT_ROOT}/.sglang.pid}"
SGLANG_LOG_FILE="${SGLANG_LOG_FILE:-${PROJECT_ROOT}/sglang.log}"

# Runtime knobs — shared with sglang_launcher.sh.
SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
SGLANG_HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-60}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
SGLANG_ENABLE_HIERARCHICAL_CACHE="${SGLANG_ENABLE_HIERARCHICAL_CACHE:-1}"

# Singularity-specific.
SGLANG_SIF_IMAGE="${SGLANG_SIF_IMAGE:-}"
SGLANG_MOUNTS="${SGLANG_MOUNTS:-}"
SGLANG_MODULES="${SGLANG_MODULES:-singularity}"
SGLANG_PATCH_DIR="${SGLANG_PATCH_DIR:-}"

_gpu_count() {
    local devs="$1"
    if [[ -z "${devs}" ]]; then
        echo 0
        return
    fi
    local count=0
    local IFS=','
    # shellcheck disable=SC2206
    local parts=(${devs})
    for p in "${parts[@]}"; do
        [[ -n "${p}" ]] && count=$((count + 1))
    done
    echo "${count}"
}

_validate() {
    if [[ -z "${SGLANG_SIF_IMAGE}" ]]; then
        echo "error: SGLANG_SIF_IMAGE is required (path to the .sif image)" >&2
        return 1
    fi
    if [[ ! -f "${SGLANG_SIF_IMAGE}" ]]; then
        echo "error: SIF image not found: ${SGLANG_SIF_IMAGE}" >&2
        return 1
    fi
    if [[ -n "${SGLANG_PATCH_DIR}" && ! -d "${SGLANG_PATCH_DIR}/sglang" ]]; then
        echo "error: SGLANG_PATCH_DIR=${SGLANG_PATCH_DIR} does not contain a 'sglang/' subdir" >&2
        return 1
    fi
}

# Probe the SIF for sglang's site-packages parent (the dir that contains the
# `sglang/` package), so we know where to bind-mount patched files. Prints
# the resolved path on stdout; caller captures it into SGLANG_SITE_PACKAGES.
#
# Why parent-of-parent: python's `sglang.__file__` returns
# `<site-packages>/sglang/__init__.py`, dirname → `<site-packages>/sglang`,
# dirname again → `<site-packages>`. Our patch tree starts at `sglang/...`,
# so binds expand as `<site-packages>/sglang/...`.
_probe_sglang_site_packages() {
    local module_prefix=""
    for m in ${SGLANG_MODULES}; do
        module_prefix+="module load ${m} && "
    done
    local py="import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))"
    local probe_cmd="${module_prefix}singularity exec ${SGLANG_SIF_IMAGE} python3 -c '${py}'"
    local site_packages
    site_packages="$(bash -c "${probe_cmd}" 2>/dev/null | tail -1)"
    if [[ -z "${site_packages}" ]]; then
        echo "error: probe for sglang site-packages path failed; tried: ${probe_cmd}" >&2
        return 1
    fi
    echo "${site_packages}"
}

# Build the singularity command, emitted as a single shell string. We keep it
# as a string (not an array) because `module load` is a shell function, not
# an executable, and must run in the same shell as `singularity run`.
_build_singularity_cmd() {
    local gpu_devs="$1"
    local python_cmd="$2"

    local module_prefix=""
    for m in ${SGLANG_MODULES}; do
        module_prefix+="module load ${m} && "
    done

    local bind_args=""
    for pair in ${SGLANG_MOUNTS}; do
        # each pair is 'host:container' (or 'host:container:opts'); pass through.
        bind_args+=" --bind ${pair}"
    done

    # Patch overlay: for each *.py under $SGLANG_PATCH_DIR/sglang, add a
    # bind pointing at the matching file inside the container's
    # site-packages. $SGLANG_SITE_PACKAGES must be set by start_sglang.
    if [[ -n "${SGLANG_PATCH_DIR}" && -n "${SGLANG_SITE_PACKAGES:-}" ]]; then
        while IFS= read -r -d '' host_file; do
            rel="${host_file#${SGLANG_PATCH_DIR}/}"      # e.g. sglang/srt/managers/scheduler.py
            target="${SGLANG_SITE_PACKAGES}/${rel}"
            bind_args+=" --bind ${host_file}:${target}"
        done < <(find "${SGLANG_PATCH_DIR}/sglang" -type f -name '*.py' -print0)
    fi

    local env_args=""
    if [[ -n "${gpu_devs}" ]]; then
        env_args+=" --env CUDA_VISIBLE_DEVICES=${gpu_devs}"
    fi
    env_args+=" --env SGLANG_KV_MEMTRACE=0 --env SGLANG_KV_MEMTRACE_STRICT=0"

    # Forward HuggingFace-related env into the container so the HF cache on
    # NSCC $HF_HOME (same-path bound) is reachable from inside. Only forward
    # variables that are actually set on the host to avoid `--env FOO=` which
    # would shadow any default.
    for v in HF_HOME HF_HUB_OFFLINE HF_TOKEN TRANSFORMERS_CACHE; do
        if [[ -n "${!v:-}" ]]; then
            env_args+=" --env ${v}=${!v}"
        fi
    done

    echo "${module_prefix}singularity run --nv${env_args}${bind_args} ${SGLANG_SIF_IMAGE} ${python_cmd}"
}

start_sglang() {
    _validate

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

    # Resolve the in-container sglang site-packages path once, if patching.
    # Exported so _build_singularity_cmd (called below) can see it.
    if [[ -n "${SGLANG_PATCH_DIR}" ]]; then
        echo "patch: probing SIF for sglang site-packages (${SGLANG_SIF_IMAGE})..."
        export SGLANG_SITE_PACKAGES
        SGLANG_SITE_PACKAGES="$(_probe_sglang_site_packages)"
        echo "patch: site-packages=${SGLANG_SITE_PACKAGES}"
        local patch_count
        patch_count="$(find "${SGLANG_PATCH_DIR}/sglang" -type f -name '*.py' 2>/dev/null | wc -l)"
        echo "patch: overlaying ${patch_count} files from ${SGLANG_PATCH_DIR}/sglang"
    fi

    local gpu_count
    gpu_count="$(_gpu_count "${SGLANG_CUDA_VISIBLE_DEVICES}")"
    if [[ "${gpu_count}" -lt 1 ]]; then
        echo "error: SGLANG_CUDA_VISIBLE_DEVICES is empty; set it to the GPU ids to use" >&2
        return 1
    fi

    local tp_flag=""
    if [[ "${gpu_count}" -gt 1 ]]; then
        tp_flag=" --tp ${gpu_count}"
    fi

    local hier_flags=""
    if [[ "${SGLANG_ENABLE_HIERARCHICAL_CACHE}" == "1" ]]; then
        hier_flags=" --enable-hierarchical-cache --hicache-size ${SGLANG_HICACHE_SIZE}"
    fi

    local python_cmd="python3 -m sglang.launch_server \
        --model-path ${SGLANG_MODEL_PATH} \
        --host ${SGLANG_HOST} \
        --port ${SGLANG_PORT}${tp_flag} \
        --enable-metrics \
        --enable-cache-report \
        --mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}${hier_flags} \
        --page-size ${SGLANG_PAGE_SIZE} \
        --tool-call-parser ${SGLANG_TOOL_CALL_PARSER} ${SGLANG_EXTRA_ARGS}"

    local full_cmd
    full_cmd="$(_build_singularity_cmd "${SGLANG_CUDA_VISIBLE_DEVICES}" "${python_cmd}")"

    echo "sglang singularity launch: sif=${SGLANG_SIF_IMAGE}"
    echo "  port=${SGLANG_PORT} gpus=${SGLANG_CUDA_VISIBLE_DEVICES} (tp=${gpu_count}) model=${SGLANG_MODEL_PATH}"
    echo "  cmd: ${full_cmd}"

    # Run in its own process group so `kill -- -PID` later tears down both the
    # singularity wrapper and its python child.
    #
    # Unset SGLANG_* (and the matching forwarded CUDA_VISIBLE_DEVICES) BEFORE
    # the subshell so they don't leak into the container. Singularity passes
    # the host env through by default; without this, TP worker ranks all read
    # SGLANG_PORT=30000 from env and race to bind the same socket — the main
    # server wins, every other rank dies with "address already in use", and
    # the server's own warmup then times out.
    (
        unset SGLANG_PORT SGLANG_HOST SGLANG_MODEL_PATH SGLANG_CUDA_VISIBLE_DEVICES \
              SGLANG_MEM_FRACTION_STATIC SGLANG_PAGE_SIZE SGLANG_HICACHE_SIZE \
              SGLANG_TOOL_CALL_PARSER SGLANG_EXTRA_ARGS SGLANG_ENABLE_HIERARCHICAL_CACHE \
              SGLANG_SIF_IMAGE SGLANG_MOUNTS SGLANG_MODULES SGLANG_PATCH_DIR \
              SGLANG_SITE_PACKAGES SGLANG_HEALTH_URL SGLANG_LOG_FILE SGLANG_PID_FILE \
              CUDA_VISIBLE_DEVICES
        setsid bash -c "${full_cmd}"
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

    # setsid-created process: kill the whole group first, then fall back.
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
    start)   start_sglang ;;
    stop)    stop_sglang ;;
    restart) stop_sglang; start_sglang ;;
    status)  status_sglang ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 2
        ;;
esac
