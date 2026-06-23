#!/usr/bin/env bash
# scripts/serve_only/serve_vllm.sh — GPU-side, serve-ONLY launcher.
#
# Decoupled workflow: this runs on the GPU host and does nothing
# but start vLLM as an OpenAI-compatible server bound on 0.0.0.0. The agent +
# scoring run on a separate CPU node and reach this server over the network (a
# direct route where reachable, or an SSH tunnel for hosts behind a ProxyJump —
# see scripts/serve_only/README.md).
#
# It is a thin wrapper over the Python launcher (python -m agent_bench.serving.launch
# --backend vllm), which owns vLLM config resolution + command planning. This script
# only pins the decoupled serving profile, selects the host-local vLLM venv (VLLM_VENV;
# decoupled.yaml's `venv:` is a placeholder), and supervises the process group for a
# clean teardown. All vLLM flags come from the model + serving profiles.
#
# Env (all optional; sensible defaults):
#   VLLM_VENV       Path to the vLLM venv on THIS host.
#                   REQUIRED unless `vllm` is already on PATH.
#   MODEL_CONFIG    profiles/models/*.json — model identity. Its `name` is the
#                   served model id AND the weights vLLM loads, so it MUST match
#                   the --model-config used by the CPU-side runner. Default:
#                   profiles/models/qwen36_35b_a3b_local.json
#   MODEL           Override the weights path/id only (advanced). If you set this
#                   to something other than the model config `name`, the served
#                   model id will not match what the agent requests — keep them
#                   aligned or prefer a MODEL_CONFIG whose `name` is valid here.
#   SERVING_CONFIG  profiles/serving/vllm/*.yaml. Default: decoupled.yaml.
#   TP              Tensor-parallel size. Default 1 (Qwen3.6-35B-A3B-FP8 fits one
#                   48GB A6000; set 2 if you have two GPUs).
#   CUDA_VISIBLE_DEVICES  Default "0".
#   PORT            Default 8000. Server binds 0.0.0.0:$PORT.
#   LOG_DIR         Where vllm.log goes. Default ./logs/serve_only.
#
# Usage (on the GPU host, inside tmux):
#   VLLM_VENV=$HOME/vllm/.venv TP=1 CUDA_VISIBLE_DEVICES=0 \
#       bash scripts/serve_only/serve_vllm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

MODEL_CONFIG="${MODEL_CONFIG:-$KIT_ROOT/profiles/models/qwen36_35b_a3b_local.json}"
SERVING_CONFIG="${SERVING_CONFIG:-$KIT_ROOT/profiles/serving/vllm/decoupled.yaml}"
# Resolve relative config paths against KIT_ROOT. A bare relative path (e.g.
# MODEL_CONFIG=profiles/models/foo.json, as remote_serve.sh forwards it) is
# otherwise handed straight to python's open(), which is CWD-dependent and
# brittle. Make it absolute so the script works from any CWD.
[[ "$MODEL_CONFIG" = /* ]]   || MODEL_CONFIG="$KIT_ROOT/$MODEL_CONFIG"
[[ "$SERVING_CONFIG" = /* ]] || SERVING_CONFIG="$KIT_ROOT/$SERVING_CONFIG"
if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "[serve_only] FATAL: model config not found: $MODEL_CONFIG" >&2
    echo "[serve_only]   (is this checkout synced? the file may be on a newer commit.)" >&2
    exit 1
fi
PORT="${PORT:-8000}"
TP="${TP:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Translate this script's bare knobs into the names the Python launcher reads
# (launch_plan._build_vllm_plan). CUDA_VISIBLE_DEVICES is read directly.
export AB_SERVING_PORT="$PORT"
export VLLM_HOST="${BIND_HOST:-0.0.0.0}"
export AB_SERVING_TENSOR_PARALLEL_SIZE="$TP"
# fanout's per-lane packing override arrives as the bare GPU_MEM_UTIL; forward it.
[[ -n "${GPU_MEM_UTIL:-}" ]] && export VLLM_GPU_UTIL="$GPU_MEM_UTIL"
# MODEL overrides the weights path / served id (advanced); launcher reads VLLM_MODEL.
[[ -n "${MODEL:-}" ]] && export VLLM_MODEL="$MODEL"
LOG_DIR="${LOG_DIR:-$KIT_ROOT/logs/serve_only}"
mkdir -p "$LOG_DIR"
# PGID of the supervised vllm process group, so a signal (Ctrl-C, kill, or the
# remote stop in remote_serve.sh) tears down vllm AND its V1 EngineCore/worker
# subprocesses in one shot. See the supervisor block at the end of this file.
PIDFILE="${PIDFILE:-$LOG_DIR/serve_vllm.pgid}"

# The launcher reads these two config axes by path.
export AB_MODEL_CONFIG="$MODEL_CONFIG"
export AB_SERVING_CONFIG="$SERVING_CONFIG"
# decoupled.yaml's venv: is a placeholder; VLLM_VENV (if set) selects the local
# vLLM venv on this host. The launcher activates it (prepends its bin to PATH)
# when vllm isn't already on PATH.
export VLLM_VENV="${VLLM_VENV:-}"
export VLLM_LOG_FILE="$LOG_DIR/vllm.log"

# Run the launcher with the venv's python when VLLM_VENV is set: that interpreter
# has tomllib (system python3 may be 3.9 and silently drop the profile) and sits
# next to the vllm executable. PYTHONPATH lets `agent_bench` import whether or not
# the venv pip-installed it.
PY="python3"
if [[ -n "${VLLM_VENV:-}" && -x "$VLLM_VENV/bin/python" ]]; then
    PY="$VLLM_VENV/bin/python"
fi
export PYTHONPATH="$KIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Hopper (H200) flashinfer kernels JIT-compile against <cuda/ptx>, a CCCL header
# the system nvcc 12.0 toolkit lacks (build dies "cuda/ptx: No such file"). If a
# newer CCCL is pip-installed in the venv (nvidia-cuda-cccl-cu12), put its include
# on every nvcc invocation so those kernels compile. No-op where it isn't present
# (Ampere hosts compile fine without it), so this is safe to run on every host.
for _cccl_inc in "${VLLM_VENV:-/nonexistent}"/lib/python*/site-packages/nvidia/cuda_cccl/include; do
    [[ -d "$_cccl_inc/cuda" ]] || continue
    export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-I$_cccl_inc"
    echo "[serve_only] CCCL shim: NVCC_APPEND_FLAGS += -I$_cccl_inc (flashinfer Hopper JIT)"
done

# A MODEL override is authoritative for the served id (the launcher uses it as the
# vllm positional arg, with no --served-model-name). Fall back to the config name.
SERVED_NAME="${MODEL:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["name"])' "$MODEL_CONFIG")}"

echo "[serve_only] host=$(hostname) port=$PORT TP=$TP CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[serve_only] model-config=$MODEL_CONFIG"
echo "[serve_only] serving-config=$SERVING_CONFIG"
echo "[serve_only] served model id (agent must request this): $SERVED_NAME"
echo "[serve_only] vLLM venv: ${VLLM_VENV:-<vllm already on PATH>}"
echo "[serve_only] log: $VLLM_LOG_FILE"
echo "[serve_only] once ready, on the CLIENT node check: curl http://<this-host>:$PORT/v1/models"

# ---- Supervised launch (clean shutdown) ----------------------------------
# Why not `exec`: `exec vllm serve` replaces this shell, so there is no parent
# left to trap Ctrl-C/SIGTERM — and vLLM's V1 engine spawns separate EngineCore
# and TP worker processes that are NOT reaped when the launcher dies, which is
# exactly why a plain Ctrl-C / kill <pid> left the server (and GPU memory) alive
# and forced a manual `kill -9`.
#
# Instead: run serve.sh in its OWN process group (`set -m` puts each background
# job in a group whose PGID == the job PID), record that PGID, and on any
# INT/TERM signal the whole group is sent SIGTERM, then escalated to SIGKILL if
# vllm hasn't exited within the grace window. `remote_serve.sh stop` targets the
# same PGID file, so manual and remote shutdown share one clean path.
SERVE_GRACE="${SERVE_GRACE:-20}"   # seconds to wait for graceful exit before SIGKILL

set -m
"$PY" -m agent_bench.serving.launch --backend vllm "$@" &
SERVE_PID=$!
SERVE_PGID=$SERVE_PID            # group leader in monitor mode
echo "$SERVE_PGID" > "$PIDFILE"
echo "[serve_only] supervising vllm process group pgid=$SERVE_PGID (pidfile: $PIDFILE)"
echo "[serve_only] stop with: Ctrl-C, or  kill -TERM -$SERVE_PGID , or remote_serve.sh stop"

_shutdown() {
    local sig="${1:-TERM}"
    echo "" >&2
    echo "[serve_only] received $sig — stopping vllm group $SERVE_PGID ..." >&2
    kill -TERM -"$SERVE_PGID" 2>/dev/null || true
    for _ in $(seq 1 "$SERVE_GRACE"); do
        kill -0 "$SERVE_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[serve_only] still alive after ${SERVE_GRACE}s — SIGKILL group $SERVE_PGID" >&2
        kill -KILL -"$SERVE_PGID" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
}
trap '_shutdown INT;  exit 130' INT
trap '_shutdown TERM; exit 143' TERM

wait "$SERVE_PID"
rc=$?
rm -f "$PIDFILE"
exit "$rc"
