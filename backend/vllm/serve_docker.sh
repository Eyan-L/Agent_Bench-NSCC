#!/usr/bin/env bash
# backend/vllm/serve_docker.sh — containerized vLLM launcher with MIG support.
#
# Alternative to backend/vllm/serve.sh (JSON-config-driven native launch).
# Use this when you need GPU isolation via MIG slice binding, vendor-image
# pinning, or want the LLM serving container alongside the agent containers.
# `nvidia-smi -L` (the long `MIG-...` string) and the container is exposed
# only that slice — useful for shared-node experiments and the RQ3++
# slice-size sweep.
#
# Keeps the same argv surface as the old host launcher (preset / model /
# port / max-model-len) so existing make targets, sweep yaml, and PBS jobs
# don't need to know about the docker switch. The host fallback lives at
# `backend/vllm/serve_host.sh`.
#
# Defaults to detached: the container survives shell exit, `make serve-stop`
# kills it, `make serve-logs` follows its stdout.
#
# Env:
#   VLLM_IMAGE              container image (default: vllm/vllm-openai:v0.11.0)
#   VLLM_MIG_UUID           bind one MIG slice (CDI format `MIG-<uuid>`).
#                           When set: TP is forced to 1 and `--gpus` becomes
#                           `--device nvidia.com/gpu=MIG-<uuid>`.
#   VLLM_CONTAINER_NAME     docker container name (default: oc-vllm-serve)
#   VLLM_DETACH             1 = detach + tail logs (default), 0 = foreground
#   VLLM_HOST_HF_HOME       host HF cache dir (default: $HF_HOME or ~/.cache/huggingface)
#   VLLM_EXTRA_DOCKER_ARGS  pass-through to docker run (e.g. "--memory 64g")
#   HUGGING_FACE_HUB_TOKEN  forwarded into the container if set
#
# Args: same as serve_host.sh (--preset, --model, --port, --max-model-len).

set -euo pipefail

PRESET="default"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.11.0}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-oc-vllm-serve}"
VLLM_DETACH="${VLLM_DETACH:-1}"
VLLM_MIG_UUID="${VLLM_MIG_UUID:-}"
VLLM_HOST_HF_HOME="${VLLM_HOST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
VLLM_EXTRA_DOCKER_ARGS="${VLLM_EXTRA_DOCKER_ARGS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)         PRESET="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        --foreground)     VLLM_DETACH=0; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---- Pre-flight ---------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not on PATH. Use backend/vllm/serve_host.sh for native launch." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker daemon not reachable." >&2
    exit 1
fi
# nvidia runtime presence — best-effort sanity check.
if ! docker info 2>/dev/null | grep -qiE 'nvidia|cdi'; then
    echo "WARN: docker does not advertise nvidia runtime / CDI; GPU bind may fail." >&2
fi

# ---- GPU exposure: full GPUs vs single MIG slice ------------------------

GPU_ARGS=()
GPU_DESC=""
MIG_PROFILE=""

if [[ -n "$VLLM_MIG_UUID" ]]; then
    # Normalize: accept either "MIG-<uuid>" or bare "<uuid>".
    MIG_FULL="${VLLM_MIG_UUID#MIG-}"
    MIG_FULL="MIG-${MIG_FULL}"

    if command -v nvidia-smi >/dev/null 2>&1; then
        if ! nvidia-smi -L 2>/dev/null | grep -qF "$MIG_FULL"; then
            echo "ERROR: MIG slice $MIG_FULL not found in nvidia-smi -L output." >&2
            echo "       Run 'make mig-status' to list available slices." >&2
            exit 3
        fi
        # Best-effort profile lookup (e.g. "MIG 1g.24gb") for the manifest tag.
        MIG_PROFILE="$(nvidia-smi -L | awk -v u="$MIG_FULL" '
            $0 ~ "MIG " { name=$0; next }
            $0 ~ u { print name; exit }
        ' | sed -E 's/.*\(([^)]*)\).*/\1/' | tr -d ' ' || true)"
    fi

    # MIG slices are isolated — TP > 1 across them is impossible.
    if [[ "$TP" -ne 1 ]]; then
        echo "[serve_vllm] MIG mode: forcing TP=1 (slice $MIG_FULL)" >&2
        TP=1
    fi

    GPU_ARGS=(--device "nvidia.com/gpu=${MIG_FULL}")
    GPU_DESC="MIG slice $MIG_FULL${MIG_PROFILE:+ ($MIG_PROFILE)}"
else
    # All visible GPUs. CDI-style 'all' selector works with both legacy nvidia
    # runtime and modern CDI; docker resolves either path.
    GPU_ARGS=(--gpus all)
    GPU_DESC="all visible GPUs (TP=$TP)"
fi

# ---- Port collision check -----------------------------------------------

if command -v ss >/dev/null 2>&1 && ss -tlnH "sport = :$PORT" 2>/dev/null | grep -q ':'; then
    echo "ERROR: port $PORT already in use. Stop the other server or set PORT=." >&2
    ss -tlnpH "sport = :$PORT" 2>/dev/null | head -3 >&2 || true
    exit 4
fi

# ---- Container name collision -------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -qx "$VLLM_CONTAINER_NAME"; then
    if [[ "${OC_FORCE_RM:-0}" = "1" ]]; then
        echo "[serve_vllm] removing existing container $VLLM_CONTAINER_NAME (OC_FORCE_RM=1)" >&2
        docker rm -f "$VLLM_CONTAINER_NAME" >/dev/null
    else
        echo "ERROR: container '$VLLM_CONTAINER_NAME' already exists." >&2
        echo "       Stop it ('make serve-stop') or rerun with OC_FORCE_RM=1." >&2
        exit 5
    fi
fi

# ---- vLLM arg construction (mirrors serve_host.sh presets) --------------

BASE_FLAGS=(
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --trust-remote-code
)

declare -a PRESET_FLAGS
CONFIG_ID=""
case "$PRESET" in
    default)
        CONFIG_ID="vllm-default"
        PRESET_FLAGS=(--max-num-seqs 128 --enable-prefix-caching) ;;
    latency)
        CONFIG_ID="vllm-latency"
        PRESET_FLAGS=(--max-num-seqs 8 --enable-prefix-caching --scheduling-policy priority) ;;
    throughput)
        CONFIG_ID="vllm-throughput"
        PRESET_FLAGS=(--max-num-seqs 256 --enable-prefix-caching --enable-expert-parallel) ;;
    context)
        CONFIG_ID="vllm-context"
        PRESET_FLAGS=(--max-num-seqs 32 --enable-prefix-caching --kv-cache-dtype fp8) ;;
    *)
        echo "Unknown preset: $PRESET (default|latency|throughput|context)" >&2
        exit 2 ;;
esac

# MIG suffix in config_id so manifests record exactly which slice produced
# the trajectory. Keeps the RQ3 sweep table honest when slice size varies.
if [[ -n "$MIG_PROFILE" ]]; then
    CONFIG_ID="${CONFIG_ID}-mig-${MIG_PROFILE}"
elif [[ -n "$VLLM_MIG_UUID" ]]; then
    CONFIG_ID="${CONFIG_ID}-mig"
fi

# ---- Compose `docker run` -----------------------------------------------

DOCKER_ARGS=(
    run
    --name "$VLLM_CONTAINER_NAME"
    --network host
    --ipc host
    --shm-size 16g
    "${GPU_ARGS[@]}"
    -v "${VLLM_HOST_HF_HOME}:/root/.cache/huggingface:rw"
    -e VLLM_USAGE_SOURCE=oc-eval-docker
)
[[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && DOCKER_ARGS+=(-e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN")
[[ "$VLLM_DETACH" = "1" ]] && DOCKER_ARGS+=(-d) || DOCKER_ARGS+=(--rm -it)
# shellcheck disable=SC2206
[[ -n "$VLLM_EXTRA_DOCKER_ARGS" ]] && DOCKER_ARGS+=($VLLM_EXTRA_DOCKER_ARGS)
DOCKER_ARGS+=(
    "$VLLM_IMAGE"
    --model "$MODEL"
    "${BASE_FLAGS[@]}"
    "${PRESET_FLAGS[@]}"
)

# ---- Banner -------------------------------------------------------------

echo "[serve_vllm] image:        $VLLM_IMAGE"
echo "[serve_vllm] container:    $VLLM_CONTAINER_NAME"
echo "[serve_vllm] gpu binding:  $GPU_DESC"
echo "[serve_vllm] preset:       $PRESET (config_id=$CONFIG_ID)"
echo "[serve_vllm] endpoint:     http://localhost:$PORT/v1"
echo "[serve_vllm] HF cache:     $VLLM_HOST_HF_HOME"
echo "[serve_vllm] command:"
printf "  docker"; printf " %q" "${DOCKER_ARGS[@]}"; printf "\n\n"

# Persist config_id for the experiment driver.
mkdir -p "${KIT_ROOT:-$(pwd)}/data"
echo "$CONFIG_ID" > "${KIT_ROOT:-$(pwd)}/data/serving_config_id"
# Mirror MIG provenance so manifests / aggregate.py can pick it up.
{
    echo "config_id=$CONFIG_ID"
    echo "vllm_image=$VLLM_IMAGE"
    echo "mig_uuid=${VLLM_MIG_UUID:-}"
    echo "mig_profile=${MIG_PROFILE:-}"
    echo "tensor_parallel_size=$TP"
} > "${KIT_ROOT:-$(pwd)}/data/serving_metadata.env"

if [[ "${DRY_RUN:-0}" = "1" ]]; then
    echo "[serve_vllm] DRY-RUN — exiting before docker run."
    exit 0
fi

# ---- Launch -------------------------------------------------------------

if [[ "$VLLM_DETACH" = "1" ]]; then
    CID="$(docker "${DOCKER_ARGS[@]}")"
    echo "[serve_vllm] started detached (cid=${CID:0:12})"
    echo "[serve_vllm] tailing logs (Ctrl-C stops the tail; container keeps running)"
    echo "[serve_vllm] stop with: make serve-stop"
    exec docker logs -f "$VLLM_CONTAINER_NAME"
else
    exec docker "${DOCKER_ARGS[@]}"
fi
