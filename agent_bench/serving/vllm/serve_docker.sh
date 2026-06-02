#!/usr/bin/env bash
# agent_bench/serving/vllm/serve_docker.sh — containerized vLLM launcher with MIG support.
#
# Alternative to agent_bench/serving/vllm/serve.sh (JSON-config-driven native launch).
# Use this when you need GPU isolation via MIG slice binding, vendor-image
# pinning, or want the LLM serving container alongside the agent containers.
# `nvidia-smi -L` (the long `MIG-...` string) and the container is exposed
# only that slice — useful for shared-node experiments and the RQ3++
# slice-size sweep.
#
# Uses the Agent Bench serving argv surface (preset / model-name /
# serving-port / serving-max-model-len) while keeping vLLM-native env names
# for backend-specific docker controls.
#
# Defaults to detached: the container survives shell exit, `make serve-stop`
# kills it, `make serve-logs` follows its stdout.
#
# Env:
#   VLLM_IMAGE              container image (default: vllm/vllm-openai:v0.20.0)
#   VLLM_MIG_UUID           bind one MIG slice (CDI format `MIG-<uuid>`).
#                           When set: AB_SERVING_TENSOR_PARALLEL_SIZE is forced to 1 and `--gpus` becomes
#                           `--device nvidia.com/gpu=MIG-<uuid>`.
#   VLLM_CONTAINER_NAME     docker container name (default: oc-vllm-serve)
#   VLLM_DETACH             1 = detach + tail logs (default), 0 = foreground
#   VLLM_TAIL_LOGS          1 = follow logs after detached start (default), 0 = return
#   VLLM_HOST_HF_HOME       host HF cache dir (default: $HF_HOME or ~/.cache/huggingface)
#   VLLM_EXTRA_DOCKER_ARGS  pass-through to docker run (e.g. "--memory 64g")
#   HUGGING_FACE_HUB_TOKEN  forwarded into the container if set
#
# Args: --preset, --model-name, --serving-port, --serving-max-model-len.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB_REPO_ROOT="${AB_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

resolve_profile_ref() {
    local base="$1"
    local ref="$2"
    shift 2
    if [[ "$ref" == */*.* || "$ref" == /* ]]; then
        echo "$ref"
        return 0
    fi
    local ext
    for ext in "$@"; do
        if [[ -f "$base/$ref.$ext" ]]; then
            echo "$base/$ref.$ext"
            return 0
        fi
    done
    echo "$base/$ref.$1"
}

MODEL_CFG="${AB_MODEL_CONFIG:-qwen36_27b}"
MODEL_CFG="$(resolve_profile_ref "$AB_REPO_ROOT/profiles/models" "$MODEL_CFG" json yaml yml)"
if [[ -f "$MODEL_CFG" ]]; then
    eval "$(python3 "$AB_REPO_ROOT/scripts/load_model_env.py" "$MODEL_CFG")"
fi

SERVING_CFG="${AB_SERVING_CONFIG:-vllm/default}"
SERVING_CFG="$(resolve_profile_ref "$AB_REPO_ROOT/profiles/serving" "$SERVING_CFG" yaml yml json toml)"
if [[ -f "$SERVING_CFG" ]]; then
    eval "$(python3 "$AB_REPO_ROOT/scripts/load_serving_env.py" "$SERVING_CFG")"
fi

AB_SERVING_PRESET="${AB_SERVING_PRESET:-default}"
AB_MODEL_NAME="${AB_MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
AB_SERVING_TENSOR_PARALLEL_SIZE="${AB_SERVING_TENSOR_PARALLEL_SIZE:-2}"
AB_SERVING_PORT="${AB_SERVING_PORT:-8000}"
AB_SERVING_MAX_MODEL_LEN="${AB_SERVING_MAX_MODEL_LEN:-${AB_MODEL_CONTEXT_WINDOW:-131072}}"
AB_SERVING_GPU_MEMORY_UTILIZATION="${AB_SERVING_GPU_MEMORY_UTILIZATION:-0.90}"
AB_SERVING_BIND_HOST="${AB_SERVING_BIND_HOST:-0.0.0.0}"

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-oc-vllm-serve}"
VLLM_DETACH="${VLLM_DETACH:-1}"
VLLM_TAIL_LOGS="${VLLM_TAIL_LOGS:-1}"
VLLM_MIG_UUID="${VLLM_MIG_UUID:-}"
VLLM_HOST_HF_HOME="${VLLM_HOST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
VLLM_EXTRA_DOCKER_ARGS="${VLLM_EXTRA_DOCKER_ARGS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)         AB_SERVING_PRESET="$2"; shift 2 ;;
        --model-name)     AB_MODEL_NAME="$2"; shift 2 ;;
        --serving-port)   AB_SERVING_PORT="$2"; shift 2 ;;
        --serving-max-model-len) AB_SERVING_MAX_MODEL_LEN="$2"; shift 2 ;;
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
    echo "ERROR: docker not on PATH. Use agent_bench/serving/vllm/serve_host.sh for native launch." >&2
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
        # Best-effort profile lookup (e.g. "1g.24gb") for the manifest tag.
        MIG_PROFILE="$(nvidia-smi -L | awk -v u="$MIG_FULL" '
            $0 ~ u {
                prof = $0
                sub(/.*MIG /, "", prof)
                sub(/ +Device.*/, "", prof)
                print prof
                exit
            }
        ' | tr -d ' ' || true)"
    fi

    # MIG slices are isolated — AB_SERVING_TENSOR_PARALLEL_SIZE > 1 across them is impossible.
    if [[ "$AB_SERVING_TENSOR_PARALLEL_SIZE" -ne 1 ]]; then
        echo "[serve_vllm] MIG mode: forcing AB_SERVING_TENSOR_PARALLEL_SIZE=1 (slice $MIG_FULL)" >&2
        AB_SERVING_TENSOR_PARALLEL_SIZE=1
    fi

    GPU_ARGS=(--device "nvidia.com/gpu=${MIG_FULL}")
    GPU_DESC="MIG slice $MIG_FULL${MIG_PROFILE:+ ($MIG_PROFILE)}"
else
    # All visible GPUs. CDI-style 'all' selector works with both legacy nvidia
    # runtime and modern CDI; docker resolves either path.
    GPU_ARGS=(--gpus all)
    GPU_DESC="all visible GPUs (AB_SERVING_TENSOR_PARALLEL_SIZE=$AB_SERVING_TENSOR_PARALLEL_SIZE)"
fi

# ---- Port collision check -----------------------------------------------

if command -v ss >/dev/null 2>&1 && ss -tlnH "sport = :$AB_SERVING_PORT" 2>/dev/null | grep -q ':'; then
    echo "ERROR: port $AB_SERVING_PORT already in use. Stop the other server or set AB_SERVING_PORT=." >&2
    ss -tlnpH "sport = :$AB_SERVING_PORT" 2>/dev/null | head -3 >&2 || true
    exit 4
fi

# ---- Container name collision -------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -qx "$VLLM_CONTAINER_NAME"; then
    if [[ "${AB_SERVING_FORCE_RM:-0}" = "1" ]]; then
        echo "[serve_vllm] removing existing container $VLLM_CONTAINER_NAME (AB_SERVING_FORCE_RM=1)" >&2
        docker rm -f "$VLLM_CONTAINER_NAME" >/dev/null
    else
        echo "ERROR: container '$VLLM_CONTAINER_NAME' already exists." >&2
        echo "       Stop it ('make serve-stop') or rerun with AB_SERVING_FORCE_RM=1." >&2
        exit 5
    fi
fi

# ---- vLLM arg construction (mirrors serve_host.sh presets) --------------

# AB_MODEL_TOOL_CALL_PARSER selects which family of <tool_call> markers vLLM extracts
# from the model output into the OpenAI-compat `tool_calls` field.
#   qwen3_coder   Qwen3-Coder / Qwen3.6 (default; the agent-bench reference model)
#   hermes        Qwen2.5-Instruct, Hermes models — `<tool_call>{...}</tool_call>`
#   ""            no parser; tools markers stay in `content` (OC plugin parses them)
AB_MODEL_TOOL_CALL_PARSER="${AB_MODEL_TOOL_CALL_PARSER:-qwen3_coder}"
AB_MODEL_REASONING_PARSER="${AB_MODEL_REASONING_PARSER-}"

BASE_FLAGS=(
    --host "$AB_SERVING_BIND_HOST"
    --port "$AB_SERVING_PORT"
    --tensor-parallel-size "$AB_SERVING_TENSOR_PARALLEL_SIZE"
    --max-model-len "$AB_SERVING_MAX_MODEL_LEN"
    --gpu-memory-utilization "$AB_SERVING_GPU_MEMORY_UTILIZATION"
    --trust-remote-code
)
if [[ -n "$AB_MODEL_TOOL_CALL_PARSER" ]]; then
    BASE_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "$AB_MODEL_TOOL_CALL_PARSER")
fi
if [[ -n "$AB_MODEL_REASONING_PARSER" ]]; then
    BASE_FLAGS+=(--reasoning-parser "$AB_MODEL_REASONING_PARSER")
fi
# Qwen3 thinking-mode routes the model's chain-of-thought to a separate
# `reasoning` field. Most agent harnesses read `content` only, so they see
# an empty response. Default OFF here; set NO_REASONING=0 to opt back in.
if [[ "${NO_REASONING:-1}" != "1" ]]; then
    BASE_FLAGS+=(--reasoning-parser qwen3)
fi
# Qwen3.6 (qwen3_5_moe arch) uses GDN linear attention; the FlashInfer
# prefill kernel for it is JIT-compiled with nvcc at first run. The
# chengzhilu/vllm image has no nvcc, so the JIT crashes the worker.
# Force the pure-Triton GDN backend to skip nvcc entirely.
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
if [[ -n "$GDN_PREFILL_BACKEND" ]]; then
    BASE_FLAGS+=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
fi

declare -a PRESET_FLAGS
CONFIG_ID=""
case "$AB_SERVING_PRESET" in
    default)
        CONFIG_ID="${AB_SERVING_CONFIG_ID:-vllm-default}"
        PRESET_FLAGS=(--max-num-seqs "${AB_SERVING_MAX_NUM_SEQS:-128}")
        [[ "${AB_SERVING_ENABLE_PREFIX_CACHING:-1}" == "1" ]] && PRESET_FLAGS+=(--enable-prefix-caching) ;;
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
        echo "Unknown preset: $AB_SERVING_PRESET (default|latency|throughput|context)" >&2
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
    --model "$AB_MODEL_NAME"
    "${BASE_FLAGS[@]}"
    "${PRESET_FLAGS[@]}"
)

# ---- Banner -------------------------------------------------------------

echo "[serve_vllm] image:        $VLLM_IMAGE"
echo "[serve_vllm] container:    $VLLM_CONTAINER_NAME"
echo "[serve_vllm] gpu binding:  $GPU_DESC"
echo "[serve_vllm] preset:       $AB_SERVING_PRESET (config_id=$CONFIG_ID)"
echo "[serve_vllm] endpoint:     http://localhost:$AB_SERVING_PORT/v1"
echo "[serve_vllm] HF cache:     $VLLM_HOST_HF_HOME"
echo "[serve_vllm] command:"
printf "  docker"; printf " %q" "${DOCKER_ARGS[@]}"; printf "\n\n"

# Persist config_id for the experiment driver.
mkdir -p "${AB_REPO_ROOT:-$(pwd)}/data"
echo "$CONFIG_ID" > "${AB_REPO_ROOT:-$(pwd)}/data/serving_config_id"
# Mirror MIG provenance so manifests / aggregate.py can pick it up.
{
    echo "config_id=$CONFIG_ID"
    echo "vllm_image=$VLLM_IMAGE"
    echo "mig_uuid=${MIG_FULL:-}"
    echo "mig_profile=${MIG_PROFILE:-}"
    echo "tensor_parallel_size=$AB_SERVING_TENSOR_PARALLEL_SIZE"
} > "${AB_REPO_ROOT:-$(pwd)}/data/serving_metadata.env"

if [[ "${DRY_RUN:-0}" = "1" ]]; then
    echo "[serve_vllm] DRY-RUN — exiting before docker run."
    exit 0
fi

# ---- Launch -------------------------------------------------------------

if [[ "$VLLM_DETACH" = "1" ]]; then
    CID="$(docker "${DOCKER_ARGS[@]}")"
    echo "[serve_vllm] started detached (cid=${CID:0:12})"
    if [[ "$VLLM_TAIL_LOGS" != "1" ]]; then
        echo "[serve_vllm] VLLM_TAIL_LOGS=0 — returning after detached start"
        exit 0
    fi
    echo "[serve_vllm] tailing logs (Ctrl-C stops the tail; container keeps running)"
    echo "[serve_vllm] stop with: make serve-stop"
    exec docker logs -f "$VLLM_CONTAINER_NAME"
else
    exec docker "${DOCKER_ARGS[@]}"
fi
