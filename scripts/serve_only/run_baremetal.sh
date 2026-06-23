#!/usr/bin/env bash
# scripts/serve_only/run_baremetal.sh — ONE-command decoupled run on a FIXED
# bare-metal GPU host (no scheduler) from ntu_cpu. It is the third sibling of
# run_ccds.sh (Slurm) and run_nscc.sh (PBS): same per-cell contract, but the GPU
# is acquired by picking a card on a known host instead of asking a scheduler.
#
# Lifecycle: pick/confirm a card (nvidia-smi probe) -> remote_serve.sh start
# (detached vllm on that host, waits for /v1/models) -> tunnel.sh (skip for a
# directly-routed host) -> run_tb_client.sh -> trap: remote_serve.sh stop + kill
# tunnel. Multiple servers can share ONE card: give each its own PORT (auto) and
# a fractional GPU_MEM_UTIL — remote_serve.sh PORT-scopes all supervisor state.
#
# Env (override K=V on the command line):
#   REMOTE           SSH alias OR user@ip of the GPU host. REQUIRED (set in .env
#                    or on the command line; see .env.example).
#   REMOTE_KIT_ROOT  Agent_Bench checkout on the GPU host. Default: ~/Agent_Bench
#   VLLM_VENV        vLLM venv on the GPU host.
#                    Default: <remote $HOME>/vllm/.venv
#   MODEL_TEMPLATE   Committed model config (params; name overridden).
#                    Default: profiles/models/qwen36_35b_a3b_fp8.json
#   MODEL_NAME       Served id + weights path. Default: <remote $HOME>/scratch/models/Qwen3.6-35B-A3B-FP8
#   SERVING_CONFIG   Default: profiles/serving/vllm/decoupled.yaml
#   CUDA_VISIBLE_DEVICES  Force the card(s) (skip the free-VRAM pick). The
#                    orchestrator sets this to place packed servers deterministically.
#   GPUS / TP        Card count / tensor-parallel. Default: count of CVD, else 1.
#   GPU_MEM_UTIL    Fractional GPU memory (pack N/card at ~1/N). Default: whole card.
#   DIRECT_ROUTE     1 = host is directly routable from this node — no tunnel; the
#                    client points at ENDPOINT_HOST (derived from REMOTE's user@ip
#                    or set ENDPOINT_HOST explicitly).
#   READY_TIMEOUT    Seconds remote_serve.sh waits for /v1/models. Default: 1200.
#   APP HARNESS_CONFIG ARM ARM_ENV SKIP_PULL
#                    Passed through to run_tb_client.sh.
#   ALLOC_MANIFEST RUN_ID  Placement-ledger path + stable id (set by the orchestrator).
set -uo pipefail

# Detach stdin so this is safe to background (`&`) for parallel fan-out — the
# inner ssh calls would otherwise read the tty and earn SIGTTIN ([N]+ Stopped).
exec </dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/serve_only/_serve_only_common.sh
source "$SCRIPT_DIR/_serve_only_common.sh"

REMOTE="${REMOTE:-}"   # REQUIRED — SSH alias or user@ip (see .env.example)
REMOTE_KIT_ROOT="${REMOTE_KIT_ROOT:-\$HOME/Agent_Bench}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-profiles/models/qwen36_35b_a3b_fp8.json}"
SERVING_CONFIG="${SERVING_CONFIG:-profiles/serving/vllm/decoupled.yaml}"
PORT="${PORT:-$(so_pick_free_port)}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
RUN_ID="${RUN_ID:-ab-bm-$(date +%s)-$$}"
ARM="${ARM:-default}"   # generic experiment-variant label (orchestrator sets it)

die() { echo "[bm] ERROR: $*" >&2; exit 1; }

[[ -n "$REMOTE" ]] || die "REMOTE is required — set the GPU host (SSH alias or user@ip) in .env or on the command line (see .env.example)."
echo "[bm] host=$REMOTE port=$PORT util=${GPU_MEM_UTIL:-<whole card>}"
ssh -G "$REMOTE" >/dev/null 2>&1 || die "'$REMOTE' is not resolvable by ssh (add a Host entry or use user@ip)."
REMOTE_HOME="$(ssh -o BatchMode=yes "$REMOTE" 'printf %s "$HOME"' 2>/dev/null)"; [[ -n "$REMOTE_HOME" ]] || die "could not resolve \$HOME on $REMOTE."
MODEL_NAME="${MODEL_NAME:-$REMOTE_HOME/scratch/models/Qwen3.6-35B-A3B-FP8}"
VLLM_VENV="${VLLM_VENV:-$REMOTE_HOME/vllm/.venv}"

# --- pick / confirm the card, and read its type + size for the manifest -------
# Probe all cards once. If CUDA_VISIBLE_DEVICES is forced, look up that index's
# name/size; otherwise pick the card with the most free VRAM (so a packed server
# lands on a card that still has room even if another server is already on it).
SMI="$(ssh "$REMOTE" 'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits' 2>/dev/null)" \
    || die "nvidia-smi failed on $REMOTE."
[[ -n "$SMI" ]] || die "no GPUs reported by nvidia-smi on $REMOTE."
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CVD="$CUDA_VISIBLE_DEVICES"
    read -r GPU_NAME GPU_MEM_MIB < <(awk -F', *' -v want="${CVD%%,*}" \
        '$1==want{gsub(/ /,"_",$2); print $2, $4}' <<<"$SMI")
else
    read -r CVD GPU_NAME GPU_MEM_MIB < <(awk -F', *' \
        '{free=$4-$3; if(NR==1||free>best){best=free; idx=$1; nm=$2; tot=$4}} END{gsub(/ /,"_",nm); print idx, nm, tot}' <<<"$SMI")
fi
[[ -n "${CVD:-}" ]] || die "could not select a GPU on $REMOTE."
GPU_NAME="${GPU_NAME:-unknown}"
GPUS="${GPUS:-$(awk -F, '{print NF}' <<<"$CVD")}"
echo "[bm] card CUDA_VISIBLE_DEVICES=$CVD GPUS=$GPUS gpu=$GPU_NAME (${GPU_MEM_MIB:-?}MiB)"

# --- client model-config (name must match the served id) ----------------------
CLIENT_CFG="$KIT_ROOT/data/serve_only/model_bm_$(basename "$MODEL_NAME").json"
so_gen_client_cfg "$KIT_ROOT/$MODEL_TEMPLATE" "$MODEL_NAME" "$CLIENT_CFG"

# Record the placement now (card + GPU type known) so even a serve failure still
# lands in the ledger with the GPU it picked.
so_write_alloc "run_id=$RUN_ID" "cluster=baremetal" "host=$REMOTE" "node=$REMOTE" \
    "gpu_name=$GPU_NAME" "gpu_mem_mib=${GPU_MEM_MIB:-}" "gpus=$GPUS" "cuda_visible_devices=$CVD" \
    "gpu_mem_util=${GPU_MEM_UTIL:-}" "port=$PORT" "model=$MODEL_NAME" \
    "serving=$SERVING_CONFIG" "arm=$ARM" "status=serving"

# --- cleanup on any exit ------------------------------------------------------
# Armed BEFORE remote_serve start: that call blocks up to READY_TIMEOUT (~20min)
# waiting for /v1/models, and an interrupt during it must still tear the detached
# remote server down — otherwise it leaks on the GPU (the bug that left xu/h200
# servers running on Ctrl-C while ccds/nscc, which arm their traps first, did not).
TUNNEL_PID=""; CLIENT_PID=""; CLEANED=0
cleanup() {
    [[ "$CLEANED" == 1 ]] && return 0; CLEANED=1   # EXIT + INT/TERM can both fire
    echo "[bm] cleanup: remote_serve stop ($REMOTE:$PORT) and tunnel ..."
    [[ -n "$CLIENT_PID" ]] && kill -TERM "$CLIENT_PID" 2>/dev/null || true
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    REMOTE="$REMOTE" REMOTE_KIT_ROOT="$REMOTE_KIT_ROOT" PORT="$PORT" \
        bash "$SCRIPT_DIR/remote_serve.sh" stop >/dev/null 2>&1 || true
    so_write_alloc "run_id=$RUN_ID" "port=$PORT" "status=done" "rc=${RC:-aborted}"
}
trap cleanup EXIT INT TERM

# --- start the server on the host (detached; waits until /v1/models is up) -----
echo "[bm] remote_serve start on $REMOTE (CVD=$CVD TP=$GPUS) ..."
# GPU_MEM_UTIL (if the orchestrator set it, for packing) is already exported in
# this process's environment; remote_serve.sh forwards it via its PASSTHROUGH.
# MODEL_CONFIG/SERVING_CONFIG must be repo-relative — serve_vllm.sh resolves them
# against the GPU host's OWN checkout ($REMOTE_KIT_ROOT), not this node's.
REMOTE="$REMOTE" REMOTE_KIT_ROOT="$REMOTE_KIT_ROOT" PORT="$PORT" READY_TIMEOUT="$READY_TIMEOUT" \
    VLLM_VENV="$VLLM_VENV" TP="$GPUS" CUDA_VISIBLE_DEVICES="$CVD" \
    MODEL="$MODEL_NAME" MODEL_CONFIG="$MODEL_TEMPLATE" SERVING_CONFIG="$SERVING_CONFIG" \
    bash "$SCRIPT_DIR/remote_serve.sh" start
SERVE_RC=$?
if (( SERVE_RC == 75 )); then
    # Readiness TIMEOUT (slow host) — reallocate to another GPU. The cleanup trap
    # (armed above) stops the still-loading server on exit, so it doesn't leak.
    echo "[bm] reallocate: server not ready on $REMOTE within ${READY_TIMEOUT}s — releasing this GPU for another lane." >&2
    exit 1
elif (( SERVE_RC != 0 )); then
    # Crashed during init (e.g. OOM, bad config). Do NOT reallocate — it would
    # recur on every GPU and stampede; hard-fail so it surfaces. (Often the card
    # is fouled by a leaked server — clear those first.)
    die "remote_serve start failed on $REMOTE (server crashed during init, rc=$SERVE_RC)."
fi

# --- endpoint wiring: tunnel (proxied hosts) or direct route ------------------
if [[ "${DIRECT_ROUTE:-0}" == "1" ]]; then
    # Resolve the routable address the CLIENT must hit: the ssh HostName (works for
    # both an alias like `h200` and a user@ip), falling back to stripping user@.
    if [[ -z "${ENDPOINT_HOST:-}" ]]; then
        ENDPOINT_HOST="$(ssh -G "$REMOTE" 2>/dev/null | awk '/^hostname /{print $2; exit}')"
        ENDPOINT_HOST="${ENDPOINT_HOST:-${REMOTE#*@}}"
    fi
    echo "[bm] direct route — client endpoint $ENDPOINT_HOST:$PORT (no tunnel)"
else
    ENDPOINT_HOST="127.0.0.1"
    echo "[bm] tunnel: ntu_cpu -> $REMOTE:$PORT"
    FORWARD_HOST="127.0.0.1" REMOTE="$REMOTE" LOCAL_PORT="$PORT" REMOTE_PORT="$PORT" \
        bash "$SCRIPT_DIR/tunnel.sh" &
    TUNNEL_PID=$!
    sleep 2
fi

echo "[bm] starting run_tb_client.sh ..."
# Background the client + wait: a TERM/INT (budget kill or Ctrl-C) then interrupts
# `wait` at once so the trap runs remote_serve stop. A foreground client would hold
# the trap until it returned — and the benchmark harness ignores the signal — so
# the SIGKILL backstop hits before the trap, leaking the remote vllm on the GPU.
ENDPOINT_HOST="$ENDPOINT_HOST" PORT="$PORT" \
    MODEL_CONFIG="$CLIENT_CFG" SERVING_CONFIG="$KIT_ROOT/profiles/serving/vllm/existing_endpoint.yaml" \
    bash "$KIT_ROOT/scripts/execution_client/run_tb_client.sh" &
CLIENT_PID=$!
wait "$CLIENT_PID"; RC=$?
echo "[bm] run_tb_client exit rc=$RC"
exit "$RC"
