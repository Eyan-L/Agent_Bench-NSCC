#!/usr/bin/env bash
# scripts/serve_only/run_nscc.sh — ONE-command decoupled run on nscc from ntu_cpu.
# Submits a batch PBS job that serves vLLM on a GPU node,
# auto-discovers the exec node, runs the Terminal-Bench client over a 2-hop SSH
# tunnel, and qdel's the job on exit.
#
# Lifecycle: qsub serve_nscc.pbs -> poll qstat until R -> read exec node from the
# job's .node file -> tunnel.sh (ntu_cpu -> nscc login -> compute:8000) ->
# run_tb_client.sh -> trap: kill tunnel + qdel.
#
# Prerequisite: ntu_cpu must be able to `ssh nscc` (a Host entry + your SSH key).
# See scripts/serve_only/README.md.
#
# Env (override K=V on the command line):
#   REMOTE           SSH alias of the nscc LOGIN node. Default: nscc
#   REMOTE_KIT_ROOT  Agent_Bench checkout on nscc. Default: ~/Agent_Bench
#   PROJECT          PBS project/account to charge. REQUIRED (set in .env or on
#                    the command line; see .env.example).
#   NGPUS            GPUs to request (TP). Default: 2 (40GB A100 -> 2x40GB).
#                    Set 1 for an 80GB / H100 / H200 node.
#   WALLTIME         PBS walltime. Default: 02:00:00 (set e.g. 00:30:00 for a
#                    queue with a short cap, like hpcc's gpu_free).
#   QUEUE            PBS queue (`-q`). Default: unset -> the .pbs script's `normal`.
#                    Single-account hosts may need one (hpcc: gpu_free).
#   NCPUS            ncpus in the select chunk. Default: unset (omit). Some queues
#                    require it (hpcc: 4).
#
# Despite the name this also drives a SINGLE-ACCOUNT PBS host already in
# ~/.ssh/config (e.g. hpcc, an H200 cluster): set REMOTE=hpcc plus PROJECT/QUEUE/
# NCPUS/NGPUS/WALLTIME. SSH_CONFIG stays unset so so_ssh uses plain `ssh hpcc`;
# logs + the placement manifest are tagged with the cluster (REMOTE's first token).
#   NSCC_VLLM_VENV   vLLM venv on nscc. Default: <remote $HOME>/vllm/.venv
#   MODEL_TEMPLATE   Committed FP8 model config (params; name overridden).
#                    Default: profiles/models/qwen36_35b_a3b_fp8.json
#   MODEL_NAME       Served id + weights path. Default: <remote $HOME>/scratch/models/Qwen3.6-35B-A3B-FP8
#   READY_TIMEOUT    Seconds to wait for the job to enter R. Default: 1800
#   APP HARNESS_CONFIG ARM ARM_ENV SKIP_PULL
#                    Passed through to run_tb_client.sh.
set -uo pipefail

# Detach stdin from the terminal so this is safe to run backgrounded (`&`) for
# parallel fan-out: the many `ssh` calls below would otherwise read the tty,
# earn SIGTTIN, and stop the whole job ([N]+ Stopped). Nothing here reads stdin;
# ssh heredocs set their own. Redirect output to a log if backgrounding many.
exec </dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/serve_only/_serve_only_common.sh
source "$SCRIPT_DIR/_serve_only_common.sh"

REMOTE="${REMOTE:-nscc}"
REMOTE_KIT_ROOT="${REMOTE_KIT_ROOT:-\$HOME/Agent_Bench}"
PROJECT="${PROJECT:-}"   # REQUIRED — set in .env or on the command line (see .env.example)
NGPUS="${NGPUS:-2}"
WALLTIME="${WALLTIME:-02:00:00}"
# Optional PBS knobs for single-account hosts (e.g. hpcc): a named queue and a
# CPU count in the select statement. Empty = omit (nscc's default `normal` queue
# and no ncpus request, as before). CLUSTER/TAG label the logs + manifest by the
# cluster (REMOTE is "nscc-<user>" when orchestrated, "hpcc" for the H200 host).
QUEUE="${QUEUE:-}"
NCPUS="${NCPUS:-}"
CLUSTER="${REMOTE%%-*}"
TAG="[$CLUSTER]"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-profiles/models/qwen36_35b_a3b_fp8.json}"
# Default to an auto-allocated free local port so multiple runs can fan out in
# parallel (the tunnel binds 0.0.0.0:$PORT — a fixed 8000 collides). The same
# number is the served port on the compute node, so it's unique end-to-end.
PORT="${PORT:-$(so_pick_free_port)}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
# Identity for the placement manifest; the fan-out orchestrator passes a stable
# per-cell RUN_ID (so its serving/exit lines merge), standalone falls back later.
RUN_ID="${RUN_ID:-}"
ARM="${ARM:-default}"   # generic experiment-variant label (orchestrator sets it)
echo "$TAG using port $PORT (tunnel + compute-node serve + client endpoint)"

die() { echo "$TAG ERROR: $*" >&2; exit 1; }

# All ssh goes through so_ssh: when the orchestrator set SSH_CONFIG (multi-account
# nscc, one Host nscc-<user> block per ~/.ssh/nscc_<user> key) the same REMOTE
# alias resolves under that account; unset = plain ssh / ~/.ssh/config.
so_ssh -G "$REMOTE" >/dev/null 2>&1 || die "'$REMOTE' is not resolvable by ssh on this node (set up the nscc Host entry + your SSH key; see README)."
REMOTE_KIT_ROOT="$(so_ssh "$REMOTE" "cd ${REMOTE_KIT_ROOT} 2>/dev/null && pwd")" \
    || die "REMOTE_KIT_ROOT not found on ${REMOTE}."
RUN_ID="${RUN_ID:-ab-nscc-$(date +%s)-$$}"

REMOTE_HOME="$(so_remote_home "$REMOTE")"; [[ -n "$REMOTE_HOME" ]] || die "could not resolve \$HOME on $REMOTE."
MODEL_NAME="${MODEL_NAME:-$REMOTE_HOME/scratch/models/Qwen3.6-35B-A3B-FP8}"
NSCC_VLLM_VENV="${NSCC_VLLM_VENV:-$REMOTE_HOME/vllm/.venv}"

# --- write the client model-config (name must match the served id) --------
# Keyed by model (not port): parallel runs of the SAME model reuse one file;
# different models get their own. so_gen_client_cfg writes it atomically, so a
# concurrent preflight json.load never sees a half-written file.
CLIENT_CFG="$KIT_ROOT/data/serve_only/model_nscc_$(basename "$MODEL_NAME").json"
so_gen_client_cfg "$KIT_ROOT/$MODEL_TEMPLATE" "$MODEL_NAME" "$CLIENT_CFG"

# --- submit the PBS job ---------------------------------------------------
[[ -n "$PROJECT" ]] || die "PROJECT is required — set your PBS project/account in .env or on the command line (see .env.example)."
# Build the select chunk (optionally with ncpus, for queues that require it like
# hpcc's gpu_free) and an optional -q queue arg (empty = the .pbs script's default
# `normal`, as nscc uses). Both expand to nothing when their env is unset.
SELECT="select=1${NCPUS:+:ncpus=$NCPUS}:ngpus=$NGPUS"
QUEUE_ARG="${QUEUE:+-q $QUEUE}"
echo "$TAG qsub serve_nscc.pbs (project=$PROJECT queue=${QUEUE:-<default>} $SELECT walltime=$WALLTIME)"
# Fractional GPU memory so several servers can share one card (orchestrator sets
# it; unset = whole-card default). Appended to qsub -v only when set; serve_vllm.sh
# picks it up from the job environment.
GPU_UTIL_KV="${GPU_MEM_UTIL:+,GPU_MEM_UTIL=$GPU_MEM_UTIL}"
JOBID="$(so_ssh "$REMOTE" "cd '$REMOTE_KIT_ROOT' && qsub \
    -P '$PROJECT' $QUEUE_ARG \
    -l $SELECT -l walltime=$WALLTIME \
    -v NGPUS=$NGPUS,NSCC_VLLM_VENV=$NSCC_VLLM_VENV,MODEL=$MODEL_NAME,PORT=$PORT$GPU_UTIL_KV \
    scripts/serve_only/serve_nscc.pbs")" || die "qsub failed."
JOBID="$(echo "$JOBID" | tr -d '[:space:]')"
[[ -n "$JOBID" ]] || die "qsub returned no job id."
echo "$TAG submitted job: $JOBID"

TUNNEL_PID=""
CLIENT_PID=""
_CLEANED=""
cleanup() {
    [[ -n "$_CLEANED" ]] && return    # run once; ignore repeated ^C
    _CLEANED=1
    trap - EXIT INT TERM              # disarm so re-signals can't re-enter
    echo "$TAG cleanup: qdel $JOBID and tunnel ..."
    [[ -n "$CLIENT_PID" ]] && kill -TERM "$CLIENT_PID" 2>/dev/null || true
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    so_ssh "$REMOTE" "qdel '$JOBID' 2>/dev/null || true" || true
    so_write_alloc "run_id=$RUN_ID" "port=$PORT" "status=done" "rc=${RC:-aborted}"
}
trap cleanup EXIT INT TERM

# --- wait for the job to start, then learn the exec node ------------------
echo "$TAG waiting up to ${READY_TIMEOUT}s for job $JOBID to enter R ..."
NODE=""
waited=0
while (( waited < READY_TIMEOUT )); do
    state="$(so_ssh "$REMOTE" "qstat -f '$JOBID' 2>/dev/null" | awk -F= '/job_state/{gsub(/ /,"",$2);print $2;exit}')"
    if [[ "$state" == "R" ]]; then
        NODE="$(so_ssh "$REMOTE" "cat '$REMOTE_KIT_ROOT/logs/serve_only/$JOBID.node' 2>/dev/null" | tr -d '[:space:]')"
        if [[ -z "$NODE" ]]; then
            # Fallback: parse exec_host (e.g. "node1234/0*2") from qstat.
            NODE="$(so_ssh "$REMOTE" "qstat -f '$JOBID' 2>/dev/null" \
                | awk -F= '/exec_host/{print $2;exit}' | tr -d ' ' | cut -d/ -f1)"
        fi
        [[ -n "$NODE" ]] && break
    fi
    if [[ "$state" == "F" || "$state" == "E" ]]; then
        die "job $JOBID ended early (state=$state) — check the PBS .o log on $REMOTE."
    fi
    sleep 10; (( waited += 10 ))
done
[[ -n "$NODE" ]] || die "timed out waiting for job $JOBID to start (state=${state:-?})."
echo "$TAG job running on exec node: $NODE"

# Read the GPU type + size the PBS job published ("NAME TOTAL_MIB"), then record
# the placement in the fan-out ledger.
GPU_LINE="$(so_ssh "$REMOTE" "cat '$REMOTE_KIT_ROOT/logs/serve_only/$JOBID.gpu' 2>/dev/null" | head -1)"
GPU_NAME="$(awk '{print $1}' <<<"$GPU_LINE")"; GPU_NAME="${GPU_NAME:-unknown}"
GPU_MEM_MIB="$(awk '{print $2}' <<<"$GPU_LINE")"
echo "$TAG gpu=$GPU_NAME (${GPU_MEM_MIB:-?}MiB)"
so_write_alloc "run_id=$RUN_ID" "cluster=$CLUSTER" "account=$(so_ssh "$REMOTE" 'printf %s "$USER"' 2>/dev/null)" \
    "node=$NODE" "gpu_name=$GPU_NAME" "gpu_mem_mib=${GPU_MEM_MIB:-}" "gpus=$NGPUS" \
    "gpu_mem_util=${GPU_MEM_UTIL:-}" "port=$PORT" "jobid=$JOBID" "model=$MODEL_NAME" \
    "serving=profiles/serving/vllm/decoupled.yaml" "arm=$ARM" "status=serving"

# --- 2-hop tunnel + client ------------------------------------------------
echo "$TAG tunnel: ntu_cpu -> $REMOTE -> $NODE:$PORT"
FORWARD_HOST="$NODE" REMOTE="$REMOTE" LOCAL_PORT="$PORT" REMOTE_PORT="$PORT" \
    bash "$SCRIPT_DIR/tunnel.sh" &
TUNNEL_PID=$!
sleep 2

echo "$TAG starting run_tb_client.sh (it waits up to ${ENDPOINT_WAIT_SECS:-1200}s for /v1/models)..."
# Background the client + wait: a TERM/INT (budget kill or Ctrl-C) then interrupts
# `wait` at once so the trap runs qdel + tunnel teardown. A foreground client would
# hold the trap until it returned — and the benchmark harness ignores the signal —
# stranding the PBS job (and its GPU) until the walltime cap.
# CLIENT_HOST is the host the client puts in the endpoint URL — and thus the HTTP
# Host header. Default 127.0.0.1. hpcc sits behind a SonicWall content filter on
# the login<->compute path that BLOCKS forwarded HTTP whose Host it can't rate
# (Host: 127.0.0.1 -> 403 "Web Page Blocked"); `localhost` is rated OK -> 200. The
# tunnel still binds 0.0.0.0:$PORT, so localhost resolves to it on ntu_cpu. Set
# CLIENT_HOST=localhost for such hosts (the fanout hpcc target does).
ENDPOINT_HOST="${CLIENT_HOST:-127.0.0.1}" PORT="$PORT" \
    MODEL_CONFIG="$CLIENT_CFG" SERVING_CONFIG="$KIT_ROOT/profiles/serving/vllm/existing_endpoint.yaml" \
    bash "$KIT_ROOT/scripts/execution_client/run_tb_client.sh" &
CLIENT_PID=$!
wait "$CLIENT_PID"; RC=$?
echo "$TAG run_tb_client exit rc=$RC"
exit "$RC"
