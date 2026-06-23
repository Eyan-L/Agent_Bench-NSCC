#!/usr/bin/env bash
# scripts/serve_only/run_ccds.sh — ONE-command decoupled run on ccds-ai from
# the client node. Allocates a GPU node sized to Qwen3.6-35B-A3B-FP8 (2x40GB or
# 1x80GB), serves vLLM on it, runs the Terminal-Bench client over a 2-hop SSH
# tunnel, and releases the allocation on exit.
#
# Why srun (not sbatch): ccds-ai's scheduler hands out partially-used cards via
# sbatch. We pick genuinely-free cards from the `gpu-status` helper's own logic
# (idle/mix nodes -> srun nvidia-smi VRAM probe, free = <=10 MiB used) and PIN
# CUDA_VISIBLE_DEVICES to those physical ids — we do NOT let Slurm choose them.
#
# Lifecycle: gpu-status discover+size -> detached `srun ... serve_vllm.sh` ->
# tunnel.sh (client -> ccds-ai login -> compute:8000) -> run_tb_client.sh ->
# trap: kill tunnel + scancel.
#
# Env (override K=V on the command line):
#   REMOTE           SSH alias of the ccds-ai LOGIN node. Default: ccds-ai
#   REMOTE_KIT_ROOT  Agent_Bench checkout on ccds-ai. Default: ~/Agent_Bench
#   VLLM_VENV        vLLM venv on ccds-ai.
#                    Default: <remote $HOME>/vllm/.venv
#   MODEL_TEMPLATE   Committed FP8 model config (params; name is overridden).
#                    Default: profiles/models/qwen36_35b_a3b_fp8.json
#   MODEL_NAME       Served id + weights path. Default: <remote $HOME>/scratch/models/Qwen3.6-35B-A3B-FP8
#   SERVING_CONFIG   Default: profiles/serving/vllm/decoupled.yaml
#   GPUS             Force TP / card count (skip auto-size). Default: auto (1 or 2).
#   NODE PARTITION CUDA_VISIBLE_DEVICES  Force the allocation (skip gpu-status).
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

REMOTE="${REMOTE:-ccds-ai}"
REMOTE_KIT_ROOT="${REMOTE_KIT_ROOT:-\$HOME/Agent_Bench}"
VLLM_VENV="${VLLM_VENV:-}"   # default derived from the remote $HOME once resolved
MODEL_TEMPLATE="${MODEL_TEMPLATE:-profiles/models/qwen36_35b_a3b_fp8.json}"
SERVING_CONFIG="${SERVING_CONFIG:-profiles/serving/vllm/decoupled.yaml}"
# Default to an auto-allocated free local port so multiple runs can fan out in
# parallel (the tunnel binds 0.0.0.0:$PORT — a fixed 8000 collides). The same
# number is the served port on the compute node, so it's unique end-to-end.
PORT="${PORT:-$(so_pick_free_port)}"
# Slurm wall-clock cap on the serve job: a backstop so an ungracefully-killed
# wrapper (SIGKILL / terminal hangup, where the scancel trap never runs) can't
# strand the allocation forever — Slurm auto-reaps it after this. Override for
# longer runs (e.g. SRUN_TIME=06:00:00).
SRUN_TIME="${SRUN_TIME:-02:00:00}"
PARTS="${CCDS_PARTITIONS:-HPCAIq,NH100q,PH100q,NA100q,PA100q}"
BIG_VRAM_MIB="${BIG_VRAM_MIB:-70000}"   # >= this total = an 80GB card (1 suffices)
JOBNAME="ab-serve-$(date +%s)-$$"
# Identity for the placement manifest (the fan-out orchestrator passes a stable
# per-cell RUN_ID so its serving/exit lines can be merged; standalone = JOBNAME).
RUN_ID="${RUN_ID:-$JOBNAME}"
# Fractional GPU memory so several servers can share one card (orchestrator sets
# it; unset = serve_vllm/launch_plan default = whole card). Forwarded to the srun
# env below only when set.
GPU_UTIL_ENV="${GPU_MEM_UTIL:+GPU_MEM_UTIL=$GPU_MEM_UTIL}"
# Generic experiment-variant label, recorded in the placement manifest and passed
# through to the client. The fan-out orchestrator sets it per cell; standalone
# runs default to "default". Any per-arm env the orchestrator wants applied is
# already in this process's environment (see fanout.py arm_env overlay).
ARM="${ARM:-default}"
# Shared "NODE IDX" claims file (set by the fan-out orchestrator) used to keep
# concurrent ccds lanes off the same cards — see so_ccds_* below. Unset for a
# standalone run, where there is no sibling to collide with.
CCDS_CLAIMS="${CCDS_CLAIMS:-}"

die() { echo "[ccds] ERROR: $*" >&2; exit 1; }

# --- disjoint card claiming across concurrent fan-out lanes ----------------
# The free-GPU probe is NOT atomic: a sibling lane that just launched its serve
# hasn't grabbed VRAM yet, so its cards still read "free" — two lanes then pick
# the SAME node+cards and collide. We serialize discover->pick->record under a
# flock and exclude cards a live sibling already claimed, so picks stay disjoint.
# All a no-op when CCDS_CLAIMS is unset (standalone run).
CLAIMED=0
so_ccds_claim_lock() {
    [[ -n "$CCDS_CLAIMS" ]] || return 0
    mkdir -p "$(dirname "$CCDS_CLAIMS")"
    exec 9>"$CCDS_CLAIMS.lock"
    flock 9
}
so_ccds_claim_unlock() { [[ -n "$CCDS_CLAIMS" ]] && flock -u 9 2>/dev/null || true; }
so_ccds_filter_claimed() {
    # stdin->stdout: drop FREE lines whose "NODE($2) IDX($3)" a sibling holds.
    if [[ -z "$CCDS_CLAIMS" || ! -s "$CCDS_CLAIMS" ]]; then cat; return; fi
    awk -v cf="$CCDS_CLAIMS" 'BEGIN{while((getline l<cf)>0){split(l,a," ");c[a[1]" "a[2]]=1}}
        { if(!(($2" "$3) in c)) print }'
}
so_ccds_record_claim() {
    # Append our picked cards, mark claimed, then release the lock.
    [[ -n "$CCDS_CLAIMS" ]] || return 0
    local idx; for idx in ${CVD//,/ }; do echo "$NODE $idx" >> "$CCDS_CLAIMS"; done
    CLAIMED=1
    so_ccds_claim_unlock
}
so_ccds_release_claim() {
    # Drop our cards from the shared file on exit so a requeue can reuse them.
    [[ "$CLAIMED" == 1 && -n "$CCDS_CLAIMS" ]] || return 0
    exec 9>"$CCDS_CLAIMS.lock"; flock 9
    local tmp mine idx; tmp="$(mktemp)"; mine="$(mktemp)"
    for idx in ${CVD//,/ }; do echo "$NODE $idx" >> "$mine"; done
    grep -vxF -f "$mine" "$CCDS_CLAIMS" > "$tmp" 2>/dev/null || true
    mv "$tmp" "$CCDS_CLAIMS"; rm -f "$mine"
    CLAIMED=0
    flock -u 9 2>/dev/null || true
}

echo "[ccds] using port $PORT (tunnel + compute-node serve + client endpoint)"

ssh -G "$REMOTE" >/dev/null 2>&1 || die "'$REMOTE' is not resolvable by ssh on this node."
REMOTE_KIT_ROOT="$(ssh "$REMOTE" "cd ${REMOTE_KIT_ROOT} 2>/dev/null && pwd")" \
    || die "REMOTE_KIT_ROOT not found on ${REMOTE}."

# --- resolve the served model id (== weights path on ccds-ai) -------------
REMOTE_HOME="$(so_remote_home "$REMOTE")"; [[ -n "$REMOTE_HOME" ]] || die "could not resolve \$HOME on $REMOTE."
MODEL_NAME="${MODEL_NAME:-$REMOTE_HOME/scratch/models/Qwen3.6-35B-A3B-FP8}"
VLLM_VENV="${VLLM_VENV:-$REMOTE_HOME/vllm/.venv}"

# --- discover + size free GPUs (reuses gpu-status's own free-GPU logic) ----
# Emit "PART NODE IDX NAME TOTAL_MIB" per completely-free card across idle/mix
# nodes. NAME is the GPU model (e.g. NVIDIA_A100-SXM4-40GB) with spaces squashed
# to underscores so it stays a single whitespace-delimited field for the pick awk
# and the placement manifest (records the GPU type, not just its size).
discover_free() {
    ssh -o BatchMode=yes "$REMOTE" 'bash -s' <<REMOTE_EOF
# NB: no 'set -u' — sourcing modules.sh references unset vars and would abort.
set -o pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load slurm/slurm/24.11 2>/dev/null || true
PARTS="$PARTS"
PART_REGEX=\$(echo "\$PARTS" | tr ',' '|')
NODES=\$(sinfo -p "\$PARTS" -h -t idle,mix -o "%N" 2>/dev/null | sort -u)
[[ -z "\$NODES" ]] && exit 0
for node in \$(scontrol show hostnames \$NODES 2>/dev/null); do
    PART=\$(scontrol show node "\$node" 2>/dev/null | grep -i "Partitions=" | head -n1 \
        | awk -F'=' '{print \$2}' | awk '{print \$1}' | tr ',' '\n' | grep -E "\$PART_REGEX" | head -n1)
    [[ -z "\$PART" ]] && continue
    OUT=\$(srun -p "\$PARTS" -w "\$node" -n1 -c1 --mem=100M -t1 --quiet \
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null) || continue
    while IFS=, read -r idx name used total; do
        idx=\$(echo "\$idx" | tr -d ' '); used=\$(echo "\$used" | tr -d ' '); total=\$(echo "\$total" | tr -d ' ')
        name=\$(echo "\$name" | sed 's/^ *//; s/ *\$//; s/ /_/g')
        [[ -z "\$idx" ]] && continue
        if [[ "\$used" -le 10 ]]; then echo "\$PART \$node \$idx \$name \$total"; fi
    done <<< "\$OUT"
done
REMOTE_EOF
}

if [[ -n "${NODE:-}" && -n "${PARTITION:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    PARTITION="$PARTITION"; NODE="$NODE"; CVD="$CUDA_VISIBLE_DEVICES"
    GPUS="${GPUS:-$(awk -F, '{print NF}' <<<"$CVD")}"
    GPU_NAME="${GPU_NAME:-forced}"; GPU_MEM_MIB="${GPU_MEM_MIB:-}"
    echo "[ccds] using forced allocation: partition=$PARTITION node=$NODE CUDA_VISIBLE_DEVICES=$CVD GPUS=$GPUS"
else
    echo "[ccds] querying free GPUs on $REMOTE (gpu-status logic; this srun-probes idle/mix nodes)..."
    so_ccds_claim_lock                       # serialize discover+pick across lanes
    FREE="$(discover_free)" || { so_ccds_claim_unlock; die "free-GPU discovery failed."; }
    FREE="$(so_ccds_filter_claimed <<<"$FREE")"   # exclude cards a live sibling holds
    [[ -n "$FREE" ]] || { so_ccds_claim_unlock; die "no completely-free GPUs found on $PARTS (try again later or force NODE/PARTITION/CUDA_VISIBLE_DEVICES)."; }
    # Pass 1: a node with a free >=80GB card -> 1 card. Pass 2: a node with >=2 free 40GB cards.
    # Fields per line: PART($1) NODE($2) IDX($3) NAME($4) TOTAL_MIB($5). We also
    # carry the picked card's NAME + TOTAL_MIB out for the placement manifest.
    read -r PARTITION NODE CVD GPUS GPU_NAME GPU_MEM_MIB < <(
        awk -v big="$BIG_VRAM_MIB" -v want="${GPUS:-0}" '
        { part[$2]=$1; name[$2]=$4; total[$2]=$5; ids[$2]=ids[$2] $3 " "; cnt[$2]++; if ($5>=big) big1[$2]=$3 }
        END {
            # forced GPUS overrides sizing
            if (want==1) { for (n in cnt) if (cnt[n]>=1) { split(ids[n],a," "); print part[n], n, a[1], 1, name[n], total[n]; exit } }
            else if (want>=2) { for (n in cnt) if (cnt[n]>=want) { s=""; for(i=1;i<=want;i++){split(ids[n],a," "); s=(i==1?a[i]:s","a[i])} print part[n], n, s, want, name[n], total[n]; exit } }
            else {
                for (n in big1) { print part[n], n, big1[n], 1, name[n], total[n]; exit }   # 1x80GB
                for (n in cnt) if (cnt[n]>=2) { split(ids[n],a," "); print part[n], n, a[1]","a[2], 2, name[n], total[n]; exit }  # 2x40GB
            }
        }' <<<"$FREE"
    )
    [[ -n "${NODE:-}" ]] || { so_ccds_claim_unlock; die "no node satisfies the model's GPU need (1x>=${BIG_VRAM_MIB}MiB or 2x free cards)."; }
    so_ccds_record_claim                     # reserve our cards, then release the lock
    echo "[ccds] picked partition=$PARTITION node=$NODE CUDA_VISIBLE_DEVICES=$CVD GPUS=$GPUS gpu=$GPU_NAME (${GPU_MEM_MIB}MiB)"
    echo "[ccds] free cards seen:"; echo "$FREE" | sed 's/^/[ccds]   /'
fi

# --- size GPU memory to the model footprint on the picked card ------------
# A flat util (serve's 0.9 default) over-requests on big cards: 0.9 of an 80GB
# card is ~71GB, but the 35B FP8 model needs ~38GB — so the serve dies the moment
# another tenant holds a few GB (ValueError: free memory < desired utilization).
# Size util to MODEL_FOOTPRINT_MIB / card-total (capped): ~0.92 on a 40GB card,
# ~0.47 on an 80GB one. Skipped if the caller pinned GPU_MEM_UTIL explicitly.
MODEL_FOOTPRINT_MIB="${MODEL_FOOTPRINT_MIB:-38000}"
GPU_UTIL_CAP="${GPU_UTIL_CAP:-0.92}"
if [[ -z "${GPU_MEM_UTIL:-}" && "${GPU_MEM_MIB:-}" =~ ^[0-9]+$ && "$GPU_MEM_MIB" -gt 0 ]]; then
    GPU_MEM_UTIL="$(awk -v f="$MODEL_FOOTPRINT_MIB" -v t="$GPU_MEM_MIB" -v cap="$GPU_UTIL_CAP" \
        'BEGIN{ u=f/t; if(u>cap)u=cap; printf "%.2f", u }')"
    GPU_UTIL_ENV="GPU_MEM_UTIL=$GPU_MEM_UTIL"
    echo "[ccds] adaptive GPU_MEM_UTIL=$GPU_MEM_UTIL (model ~${MODEL_FOOTPRINT_MIB}MiB / ${GPU_MEM_MIB}MiB card)"
fi

# --- write the client model-config (name must match the served id) --------
# Keyed by model (not port): parallel runs of the SAME model reuse one file;
# different models get their own. so_gen_client_cfg writes it atomically, so a
# concurrent preflight json.load never sees a half-written file.
CLIENT_CFG="$KIT_ROOT/data/serve_only/model_ccds_$(basename "$MODEL_NAME").json"
so_gen_client_cfg "$KIT_ROOT/$MODEL_TEMPLATE" "$MODEL_NAME" "$CLIENT_CFG"

# --- launch the server detached on the picked node ------------------------
echo "[ccds] launching srun serve (job=$JOBNAME) on $NODE ..."
SRUN_LOG="logs/serve_only/srun_${JOBNAME}.log"
ssh "$REMOTE" "cd '$REMOTE_KIT_ROOT' && mkdir -p logs/serve_only && \
    source /etc/profile.d/modules.sh 2>/dev/null; module load slurm/slurm/24.11 2>/dev/null; \
    nohup srun -p '$PARTITION' -w '$NODE' --gres=gpu:$GPUS --time='$SRUN_TIME' --job-name='$JOBNAME' \
        env CUDA_VISIBLE_DEVICES='$CVD' VLLM_VENV='$VLLM_VENV' MODEL='$MODEL_NAME' \
            MODEL_CONFIG='$MODEL_TEMPLATE' SERVING_CONFIG='$SERVING_CONFIG' TP='$GPUS' PORT='$PORT' \
            LOG_DIR='logs/serve_only/p$PORT' $GPU_UTIL_ENV \
            bash scripts/serve_only/serve_vllm.sh \
        > '$SRUN_LOG' 2>&1 < /dev/null & echo started" \
    || die "remote srun launch failed."

# Record the placement now that node + GPU + port are known (the fan-out ledger).
so_write_alloc "run_id=$RUN_ID" "cluster=ccds" "node=$NODE" "partition=$PARTITION" \
    "gpu_name=$GPU_NAME" "gpu_mem_mib=$GPU_MEM_MIB" "gpus=$GPUS" "cuda_visible_devices=$CVD" \
    "gpu_mem_util=${GPU_MEM_UTIL:-}" "port=$PORT" "jobid=$JOBNAME" "model=$MODEL_NAME" \
    "serving=$SERVING_CONFIG" "arm=$ARM" "status=serving"

# --- cleanup on any exit --------------------------------------------------
TUNNEL_PID=""; CLIENT_PID=""; CLEANED=0
cleanup() {
    [[ "$CLEANED" == 1 ]] && return 0; CLEANED=1   # EXIT + INT/TERM can both fire
    echo "[ccds] cleanup: releasing allocation '$JOBNAME' and tunnel ..."
    so_ccds_release_claim   # free our cards in the shared ledger for a requeue
    [[ -n "$CLIENT_PID" ]] && kill -TERM "$CLIENT_PID" 2>/dev/null || true
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    # Load the Slurm module first: a non-interactive ssh has no `scancel` on PATH,
    # so without this the cancel silently no-ops and the allocation leaks (the
    # launch path above already loads it; the cleanup must match).
    ssh "$REMOTE" "source /etc/profile.d/modules.sh 2>/dev/null; module load slurm/slurm/24.11 2>/dev/null; scancel -n '$JOBNAME' 2>/dev/null || true" || true
    so_write_alloc "run_id=$RUN_ID" "port=$PORT" "status=done" "rc=${RC:-aborted}"
}
trap cleanup EXIT INT TERM

# --- confirm the serve job actually got resources -------------------------
# The free-VRAM probe finds idle cards, but Slurm may still refuse to schedule
# there (gres reserved by other tenants) — the job then sits PENDING and the
# client would burn its full 20-min wait before failing. Poll squeue: if the job
# isn't RUNNING within CCDS_ALLOC_TIMEOUT, bail fast with a marker the fan-out
# orchestrator treats as "GPU unavailable" -> requeue onto another node.
CCDS_ALLOC_TIMEOUT="${CCDS_ALLOC_TIMEOUT:-90}"
confirm_allocation() {
    local waited=0 state
    while (( waited < CCDS_ALLOC_TIMEOUT )); do
        state="$(ssh -o BatchMode=yes "$REMOTE" \
            "source /etc/profile.d/modules.sh 2>/dev/null; module load slurm/slurm/24.11 2>/dev/null; \
             squeue -h -n '$JOBNAME' -o '%T' 2>/dev/null | head -1")"
        [[ "$state" == RUNNING ]] && { echo "[ccds] serve job RUNNING on $NODE after ${waited}s."; return 0; }
        sleep 6; waited=$((waited + 6))
    done
    return 1
}
if ! confirm_allocation; then
    # "node not allocatable" is a REQUEUE_MARKER in orchestration/fanout/fanout.py.
    echo "[ccds] FATAL: serve job stayed PENDING >${CCDS_ALLOC_TIMEOUT}s on $NODE — node not allocatable (gres reserved by other tenants)."
    RC=1; exit 1
fi

# --- 2-hop tunnel + client ------------------------------------------------
echo "[ccds] tunnel: ntu_cpu -> $REMOTE -> $NODE:$PORT"
FORWARD_HOST="$NODE" REMOTE="$REMOTE" LOCAL_PORT="$PORT" REMOTE_PORT="$PORT" \
    bash "$SCRIPT_DIR/tunnel.sh" &
TUNNEL_PID=$!
sleep 2

echo "[ccds] starting run_tb_client.sh (it waits up to 10 min for /v1/models)..."
# Background the client + wait: a TERM/INT (budget kill or Ctrl-C) then interrupts
# `wait` at once so the trap runs scancel + tunnel teardown. A foreground client
# would hold the trap until it returned — and the benchmark harness ignores the
# signal — stranding the srun allocation (and its GPU) until Slurm's time cap.
ENDPOINT_HOST=127.0.0.1 PORT="$PORT" \
    MODEL_CONFIG="$CLIENT_CFG" SERVING_CONFIG="$KIT_ROOT/profiles/serving/vllm/existing_endpoint.yaml" \
    bash "$KIT_ROOT/scripts/execution_client/run_tb_client.sh" &
CLIENT_PID=$!
wait "$CLIENT_PID"; RC=$?
echo "[ccds] run_tb_client exit rc=$RC"
exit "$RC"
