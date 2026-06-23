#!/usr/bin/env bash
# scripts/serve_only/remote_serve.sh — drive the GPU host's serve_vllm.sh FROM
# ntu_cpu over SSH, so you never have to open a shell on the GPU host by hand.
#
#   start    launch serve_vllm.sh on the GPU host (detached; survives SSH drop),
#            then wait until /v1/models answers.
#   stop     graceful shutdown of the vllm process group (SIGTERM -> SIGKILL
#            escalation, via serve_vllm.sh's supervisor / PGID file).
#   restart  stop then start.
#   status   is the supervisor alive + is /v1/models answering?
#   logs     tail the remote vllm.log (Ctrl-C to detach).
#
# This pairs with tunnel.sh (data plane) — remote_serve.sh is the *control*
# plane. A full decoupled run from ntu_cpu becomes:
#   bash scripts/serve_only/remote_serve.sh start      # GPU host comes up
#   bash scripts/serve_only/tunnel.sh &                # forward :8000
#   MODEL_CONFIG=... HARNESS_CONFIG=... bash scripts/execution_client/run_tb_client.sh
#   bash scripts/serve_only/remote_serve.sh stop       # free the GPU
#
# Env:
#   REMOTE           SSH host alias of the GPU node. REQUIRED (set in .env or on
#                    the command line; see .env.example).
#   REMOTE_KIT_ROOT  Agent_Bench checkout path ON the GPU host. Default: ~/Agent_Bench
#   PORT             Port vLLM listens on at the GPU host. Default: 8000
#   READY_TIMEOUT    Seconds to wait for /v1/models on `start`. Default: 600
#
# Passed through to the remote serve_vllm.sh (set them on THIS command line; they
# are evaluated on the GPU host, so quote any value that references the GPU host's
# environment, e.g. MODEL='$HOME/scratch/models/Qwen3.6-35B-A3B'):
#   VLLM_VENV TP CUDA_VISIBLE_DEVICES MODEL MODEL_CONFIG SERVING_CONFIG SERVE_GRACE
#   BIND_HOST GPU_MEM_UTIL
#
# Packing multiple servers on ONE card: give each a distinct PORT and a fractional
# GPU_MEM_UTIL (e.g. two servers at 0.45 on an 80GB card). All supervisor state
# (pidfile, pgid, logs) is PORT-scoped under logs/serve_only/p<PORT>/, so the
# servers don't clobber each other and `stop` targets the right one.
#
# Example:
#   VLLM_VENV='$HOME/vllm/.venv' TP=2 CUDA_VISIBLE_DEVICES=0,1 \
#       MODEL='$HOME/scratch/models/Qwen3.6-35B-A3B' \
#       MODEL_CONFIG=profiles/models/qwen36_35b_a3b_a6000.json \
#       bash scripts/serve_only/remote_serve.sh start
set -uo pipefail

REMOTE="${REMOTE:-}"   # REQUIRED — SSH alias or user@ip (see .env.example)
REMOTE_KIT_ROOT="${REMOTE_KIT_ROOT:-\$HOME/Agent_Bench}"   # expanded on the GPU host
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"

# Remote paths (relative to REMOTE_KIT_ROOT, resolved on the GPU host). PORT-scoped
# so several servers can share one host/card without clobbering each other's
# pidfile/pgid/logs — serve_vllm.sh writes here too (we forward LOG_DIR below).
REMOTE_LOGDIR="logs/serve_only/p${PORT}"
SUP_PIDFILE="$REMOTE_LOGDIR/supervisor.pid"   # serve_vllm.sh supervisor PID
PGID_FILE="$REMOTE_LOGDIR/serve_vllm.pgid"    # vllm process-group id
SUP_OUT="$REMOTE_LOGDIR/supervisor.out"
VLLM_LOG="$REMOTE_LOGDIR/vllm.log"

# Env vars forwarded to the remote serve_vllm.sh, taken verbatim from this
# process's environment (so a `$HOME` in the value expands on the GPU host).
PASSTHROUGH=(VLLM_VENV TP CUDA_VISIBLE_DEVICES MODEL MODEL_CONFIG SERVING_CONFIG SERVE_GRACE BIND_HOST PORT GPU_MEM_UTIL)

CMD="${1:-}"; [[ $# -gt 0 ]] && shift || true

die() { echo "[remote_serve] ERROR: $*" >&2; exit 1; }

[[ -n "$CMD" ]] || die "usage: remote_serve.sh {start|stop|restart|status|logs}"
ssh -G "$REMOTE" >/dev/null 2>&1 || die "'$REMOTE' is not resolvable by ssh (add a Host entry; see tunnel.sh)."

# Resolve REMOTE_KIT_ROOT to an absolute path ON the GPU host once, so every
# later ssh can single-quote it safely (a literal `$HOME` inside single quotes
# would not expand remotely). Also validates the checkout exists.
REMOTE_KIT_ROOT="$(ssh "$REMOTE" "cd ${REMOTE_KIT_ROOT} 2>/dev/null && pwd")" \
    || die "REMOTE_KIT_ROOT not found on ${REMOTE} (set REMOTE_KIT_ROOT to the Agent_Bench checkout path there)."
[[ -n "$REMOTE_KIT_ROOT" ]] || die "could not resolve REMOTE_KIT_ROOT on ${REMOTE}."

# Build the `VAR=value ...` prefix for the remote launch from PASSTHROUGH.
_env_prefix() {
    local out="" name
    for name in "${PASSTHROUGH[@]}"; do
        if [[ -n "${!name:-}" ]]; then
            out+="$name=${!name} "
        fi
    done
    printf '%s' "$out"
}

remote_ready() {
    # Probe /v1/models from the GPU host itself (no tunnel needed for control).
    ssh "$REMOTE" "curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1"
}

do_start() {
    if remote_ready; then
        echo "[remote_serve] already serving on ${REMOTE}:${PORT} — nothing to do."
        return 0
    fi
    local env_prefix; env_prefix="$(_env_prefix)"
    echo "[remote_serve] starting serve_vllm.sh on ${REMOTE} (${REMOTE_KIT_ROOT})"
    echo "[remote_serve]   env: ${env_prefix:-<none — using serve_vllm.sh defaults>}"
    # nohup so it survives the SSH session closing; stdin from /dev/null so it
    # doesn't block; serve_vllm.sh's own `set -m`+trap own the clean shutdown and
    # write the PGID file we use for stop.
    ssh "$REMOTE" "bash -lc '
        set -e
        cd \"${REMOTE_KIT_ROOT}\"
        mkdir -p \"${REMOTE_LOGDIR}\"
        nohup env ${env_prefix} PORT=${PORT} LOG_DIR=\"${REMOTE_LOGDIR}\" bash scripts/serve_only/serve_vllm.sh \
            > \"${SUP_OUT}\" 2>&1 < /dev/null &
        echo \$! > \"${SUP_PIDFILE}\"
        echo \"[remote_serve] supervisor pid=\$(cat ${SUP_PIDFILE}) (remote)\"
    '" || die "remote launch failed (see ${SUP_OUT} on ${REMOTE})."

    echo "[remote_serve] waiting up to ${READY_TIMEOUT}s for ${REMOTE}:${PORT}/v1/models ..."
    local waited=0
    while (( waited < READY_TIMEOUT )); do
        if remote_ready; then
            echo "[remote_serve] READY — ${REMOTE}:${PORT}/v1/models is up."
            echo "[remote_serve] next: bash scripts/serve_only/tunnel.sh   (then run_tb_client.sh)"
            return 0
        fi
        # Fail fast if the supervisor died during model load.
        if ! ssh "$REMOTE" "kill -0 \$(cat '${REMOTE_KIT_ROOT}/${SUP_PIDFILE}' 2>/dev/null) 2>/dev/null"; then
            echo "[remote_serve] supervisor exited before becoming ready — last log:" >&2
            ssh "$REMOTE" "tail -n 40 '${REMOTE_KIT_ROOT}/${VLLM_LOG}' '${REMOTE_KIT_ROOT}/${SUP_OUT}' 2>/dev/null" >&2 || true
            die "server failed to start."
        fi
        sleep 5; (( waited += 5 ))
    done
    # Distinct exit code (75) for a readiness TIMEOUT vs a crash (die -> 1), so the
    # caller can reallocate a slow host but hard-fail a server that crashed.
    echo "[remote_serve] ERROR: timed out after ${READY_TIMEOUT}s waiting for /v1/models." >&2
    exit 75
}

do_stop() {
    echo "[remote_serve] stopping vllm on ${REMOTE} ..."
    ssh "$REMOTE" "bash -lc '
        cd \"${REMOTE_KIT_ROOT}\" || exit 0
        sup=\$(cat \"${SUP_PIDFILE}\" 2>/dev/null || true)
        pgid=\$(cat \"${PGID_FILE}\" 2>/dev/null || true)
        if [[ -n \"\$sup\" ]] && kill -0 \"\$sup\" 2>/dev/null; then
            echo \"[remote_serve] SIGTERM supervisor \$sup (graceful + escalation)\"
            kill -TERM \"\$sup\" 2>/dev/null || true
        elif [[ -n \"\$pgid\" ]]; then
            echo \"[remote_serve] supervisor gone; SIGTERM vllm group \$pgid\"
            kill -TERM -\"\$pgid\" 2>/dev/null || true
        else
            echo \"[remote_serve] no pidfile found — already stopped?\"
        fi
        # Wait for the port to go quiet, then hard-kill the group as a backstop.
        for _ in \$(seq 1 30); do
            curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1 || break
            sleep 1
        done
        if [[ -n \"\$pgid\" ]] && kill -0 -\"\$pgid\" 2>/dev/null; then
            echo \"[remote_serve] backstop SIGKILL group \$pgid\"
            kill -KILL -\"\$pgid\" 2>/dev/null || true
        fi
        rm -f \"${SUP_PIDFILE}\" \"${PGID_FILE}\"
    '"
    if remote_ready; then
        die "port ${PORT} still answering on ${REMOTE} — manual check needed."
    fi
    echo "[remote_serve] stopped. GPU memory on ${REMOTE}:"
    ssh "$REMOTE" "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true"
}

do_status() {
    local alive="no" ready="no"
    ssh "$REMOTE" "kill -0 \$(cat '${REMOTE_KIT_ROOT}/${SUP_PIDFILE}' 2>/dev/null) 2>/dev/null" && alive="yes"
    remote_ready && ready="yes"
    echo "[remote_serve] ${REMOTE}:${PORT}  supervisor_alive=${alive}  models_endpoint_ready=${ready}"
    [[ "$ready" == "yes" ]]
}

do_logs() {
    echo "[remote_serve] tailing ${REMOTE}:${REMOTE_KIT_ROOT}/${VLLM_LOG} (Ctrl-C to detach)"
    ssh "$REMOTE" "tail -n 80 -f '${REMOTE_KIT_ROOT}/${VLLM_LOG}'"
}

case "$CMD" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    logs)    do_logs ;;
    *)       die "unknown command '$CMD' (use start|stop|restart|status|logs)" ;;
esac
