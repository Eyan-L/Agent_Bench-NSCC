#!/usr/bin/env bash
# scripts/serve_only/tunnel.sh — ntu_cpu-side SSH port-forward to the GPU host's
# vLLM server, for GPU hosts that are NOT directly routable from this node
# (e.g. a workstation reachable only through a ProxyJump).
#
# Binds the forward on 0.0.0.0 so the Terminal-Bench docker agent container can
# reach it via vllm-host -> host-gateway, regardless of which per-run docker
# bridge network the harness creates. After this is up:
#   curl http://127.0.0.1:8000/v1/models      # from the host
# and the TB harness rewrites localhost -> vllm-host for the agent container.
#
# For a directly-routed GPU host (one your machine can reach on the vLLM port) you
# do NOT need this — skip the tunnel and point the client at the host IP:
#   OPENAI_BASE_URL=http://<gpu-host-ip>:8000/v1
#
# Prerequisite (one-time): a Host entry for the GPU host in ~/.ssh/config, e.g.:
#   Host <gpu-host>
#       HostName <gpu-host.example.com>
#       User <user>
#       IdentityFile ~/.ssh/<your-key>
#       ProxyJump <jump-host>   # omit if the GPU host is directly reachable
#
# Env:
#   REMOTE        SSH host alias of the GPU node (or the LOGIN node of a cluster
#                 when the server runs on a scheduler-allocated compute node).
#                 REQUIRED (set in .env or on the command line; see .env.example).
#   FORWARD_HOST  Host the login/SSH host should forward TO, as resolved on that
#                 host. Default: 127.0.0.1 (server runs on REMOTE itself). For a
#                 scheduler-allocated node set this to the COMPUTE node hostname so
#                 the forward chains ntu_cpu -> login (REMOTE) -> compute:REMOTE_PORT.
#   LOCAL_PORT    Local listen port. Default: 8000
#   REMOTE_PORT   Port vLLM listens on at the GPU/compute host. Default: 8000
#   BIND_ADDR     Local bind address. Default: 0.0.0.0 (needed for docker agent)
#   SSH_CONFIG    Optional scoped ssh config (`ssh -F`). The fan-out orchestrator
#                 sets this to drive nscc under a specific account; unset = plain
#                 ssh / ~/.ssh/config.
#
# Usage:
#   bash scripts/serve_only/tunnel.sh            # foreground, Ctrl-C to stop
#   REMOTE=<gpu-host> bash scripts/serve_only/tunnel.sh
#   REMOTE=ccds-ai FORWARD_HOST=node05 bash scripts/serve_only/tunnel.sh   # 2-hop
set -euo pipefail

REMOTE="${REMOTE:-}"   # REQUIRED — SSH alias or user@ip (see .env.example)
FORWARD_HOST="${FORWARD_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
BIND_ADDR="${BIND_ADDR:-0.0.0.0}"
# `ssh -F <scoped config>` when the orchestrator set SSH_CONFIG (multi-account
# nscc); empty otherwise. As an array so the unset case adds no empty arg.
SSH_F=(); [[ -n "${SSH_CONFIG:-}" ]] && SSH_F=(-F "$SSH_CONFIG")

if ! ssh "${SSH_F[@]}" -G "$REMOTE" >/dev/null 2>&1; then
    echo "[tunnel] ERROR: '$REMOTE' is not resolvable by ssh." >&2
    echo "[tunnel] Add a Host entry to ~/.ssh/config (see header of this script)." >&2
    exit 1
fi

echo "[tunnel] forwarding ${BIND_ADDR}:${LOCAL_PORT} -> ${REMOTE}:${FORWARD_HOST}:${REMOTE_PORT}"
echo "[tunnel] keep this running; test from another shell: curl http://127.0.0.1:${LOCAL_PORT}/v1/models"

# -N: no remote command. ServerAlive*: drop dead tunnels so a restart is clean.
# FORWARD_HOST is resolved on REMOTE: 127.0.0.1 = the SSH host itself; a compute
# node hostname = a 2-hop forward (ntu_cpu -> login REMOTE -> compute:REMOTE_PORT).
exec ssh "${SSH_F[@]}" -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "${BIND_ADDR}:${LOCAL_PORT}:${FORWARD_HOST}:${REMOTE_PORT}" \
    "$REMOTE"
