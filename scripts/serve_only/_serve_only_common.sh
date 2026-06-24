#!/usr/bin/env bash
# scripts/serve_only/_serve_only_common.sh — shared helpers for the cluster
# wrappers (run_ccds.sh, run_nscc.sh). Source it; do not execute directly.
#
# The decoupled model identity rule: the agent sends model_config.name as the
# request `model`, and vLLM serves under the path it loads (BASE_FLAGS[0]=$MODEL,
# no --served-model-name). So the served id == the weights path. Since the weights
# live at $HOME/scratch/models/... and $HOME differs per cluster, we resolve the
# remote $HOME at runtime: the SERVER gets MODEL=<resolved path> (+ the committed
# template for params), and the CLIENT gets a name-patched copy of that template.

# Load site-specific defaults (your PBS PROJECT, VLLM_VENV, REMOTE, …) from the
# gitignored repo-root .env so a wrapper run directly picks them up — without
# baking any personal identifier into the committed scripts (this repo is public;
# see .env.example). Only vars NOT already in the environment are set, so a CLI
# override (`REMOTE=… bash run_*.sh`) and the orchestrator's per-cell env still
# win. No-op when the file is absent. Set AB_ENV_FILE to point elsewhere.
_so_load_env() {
    local envf="${AB_ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env}"
    [[ -f "$envf" ]] || return 0
    local line key val
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"                         # strip comments
        [[ "$line" == *=* ]] || continue
        key="${line%%=*}"; key="${key//[[:space:]]/}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -n "${!key:-}" ]] && continue           # already set -> caller wins
        val="${line#*=}"; val="${val#"${val%%[![:space:]]*}"}"; val="${val%"${val##*[![:space:]]}"}"
        val="${val#[\"\']}"; val="${val%[\"\']}"   # drop one layer of quotes
        export "$key=$val"
    done < "$envf"
}
_so_load_env

# "Password mode": a per-account password file is configured (SO_SSH_PASS_FILE,
# set by the orchestrator to ~/.ssh/nscc_<user>.pass). nscc disabled SSH key auth
# — only passwords are accepted — so we either feed the password via sshpass or
# reuse a ControlMaster socket pre-opened by nscc_login.sh. Unset/absent file =
# the default key/agent behaviour (hpcc and other key hosts are untouched).
so_pass_mode() { [[ -n "${SO_SSH_PASS_FILE:-}" && -f "${SO_SSH_PASS_FILE:-}" ]]; }

# Run ssh, transparently injecting `-F $SSH_CONFIG` when the caller set it. This
# is how the fan-out orchestrator drives nscc under multiple accounts WITHOUT
# touching ~/.ssh/config: it generates a scoped config (one `Host nscc-<user>`
# block per account) and exports SSH_CONFIG=<that file>. Every `ssh "$REMOTE" …`
# in the wrappers goes through here, so the same REMOTE alias resolves under
# whichever account the scoped config defines. Unset SSH_CONFIG = plain ssh (the
# default single-account / ~/.ssh/config behaviour). Pass-through of all args
# means `-G`, `-o …`, `-N -L …` forms all keep working.
#
# Password mode (nscc key auth disabled): the generated config carries
# `ControlMaster auto` + a shared ControlPath, so the password authenticates ONCE
# per account and every later call (qstat poll, qdel, tunnel) multiplexes over
# that master with no re-auth — which also collapses the per-cell connection
# count that used to trip the login node's MaxStartups. The first connection is
# authenticated by sshpass (if installed) reading SO_SSH_PASS_FILE; without
# sshpass it reuses a master pre-opened by nscc_login.sh. stdin is /dev/null in
# the wrappers, so a missing password never hangs — ssh fails fast instead.
so_ssh() {
    local pre=() auth=()
    if so_pass_mode && command -v sshpass >/dev/null 2>&1; then
        pre=(sshpass -f "$SO_SSH_PASS_FILE")
        auth=(-o PreferredAuthentications=password -o PubkeyAuthentication=no \
              -o NumberOfPasswordPrompts=1)
    fi
    "${pre[@]}" ssh ${SSH_CONFIG:+-F "$SSH_CONFIG"} "${auth[@]}" "$@"
}

# Retry a command on transient failure with exponential backoff + jitter.
# Built for SSH-to-login-node calls (e.g. the qsub submission) that get reset
# pre-auth — "kex_exchange_identification: Connection reset by peer" — when many
# fan-out cells dogpile one login node's sshd at once (MaxStartups / fail2ban).
# Such a reset never runs the remote command (ssh exits 255 before auth), so a
# retry can't double-submit. Echoes the command's stdout; returns the last
# attempt's exit code. Tunable via env: SO_SSH_RETRIES (default 5),
# SO_SSH_BACKOFF_BASE (seconds, default 3).
so_retry() {
    local attempts="${SO_SSH_RETRIES:-5}"
    local base="${SO_SSH_BACKOFF_BASE:-3}"
    local n=1 out rc delay
    while :; do
        out="$("$@")"; rc=$?
        (( rc == 0 )) && { printf '%s' "$out"; return 0; }
        (( n >= attempts )) && { printf '%s' "$out"; return "$rc"; }
        # exponential backoff (base, 2x, 4x, ...) plus jitter, to spread the
        # retried connections out instead of dogpiling the login node again.
        delay=$(( base * 2 ** (n - 1) + RANDOM % base ))
        echo "${TAG:-[serve_only]} ssh attempt $n/$attempts failed (rc=$rc); retry in ${delay}s ..." >&2
        sleep "$delay"
        n=$(( n + 1 ))
    done
}

# Resolve $HOME on a remote SSH host (login node). Echoes the absolute path.
so_remote_home() {
    local remote="$1" bm=(-o BatchMode=yes)
    # BatchMode=yes disables password auth, which would block sshpass / a fresh
    # master in password mode; drop it there (NumberOfPasswordPrompts=1 still
    # keeps a missing password from hanging).
    so_pass_mode && bm=()
    so_ssh "${bm[@]}" "$remote" 'printf %s "$HOME"' 2>/dev/null
}

# Append one JSON record to $ALLOC_MANIFEST (the fan-out placement ledger), if
# set; a no-op otherwise (so the wrappers stay usable standalone). Args are
# key=value pairs; numeric-looking values are stored as numbers. A timestamp is
# added automatically. Each call writes a single line — the orchestrator merges
# these (keyed by run_id) into data/fanout/<run>/manifest.jsonl to record WHERE
# each subjob landed and the GPU it ran on (type + size). Wrappers call it twice:
# once at allocation (status=serving, full placement) and once on exit (status,
# rc). Line appends are atomic on a local fs, so the concurrent fan-out runs that
# share one manifest don't interleave partial lines.
so_write_alloc() {
    [[ -n "${ALLOC_MANIFEST:-}" ]] || return 0
    python3 - "$ALLOC_MANIFEST" "$@" <<'PY'
import json, os, sys, time
manifest = sys.argv[1]
rec = {"ts": round(time.time(), 3)}
for kv in sys.argv[2:]:
    k, _, v = kv.partition("=")
    if v == "":
        rec[k] = None
        continue
    try:
        rec[k] = int(v)
    except ValueError:
        try:
            rec[k] = float(v)
        except ValueError:
            rec[k] = v
d = os.path.dirname(manifest) or "."
os.makedirs(d, exist_ok=True)
with open(manifest, "a") as f:
    f.write(json.dumps(rec) + "\n")
PY
}

# Pick a currently-free TCP port on THIS (CPU) node. Used to give each parallel
# run its own local tunnel/endpoint port so many experiments can fan out at once
# without colliding on 0.0.0.0:8000. The OS hands out an ephemeral port (bind to
# :0), we read it back, then release it; the same number is reused as the remote
# served port so it's unique end-to-end. There is a tiny TOCTOU window between
# picking and the tunnel binding — tunnel.sh runs with ExitOnForwardFailure=yes,
# so a lost race fails fast and visibly rather than silently mis-routing.
so_pick_free_port() {
    python3 -c 'import socket
s = socket.socket()
s.bind(("0.0.0.0", 0))
print(s.getsockname()[1])
s.close()'
}

# Write a client model-config: copy $template, set model.name=$name -> $dst.
# Used locally on ntu_cpu so run_tb_client.sh requests the same id the server
# advertises. Written ATOMICALLY (temp file + os.replace) so parallel runs can
# share one $dst: a concurrent reader (run_tb_client's preflight json.load)
# always sees a complete file, never a half-written one. Keyed by model, so
# identical runs reuse it and two different models don't clobber each other.
so_gen_client_cfg() {
    local template="$1" name="$2" dst="$3"
    mkdir -p "$(dirname "$dst")"
    python3 - "$template" "$name" "$dst" <<'PY'
import json, os, sys, tempfile
template, name, dst = sys.argv[1:4]
with open(template) as f:
    cfg = json.load(f)
cfg["model"]["name"] = name
d = os.path.dirname(dst) or "."
fd, tmp = tempfile.mkstemp(dir=d, prefix=".model_cfg.", suffix=".json")
try:
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    os.replace(tmp, dst)   # atomic within the same filesystem
except BaseException:
    os.unlink(tmp)
    raise
print(f"[serve_only] client model-config: {dst} (name={name})")
PY
}
