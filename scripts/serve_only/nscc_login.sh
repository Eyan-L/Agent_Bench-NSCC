#!/usr/bin/env bash
# scripts/serve_only/nscc_login.sh — open a persistent SSH ControlMaster per nscc
# account so the fan-out orchestrator can run unattended AFTER nscc disabled SSH
# key auth (passwords only).
#
# Why: every fan-out cell makes many ssh calls to the nscc login node (qsub,
# qstat poll, qdel, the tunnel). With key auth gone each would prompt for a
# password — and the wrappers run with stdin=/dev/null, so they'd just fail. This
# script authenticates ONCE per account (you type the password interactively) and
# leaves a master socket up; the generated fan-out ssh config (`ControlMaster
# auto`, same ControlPath) multiplexes every later connection over it — no
# re-auth, far fewer login-node connections.
#
# This is the no-sshpass path. If you instead drop each password in
# ~/.ssh/nscc_<user>.pass (chmod 600) and install sshpass, the wrappers feed the
# password themselves and you can skip this script entirely. Both paths share the
# same ControlPath, so running this is also a harmless speed-up in the sshpass case.
#
# Usage:
#   bash scripts/serve_only/nscc_login.sh                 # every ~/.ssh/nscc_* account
#   bash scripts/serve_only/nscc_login.sh hong0259 boxi001
#   NSCC_HOSTNAME=aspire2antu.nscc.sg bash scripts/serve_only/nscc_login.sh hong0259
#   ACTION=stop  bash scripts/serve_only/nscc_login.sh   # tear the masters down
#   ACTION=check bash scripts/serve_only/nscc_login.sh   # show which are alive
set -euo pipefail

# NB: not $HOSTNAME — bash sets that to the LOCAL machine name.
HOSTNAME_NSCC="${NSCC_HOSTNAME:-aspire2antu.nscc.sg}"
CM_DIR="$HOME/.ssh/cm"
CM_PERSIST="${CM_PERSIST:-4h}"     # outlast the whole sweep; matches fanout.py
ACTION="${ACTION:-start}"
mkdir -p "$CM_DIR"; chmod 700 "$CM_DIR"

# Accounts: CLI args, else every ~/.ssh/nscc_<user> credential (.pass or a key).
accounts=("$@")
if (( ${#accounts[@]} == 0 )); then
    for p in "$HOME"/.ssh/nscc_*; do
        [[ -e "$p" ]] || continue
        n="$(basename "$p")"; n="${n#nscc_}"
        case "$n" in *.pub) continue;; *.pass) n="${n%.pass}";; esac
        accounts+=("$n")
    done
fi
(( ${#accounts[@]} )) || { echo "no nscc accounts found (~/.ssh/nscc_*)" >&2; exit 1; }
# De-dup (a user may have both a key and a .pass).
mapfile -t accounts < <(printf '%s\n' "${accounts[@]}" | sort -u)

# A scoped config matching fanout.py's gen_ssh_config so the ControlPath %C hash
# is identical — the master opened here is the one the orchestrator reuses.
cfg="$(mktemp)"; trap 'rm -f "$cfg"' EXIT
for u in "${accounts[@]}"; do
    {
        echo "Host nscc-$u"
        echo "    HostName $HOSTNAME_NSCC"
        echo "    User $u"
        echo "    ControlMaster auto"
        echo "    ControlPath \"$CM_DIR/%C\""
        echo "    ControlPersist $CM_PERSIST"
        if [[ -f "$HOME/.ssh/nscc_$u.pass" ]]; then
            echo "    PreferredAuthentications password"
            echo "    PubkeyAuthentication no"
            echo "    NumberOfPasswordPrompts 1"
        else
            echo "    IdentityFile \"$HOME/.ssh/nscc_$u\""
        fi
        echo
    } >> "$cfg"
done

for u in "${accounts[@]}"; do
    case "$ACTION" in
        stop)
            ssh -F "$cfg" -O exit "nscc-$u" 2>/dev/null \
                && echo "[nscc_login] $u: master closed" \
                || echo "[nscc_login] $u: no master"
            ;;
        check)
            if ssh -F "$cfg" -O check "nscc-$u" 2>/dev/null; then
                echo "[nscc_login] $u: master ALIVE"
            else
                echo "[nscc_login] $u: master down"
            fi
            ;;
        start)
            if ssh -F "$cfg" -O check "nscc-$u" 2>/dev/null; then
                echo "[nscc_login] $u: master already up"
                continue
            fi
            echo "[nscc_login] $u: opening master to $HOSTNAME_NSCC ..."
            # -M master, -f background after auth, -N no command. The prompt goes
            # to the tty; on success the socket persists for $CM_PERSIST.
            # If a password file exists and sshpass is installed, use it to avoid interactive prompt.
            pass_file="$HOME/.ssh/nscc_$u.pass"
            pre=()
            if [[ -f "$pass_file" ]] && command -v sshpass >/dev/null 2>&1; then
                pre=(sshpass -f "$pass_file")
            fi
            if "${pre[@]}" ssh -F "$cfg" -M -f -N "nscc-$u"; then
                echo "[nscc_login] $u: master up"
            else
                echo "[nscc_login] $u: FAILED to open master" >&2
            fi
            ;;
        *) echo "unknown ACTION=$ACTION (start|stop|check)" >&2; exit 2;;
    esac
done
