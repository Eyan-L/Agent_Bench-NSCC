#!/usr/bin/env bash
# Verify <target>.incoming, back up the old runtime, then replace it atomically.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <target.sif> [incoming.sif]" >&2
    exit 2
fi
target=$1
incoming=${2:-"${target}.incoming"}
builder=${APPTAINER:-$(command -v apptainer || command -v singularity || true)}
if [[ -z "$builder" || ! -s "$incoming" ]]; then
    echo "Apptainer/Singularity and a nonempty incoming SIF are required" >&2
    exit 2
fi
target_dir=$(dirname -- "$target")
if [[ "$(stat -c %d -- "$incoming")" != "$(stat -c %d -- "$target_dir")" ]]; then
    echo "Incoming SIF must be on the target filesystem for atomic replacement" >&2
    exit 2
fi
"$builder" exec "$incoming" sh -c \
    'HERMES_AGENT_ROOT=/opt/hermes HERMES_PATCH_MODE=baked python3 /opt/hermes-tools/apply_patch.py status'
if [[ -e "$target" ]]; then
    backup="${target}.back.$(date +%Y%m%d%H%M%S).$$"
    ln -- "$target" "$backup"
    echo "Previous SIF: $backup"
fi
mv -T -- "$incoming" "$target"
sha256sum "$target"
