#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/claude-code-runtime.sif" >&2
    exit 2
fi

runtime=${APPTAINER:-$(command -v apptainer || command -v singularity)}
sif=$1
test -s "$sif"

claude_size=$("$runtime" exec --containall "$sif" \
    stat -c %s /opt/claude-code/bin/claude)
if (( claude_size <= 1000000 )); then
    echo "Claude Code native binary is missing or still an npm placeholder" >&2
    exit 1
fi
"$runtime" exec --containall "$sif" /opt/claude-code/bin/claude --version
"$runtime" exec --containall "$sif" /opt/litellm/bin/python -c \
    'import importlib.metadata; print("LiteLLM", importlib.metadata.version("litellm"))'
"$runtime" exec --containall "$sif" test -x /opt/litellm/bin/litellm
echo "[claude-code-runtime] PASS $sif"
