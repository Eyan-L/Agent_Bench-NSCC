#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/claude-code-runtime.sif" >&2
    exit 2
fi

runtime=${APPTAINER:-$(command -v apptainer || command -v singularity)}
sif=$1
test -s "$sif"

"$runtime" exec --containall "$sif" /opt/claude-code/bin/claude --version
"$runtime" exec --containall "$sif" /opt/litellm/bin/python -c \
    'import importlib.metadata; print("LiteLLM", importlib.metadata.version("litellm"))'
"$runtime" exec --containall "$sif" test -x /opt/litellm/bin/litellm
echo "[claude-code-runtime] PASS $sif"
