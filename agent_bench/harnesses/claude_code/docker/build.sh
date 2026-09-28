#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
claude_version=$(awk -F': *' '$1 == "claude_code_version" {print $2}' "$here/VERSION")
litellm_version=$(awk -F': *' '$1 == "litellm_version" {print $2}' "$here/VERSION")
tag=${AB_AGENT_RUNTIME_IMAGE:-"agent-bench/claude-code:$claude_version"}
docker build --build-arg "CLAUDE_CODE_VERSION=$claude_version" \
    --build-arg "LITELLM_VERSION=$litellm_version" -t "$tag" \
    -f "$here/docker/Dockerfile" "$here/docker"
