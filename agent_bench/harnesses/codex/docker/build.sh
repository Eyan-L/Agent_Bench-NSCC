#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=$(awk -F': *' '$1 == "codex_version" {print $2}' "$here/VERSION")
docker build --build-arg "CODEX_VERSION=$version" -t "agent-bench/codex:$version" -f "$here/docker/Dockerfile" "$here/docker"
