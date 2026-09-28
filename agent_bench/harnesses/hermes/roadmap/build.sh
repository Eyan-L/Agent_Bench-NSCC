#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=$(awk -F': *' '$1 == "upstream_version" {print $2; exit}' "$here/patch/VERSION")
base_tag="agent-bench/hermes-roadmap-base:$version"
tag=${AB_AGENT_RUNTIME_IMAGE:-"agent-bench/hermes-roadmap:$version"}
docker build --build-arg "HERMES_VERSION=$version" -t "$base_tag" \
    -f "$here/docker/Dockerfile" "$here"
docker build --build-arg "HERMES_BASE_IMAGE=$base_tag" -t "$tag" \
    -f "$here/roadmap/Dockerfile" "$here/roadmap"
