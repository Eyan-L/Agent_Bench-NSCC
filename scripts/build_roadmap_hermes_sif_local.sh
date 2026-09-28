#!/usr/bin/env bash
# Run on local Ubuntu. The pinned Hermes Dockerfile applies the existing
# Agent_Bench bridge and telemetry patches before conversion to a SIF.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=$(awk -F': *' '$1 == "upstream_version" {print $2; exit}' \
  "$repo_root/agent_bench/harnesses/hermes/patch/VERSION")
if [[ -z "$version" ]]; then
  echo "Hermes upstream_version is missing" >&2
  exit 2
fi
builder=${APPTAINER:-$(command -v apptainer || command -v singularity || true)}
if [[ -z "$builder" ]] || ! command -v docker >/dev/null 2>&1; then
  echo "This local build needs Docker and Apptainer (or Singularity)" >&2
  exit 2
fi

output=${1:-"$repo_root/dist/hermes_roadmap_${version}.sif"}
base_tag="agent-bench/hermes-roadmap-base:${version}"
tag="agent-bench/hermes-roadmap:${version}"
mkdir -p "$(dirname "$output")"
partial="${output}.partial.$$"
trap 'rm -f "$partial"' EXIT

docker build --build-arg "HERMES_VERSION=$version" -t "$base_tag" \
  -f "$repo_root/agent_bench/harnesses/hermes/docker/Dockerfile" \
  "$repo_root/agent_bench/harnesses/hermes"
docker build --build-arg "HERMES_BASE_IMAGE=$base_tag" -t "$tag" \
  -f "$repo_root/agent_bench/harnesses/hermes/roadmap/Dockerfile" \
  "$repo_root/agent_bench/harnesses/hermes/roadmap"
"$builder" build --force "$partial" "docker-daemon://$tag"
"$builder" exec "$partial" bash -c \
  "HERMES_AGENT_ROOT=/opt/hermes HERMES_PATCH_MODE=baked python3 /opt/hermes-tools/apply_patch.py status && command -v hermes && test -d /app"
if [[ -e "$output" ]]; then
    ln -- "$output" "${output}.back.$(date +%Y%m%d%H%M%S).$$"
fi
mv -T -- "$partial" "$output"
trap - EXIT

sha256sum "$output"
echo "Built: $output"
echo "Set AB_AGENT_RUNTIME_SIF to the installed runtime path."
