#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$here/../../.." && pwd)
claude_version=$(awk -F': *' '$1 == "claude_code_version" {print $2}' "$here/VERSION")
litellm_version=$(awk -F': *' '$1 == "litellm_version" {print $2}' "$here/VERSION")
tag="agent-bench/claude-code:${claude_version}"
output=${1:-"$repo_root/dist/claude_code_${claude_version}_litellm_${litellm_version}.sif"}
builder=${APPTAINER:-$(command -v apptainer || command -v singularity || true)}

if [[ -z "$builder" ]]; then
    echo "apptainer or singularity is required" >&2
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to build the local runtime image" >&2
    exit 2
fi

mkdir -p "$(dirname "$output")"
temporary="${output}.partial.$$"
trap 'rm -f "$temporary"' EXIT

docker build \
    --build-arg "CLAUDE_CODE_VERSION=$claude_version" \
    --build-arg "LITELLM_VERSION=$litellm_version" \
    -t "$tag" \
    -f "$here/docker/Dockerfile" \
    "$here/docker"
"$builder" build --force "$temporary" "docker-daemon://$tag"
bash "$here/audit_sif.sh" "$temporary"
mv -f "$temporary" "$output"
trap - EXIT

echo "Built: $output"
echo "Upload: scp '$output' nscc:/scratch/users/ntu/n2602695/sif_images/"
