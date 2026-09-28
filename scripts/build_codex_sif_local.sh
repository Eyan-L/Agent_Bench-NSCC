#!/usr/bin/env bash
# Run in local Ubuntu; NSCC consumes the uploaded immutable runtime payload.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
here="$repo_root/agent_bench/harnesses/codex"
version=$(awk -F': *' '$1 == "codex_version" {print $2}' "$here/VERSION")
output=${1:-"$repo_root/dist/codex_${version}.sif"}
builder=${APPTAINER:-$(command -v apptainer || command -v singularity || true)}
[[ -n "$builder" ]] || { echo 'Apptainer or Singularity is required' >&2; exit 2; }
command -v docker >/dev/null
mkdir -p "$(dirname "$output")"
partial="${output}.partial.$$"
trap 'rm -f "$partial"' EXIT
bash "$here/docker/build.sh"
"$builder" build --force "$partial" "docker-daemon://agent-bench/codex:$version"
"$builder" exec "$partial" /opt/codex/bin/codex --version
"$builder" exec "$partial" sha256sum -c /opt/codex/SHA256SUMS
if [[ -e "$output" ]]; then
    backup="${output}.back.$(date +%Y%m%d%H%M%S)"
    ln -- "$output" "$backup"
fi
mv -T -- "$partial" "$output"
trap - EXIT
sha256sum "$output"
echo "Built: $output"
echo "Upload: scp '$output' nscc:/scratch/users/ntu/n2602695/sif_images/"
