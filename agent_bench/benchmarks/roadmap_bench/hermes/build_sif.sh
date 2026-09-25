#!/usr/bin/env bash
# Build the pinned, Agent_Bench-patched Hermes runtime on a local machine.
# Requires Docker and Apptainer (or Singularity); never runs on an NSCC GPU node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
VERSION_FILE="$REPO_ROOT/agent_bench/harnesses/hermes/patch/VERSION"
VERSION="$(awk -F': *' '$1 == "upstream_version" {print $2; exit}' "$VERSION_FILE")"
[[ -n "$VERSION" ]] || { echo "Cannot read Hermes version" >&2; exit 2; }

OUT="${1:-$PWD/hermes_runtime.sif}"
OUT="$(realpath -m "$OUT")"
IMAGE="agent-bench/hermes:${VERSION}"
if command -v apptainer >/dev/null 2>&1; then
    ENGINE=apptainer
elif command -v singularity >/dev/null 2>&1; then
    ENGINE=singularity
else
    echo "Apptainer or Singularity is required" >&2
    exit 2
fi
command -v docker >/dev/null 2>&1 || { echo "Docker is required" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing SIF: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
TMP="$(mktemp -d "$(dirname "$OUT")/.hermes-build.XXXXXXXX")"
trap 'rm -rf "$TMP"' EXIT

AB_AGENT_RUNTIME_IMAGE="$IMAGE" bash "$REPO_ROOT/agent_bench/harnesses/hermes/docker/build.sh"
docker save --output "$TMP/hermes.tar" "$IMAGE"
"$ENGINE" build "$TMP/hermes.sif" "docker-archive:$TMP/hermes.tar"
"$ENGINE" exec "$TMP/hermes.sif" sh -lc 'command -v hermes && python3 -c "import hermes_cli.oneshot"'
"$ENGINE" exec "$TMP/hermes.sif" sh -lc \
    'test "$HERMES_PATCH_MODE" = baked && test "$HERMES_VERSION" = "'"$VERSION"'" && python3 /opt/hermes-tools/apply_patch.py status'
mv "$TMP/hermes.sif" "$OUT"
echo "Built: $OUT"
