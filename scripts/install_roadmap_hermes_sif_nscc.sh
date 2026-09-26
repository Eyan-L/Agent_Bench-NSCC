#!/usr/bin/env bash
# Run on NSCC after uploading the new image as <target>.incoming.
set -euo pipefail

target=/scratch/users/ntu/n2602695/sif_images/hermes_roadmap_v2026.5.16.sif
incoming="${target}.incoming"

if [[ ! -s "$target" || ! -s "$incoming" ]]; then
  echo "Both the current and incoming Hermes SIF must exist and be nonempty" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "Load singularity/4.3.1 before running this script" >&2
  exit 2
fi

# Verify the new image before changing the path used by PBS jobs.
singularity exec "$incoming" sh -c \
  'HERMES_AGENT_ROOT=/opt/hermes HERMES_PATCH_MODE=baked python3 /opt/hermes-tools/apply_patch.py status'

backup="${target}.back"
if [[ -e "$backup" ]]; then
  backup="${backup}.$(date +%Y%m%d%H%M%S)"
fi
if [[ -e "$backup" ]]; then
  echo "Backup path already exists: $backup" >&2
  exit 2
fi

# The hard link preserves the exact old image; the final rename is atomic.
ln -- "$target" "$backup"
mv -T -- "$incoming" "$target"
echo "Previous SIF: $backup"
sha256sum "$target" "$backup"
