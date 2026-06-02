#!/usr/bin/env bash
# scripts/openclaw_apply_profile.sh
#
# Apply the minimal SWE-bench coding profile to the OpenClaw agent's workspace.
# Idempotent — safe to re-run.
#
# What it does:
#   1. Cleans the host-side stray testbed dir that OpenClaw created when
#      agent profile was misconfigured.
#   2. Copies our minimal AGENTS.md / SOUL.md / IDENTITY.md / USER.md /
#      TOOLS.md / HEARTBEAT.md into the agent's workspace.
#   3. Backs up the original files OpenClaw may have auto-generated.
#
# Env:
#   AB_AGENT_WORKSPACE   where the files go. Default: /testbed (the SWE-bench
#                        repo location inside the task container). For host-side
#                        testing, override to a temp dir.
#   PROFILE_SRC          source dir with the minimal .md files.
#                        Default: ./openclaw_profile (relative to project root).

set -euo pipefail

AB_REPO_ROOT="${AB_REPO_ROOT:-$(pwd)}"
AB_AGENT_WORKSPACE="${AB_AGENT_WORKSPACE:-./testbed}"
PROFILE_SRC="${PROFILE_SRC:-$AB_REPO_ROOT/openclaw_profile}"

echo "[apply_profile] workspace: $AB_AGENT_WORKSPACE"
echo "[apply_profile] source:    $PROFILE_SRC"

if [[ ! -d "$PROFILE_SRC" ]]; then
    echo "ERROR: profile source not found at $PROFILE_SRC" >&2
    exit 1
fi

# 1. Clean the stray host directory that OpenClaw polluted with its fallback
#    workspace files. Only clean if it clearly looks like OpenClaw's own output
#    (contains AGENTS.md + SOUL.md + IDENTITY.md + MEMORY markers).
STRAY="$AB_REPO_ROOT/testbed"
if [[ -d "$STRAY" ]]; then
    if [[ -f "$STRAY/AGENTS.md" && -f "$STRAY/SOUL.md" ]]; then
        echo "[apply_profile] cleaning stray host dir: $STRAY"
        rm -rf "$STRAY"
    else
        echo "[apply_profile] WARN: $STRAY exists but doesn't look like OpenClaw spill —"
        echo "                     leaving it alone. Inspect manually."
    fi
fi

# 2. Create workspace dir if missing.
mkdir -p "$AB_AGENT_WORKSPACE"

# 3. Back up any pre-existing files OpenClaw auto-generated, then overwrite.
BACKUP_DIR="$AB_AGENT_WORKSPACE/.oc_backup_$(date +%s)"
needs_backup=0
for f in AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md; do
    if [[ -f "$AB_AGENT_WORKSPACE/$f" ]]; then
        needs_backup=1
        break
    fi
done

if [[ "$needs_backup" -eq 1 ]]; then
    mkdir -p "$BACKUP_DIR"
    echo "[apply_profile] backing up existing files to $BACKUP_DIR"
    for f in AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md; do
        if [[ -f "$AB_AGENT_WORKSPACE/$f" ]]; then
            cp "$AB_AGENT_WORKSPACE/$f" "$BACKUP_DIR/"
        fi
    done
fi

# 4. Copy our profile files in.
for f in AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md; do
    cp "$PROFILE_SRC/$f" "$AB_AGENT_WORKSPACE/$f"
    echo "[apply_profile]   wrote $AB_AGENT_WORKSPACE/$f ($(wc -c < "$PROFILE_SRC/$f") bytes)"
done

echo
echo "[apply_profile] done."
echo "Total profile size:"
wc -c "$AB_AGENT_WORKSPACE"/AGENTS.md "$AB_AGENT_WORKSPACE"/SOUL.md \
      "$AB_AGENT_WORKSPACE"/IDENTITY.md "$AB_AGENT_WORKSPACE"/USER.md \
      "$AB_AGENT_WORKSPACE"/TOOLS.md "$AB_AGENT_WORKSPACE"/HEARTBEAT.md
