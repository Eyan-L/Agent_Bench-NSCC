#!/usr/bin/env bash
# Clean stale / partial / disabled plugin extension dirs from the OC plugin
# runtime-deps cache.
#
# OC's plugin extractor uses an atomic rename of a temp dir into
# .../dist/extensions/<plugin>/. If a previous run was interrupted mid-extract,
# the destination ends up half-populated and the next run dies with
#   ENOTEMPTY: directory not empty, rename ...
# OC has no clobber flag — the only fix is to remove the offender first.
#
# This script removes:
#   1. Any extensions/<plugin>/ that lacks a package.json (= partial extract).
#   2. Any extensions/<plugin>/ whose name appears in the openclaw.json
#      `plugins.entries.<name>.enabled = false` block (= we don't use it
#      anyway, so we don't care about its cached state).
#
# Idempotent. Safe to run before every gateway launch.
#
# Usage:
#   oc_plugin_cache_clean.sh <cache-dir> [<openclaw.json>]
#
# <cache-dir>    : path to the AB_HARNESS_PLUGIN_CACHE root (contains
#                  openclaw-<version>-<hash>/dist/extensions/<plugin>/...).
# <openclaw.json>: optional; reads disabled-plugin names from
#                  plugins.entries. If omitted, only partial-extract cleanup
#                  runs.

set -uo pipefail

CACHE="${1:-}"
CONFIG="${2:-}"

if [[ -z "$CACHE" ]]; then
    echo "usage: $0 <cache-dir> [<openclaw.json>]" >&2
    exit 2
fi
if [[ ! -d "$CACHE" ]]; then
    # Cache doesn't exist yet — nothing to clean.
    exit 0
fi

LOG() { echo "[oc-plugin-clean] $*"; }

cleaned_partial=0
cleaned_disabled=0

# Pass 1: partial extracts (no package.json marker → OC can't recognize them
# as installed and will retry the rename, hitting ENOTEMPTY).
while IFS= read -r -d '' ext; do
    # ext is .../openclaw-<v>/dist/extensions/<plugin>
    if [[ -d "$ext" && ! -f "$ext/package.json" ]]; then
        # Skip empties that are zero-byte placeholder dirs from in-flight rename:
        # those don't trip ENOTEMPTY anyway. We only care about partials with
        # *some* content.
        if [[ -n "$(ls -A "$ext" 2>/dev/null)" ]]; then
            rm -rf "$ext" 2>/dev/null && cleaned_partial=$((cleaned_partial + 1))
        fi
    fi
done < <(find "$CACHE" -mindepth 4 -maxdepth 4 -type d -path '*/dist/extensions/*' -print0 2>/dev/null)

# Pass 2: explicitly-disabled plugins. We don't load them, so the cache state
# is irrelevant — wipe so OC doesn't waste time on extraction or panic on
# inconsistent state.
if [[ -n "$CONFIG" && -f "$CONFIG" ]] && command -v python3 >/dev/null 2>&1; then
    disabled_names=$(python3 - "$CONFIG" <<'PY' 2>/dev/null || true
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
entries = (cfg.get("plugins") or {}).get("entries") or {}
for name, spec in entries.items():
    if isinstance(spec, dict) and spec.get("enabled") is False:
        print(name)
PY
)
    if [[ -n "$disabled_names" ]]; then
        while IFS= read -r name; do
            [[ -z "$name" ]] && continue
            # Glob across all openclaw-<version>-<hash> bundles in the cache.
            while IFS= read -r -d '' ext; do
                if [[ -d "$ext" ]]; then
                    rm -rf "$ext" 2>/dev/null && \
                        cleaned_disabled=$((cleaned_disabled + 1))
                fi
            done < <(find "$CACHE" -mindepth 4 -maxdepth 4 -type d \
                          -path "*/dist/extensions/$name" -print0 2>/dev/null)
        done <<< "$disabled_names"
    fi
fi

if (( cleaned_partial > 0 || cleaned_disabled > 0 )); then
    LOG "cleaned: partial=$cleaned_partial disabled=$cleaned_disabled (cache=$CACHE)"
fi

exit 0
