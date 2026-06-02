#!/usr/bin/env bash
# Pre-populate the OpenClaw plugin runtime deps cache.
#
# OpenClaw's gateway npm-installs each plugin's runtime deps (acpx, browser, ...)
# into OPENCLAW_STATE_DIR/plugin-runtime-deps on first start. HPC compute nodes
# typically lack internet, so the install fails (ECONNRESET) and the gateway
# never becomes ready. Run this once on a network-equipped node (login node
# usually) to populate the shared cache; subsequent compute-node runs reuse it
# offline.
#
# Usage:
#   pbs/warm_oc_plugins.sh \
#     --sif <openclaw_patched_runtime.sif> \
#     [--overlay <optional_patch_overlay.img>] \
#     [--cache <dir>] \
#     [--timeout-seconds 240]
#
# --cache defaults to $AB_HARNESS_PLUGIN_CACHE or ~/.cache/oc-eval/openclaw-plugin-deps,
# matching the path SingularityHarness binds at run time.
#
# Idempotent: re-running with a populated cache is a no-op past npm's "up to
# date" check (a few seconds).

set -euo pipefail

SIF=""
OVERLAY=""
CACHE="${AB_HARNESS_PLUGIN_CACHE:-$HOME/.cache/oc-eval/openclaw-plugin-deps}"
TIMEOUT=240
STATE_TEMPLATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sif)              SIF="$2"; shift 2 ;;
        --overlay)          OVERLAY="$2"; shift 2 ;;
        --cache)            CACHE="$2"; shift 2 ;;
        --timeout-seconds)  TIMEOUT="$2"; shift 2 ;;
        --state-template)   STATE_TEMPLATE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$SIF"     ]] || { echo "ERROR: --sif required" >&2; exit 1; }
[[ -f "$SIF"     ]] || { echo "ERROR: sif missing: $SIF" >&2; exit 1; }
[[ -z "$OVERLAY" || -f "$OVERLAY" ]] || { echo "ERROR: overlay missing: $OVERLAY" >&2; exit 1; }
command -v singularity >/dev/null || { echo "ERROR: singularity not on PATH" >&2; exit 1; }

mkdir -p "$CACHE"
TMP_STATE="$(mktemp -d -t oc-warm-state.XXXXXX)"
trap 'rm -rf "$TMP_STATE"' EXIT

# Sweep partial extracts from any prior interrupted warm so the gateway we're
# about to launch doesn't trip on ENOTEMPTY against a stale dist/extensions/<p>/.
# Idempotent — no-op when the cache is clean.
CLEAN_SH="$(dirname "$0")/oc_plugin_cache_clean.sh"
if [[ -x "$CLEAN_SH" ]]; then
    bash "$CLEAN_SH" "$CACHE" "${STATE_TEMPLATE:+$STATE_TEMPLATE/openclaw.json}" \
        2>&1 | sed 's/^/[warm-oc-plugins] /' || true
fi

OVERLAY_ARG=()
[[ -n "$OVERLAY" ]] && OVERLAY_ARG=(--overlay "$OVERLAY:ro")

mkdir -p "$TMP_STATE/plugin-runtime-deps"

# Gateway refuses to start without a configured openclaw.json. If a state
# template was given, copy its openclaw.json in; otherwise pass
# --allow-unconfigured (older OC builds may still refuse plugin loading).
GATEWAY_EXTRA_ARGS=()
if [[ -n "$STATE_TEMPLATE" ]]; then
    if [[ ! -f "$STATE_TEMPLATE/openclaw.json" ]]; then
        echo "ERROR: --state-template $STATE_TEMPLATE has no openclaw.json" >&2
        exit 1
    fi
    cp "$STATE_TEMPLATE/openclaw.json" "$TMP_STATE/openclaw.json"
    if [[ -d "$STATE_TEMPLATE/agents" ]]; then
        cp -r "$STATE_TEMPLATE/agents" "$TMP_STATE/agents"
    fi
    echo "[warm-oc-plugins] state template = $STATE_TEMPLATE (config copied)"
else
    GATEWAY_EXTRA_ARGS+=(--allow-unconfigured)
    echo "[warm-oc-plugins] no --state-template given, using --allow-unconfigured"
fi

echo "[warm-oc-plugins] cache    = $CACHE"
echo "[warm-oc-plugins] sif      = $SIF"
echo "[warm-oc-plugins] timeout  = ${TIMEOUT}s"
echo "[warm-oc-plugins] starting gateway briefly to trigger plugin install..."

# Use a non-default port to avoid colliding with anything else; bind loopback;
# kill the gateway as soon as it logs `[gateway] ready`. If it doesn't become
# ready within TIMEOUT seconds, kill anyway and surface the log.
PORT=$(( 30000 + RANDOM % 1000 ))
LOG="$TMP_STATE/gateway.log"

singularity exec \
    "${OVERLAY_ARG[@]}" \
    --bind "$TMP_STATE:$TMP_STATE:rw" \
    --bind "$CACHE:$TMP_STATE/plugin-runtime-deps:rw" \
    --env "OPENCLAW_STATE_DIR=$TMP_STATE" \
    --env "OPENCLAW_GATEWAY_TOKEN=warm-only" \
    "$SIF" \
    openclaw gateway run --port "$PORT" --auth token --bind loopback \
        "${GATEWAY_EXTRA_ARGS[@]}" \
        > "$LOG" 2>&1 &
PID=$!

DEADLINE=$(( $(date +%s) + TIMEOUT ))
READY=0
while [[ "$(date +%s)" -lt "$DEADLINE" ]]; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[warm-oc-plugins] gateway exited unexpectedly; tail of log:" >&2
        tail -40 "$LOG" >&2 || true
        exit 2
    fi
    if grep -q '\[gateway\] ready' "$LOG" 2>/dev/null; then
        READY=1
        break
    fi
    sleep 2
done

if (( READY != 1 )); then
    echo "[warm-oc-plugins] gateway did not reach 'ready' in ${TIMEOUT}s; tail:" >&2
    tail -40 "$LOG" >&2 || true
    kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    exit 3
fi

kill -TERM "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true

# Sanity: cache should now have at least one populated plugin dir.
if ! find "$CACHE" -mindepth 2 -name "package.json" -print -quit | grep -q .; then
    echo "[warm-oc-plugins] cache appears empty after warm; tail of log:" >&2
    tail -40 "$LOG" >&2 || true
    exit 4
fi

POPULATED=$(find "$CACHE" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "[warm-oc-plugins] done; $POPULATED plugin dir(s) populated under $CACHE"
