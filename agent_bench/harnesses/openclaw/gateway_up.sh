#!/usr/bin/env bash
# Start the OpenClaw Gateway inside a SWE-bench run container.
#
# Per-container lifecycle: the gateway runs `openclaw gateway run` in the
# background inside the container, then we poll the readiness marker in its
# log. Token + port are injected via environment variables; both must be
# provided by the caller and are also written to /tmp/oc_gateway_env inside
# the container so subsequent `docker exec openclaw agent ...` invocations can
# source them.
#
# Usage: agent_bench/harnesses/openclaw/gateway_up.sh <container_id> <port> <token> [run_id]
#
# `run_id` (optional) is set as an OTel resource attribute (`oc.run_id`) so
# that spans from concurrent runs can be demuxed at analysis time. Required
# once driver runs go parallel.
#
# Exit codes:
#   0  ready
#   1  bad args / docker not available
#   2  gateway did not become ready in time

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <container_id> <port> <token> [run_id]" >&2
    exit 1
fi
CID="$1"; PORT="$2"; TOKEN="$3"; RUN_ID="${4:-}"
OTEL_RES_ATTRS=""
if [ -n "$RUN_ID" ]; then
    OTEL_RES_ATTRS="oc.run_id=$RUN_ID"
fi

if ! docker exec "$CID" true 2>/dev/null; then
    echo "[gateway-up] container $CID not reachable" >&2
    exit 1
fi

# Persist env inside the container so other docker exec calls can source it.
docker exec "$CID" bash -c "cat > /tmp/oc_gateway_env <<EOF
export OPENCLAW_GATEWAY_PORT=$PORT
export OPENCLAW_GATEWAY_TOKEN=$TOKEN
export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:$PORT
export OTEL_RESOURCE_ATTRIBUTES=$OTEL_RES_ATTRS
EOF"

# Start the gateway in the background. mode=local + bind=loopback come from
# the mounted openclaw.json. --auth token activates the token surface; the
# token itself is read from $OPENCLAW_GATEWAY_TOKEN. OTEL_RESOURCE_ATTRIBUTES
# tags every span/metric with oc.run_id so concurrent runs are demuxable.
echo "[gateway-up] starting gateway on port $PORT inside $CID (run_id=${RUN_ID:-unset})" >&2
docker exec -d \
    -e OPENCLAW_GATEWAY_TOKEN="$TOKEN" \
    -e OPENCLAW_GATEWAY_PORT="$PORT" \
    -e OTEL_RESOURCE_ATTRIBUTES="$OTEL_RES_ATTRS" \
    "$CID" bash -c "openclaw gateway run --port $PORT --auth token --bind loopback > /tmp/openclaw_gateway.log 2>&1"

# Poll readiness. The gateway logs a `ready (...)` line once startup
# (including plugin install) completes. First-run plugin install can take
# 1-2 minutes if the shared plugin cache is cold.
DEADLINE=$(( $(date +%s) + 180 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if docker exec "$CID" grep -q '\[gateway\] ready' /tmp/openclaw_gateway.log 2>/dev/null; then
        echo "[gateway-up] ready" >&2
        exit 0
    fi
    sleep 1
done

echo "[gateway-up] gateway did not become ready in 180s; last log lines:" >&2
docker exec "$CID" tail -30 /tmp/openclaw_gateway.log >&2 || true
exit 2
