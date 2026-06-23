#!/usr/bin/env bash
# scripts/execution_client/run_tb_client.sh — CPU-side Terminal-Bench runner for
# the DECOUPLED workflow. Runs on a CPU node (docker, no GPU). The vLLM server runs
# on a GPU host (via scripts/serve_only/serve_vllm.sh); this node
# reaches it over an SSH tunnel (scripts/serve_only/tunnel.sh) or a direct route.
#
# It does NOT start a server — it materializes the TB instance data, pulls the
# task docker image, starts the engine-metrics tailer against the remote endpoint,
# and runs the driver with the existing-endpoint serving profile.
#
# Use tb_react (the MVP condition) to validate the decoupled TRANSPORT (GPU serve
# + tunnel + CPU agent end-to-end). Switch HARNESS_CONFIG to another harness (e.g.
# profiles/harnesses/hermes_terminal_bench.yaml) for richer agent behaviour; any
# harness-specific knobs (e.g. extra_body toggles) ride the process environment,
# which the fan-out orchestrator populates per arm (see ARM_ENV in fanout.py).
#
# Env (override K=V on the command line):
#   APP             benchmark profile. Default: profiles/benchmarks/terminal_bench/kv_store.json
#                   (kv-store-grpc). 23 tasks are vendored under tasks/ (incl.
#                   fix-git => smoke1.json works offline too); profiles that
#                   reference NON-vendored tasks need AB_BENCHMARK_TASKS_ROOT
#                   staged with a TB-2-verified snapshot.
#   HARNESS_CONFIG  Default: profiles/harnesses/tb_react_terminal_bench.yaml
#   MODEL_CONFIG    Default: profiles/models/qwen36_35b_a3b_local.json
#                   MUST match the MODEL_CONFIG used by serve_vllm.sh on the GPU
#                   host — the agent sends model_config.name as the request model
#                   and vLLM serves under that same name.
#   SERVING_CONFIG  Default: profiles/serving/vllm/existing_endpoint.yaml
#   PORT            Local port where the endpoint is reachable. Default 8000.
#   ENDPOINT_HOST   Host of the endpoint as seen from THIS node. Default 127.0.0.1
#                   (the SSH tunnel). For a directly-routed GPU host set the IP,
#                   e.g. ENDPOINT_HOST=<gpu-host-ip> — then no tunnel needed.
#   ARM             Experiment-variant label for results-dir naming. Default: default.
#   SKIP_PULL=1     Skip the docker pull of task images (already present).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ ! -d "$KIT_ROOT/orchestration" ]]; then
    echo "[tb] FATAL: KIT_ROOT=$KIT_ROOT is not an Agent_Bench checkout" >&2; exit 1
fi

APP="${APP:-$KIT_ROOT/profiles/benchmarks/terminal_bench/kv_store.json}"
HARNESS_CONFIG="${HARNESS_CONFIG:-$KIT_ROOT/profiles/harnesses/tb_react_terminal_bench.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-$KIT_ROOT/profiles/models/qwen36_35b_a3b_local.json}"
SERVING_CONFIG="${SERVING_CONFIG:-$KIT_ROOT/profiles/serving/vllm/existing_endpoint.yaml}"
PORT="${PORT:-8000}"
ENDPOINT_HOST="${ENDPOINT_HOST:-127.0.0.1}"
BASE="http://${ENDPOINT_HOST}:${PORT}"

# Generic experiment-variant label (used only for results-dir naming). The fan-out
# orchestrator sets it per cell; any harness-specific env knobs an arm needs are
# already in this process's environment (fanout.py ARM_ENV overlay) and inherited
# by the driver subprocess below.
ARM="${ARM:-default}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Include the endpoint PORT so parallel fan-out runs (each on its own tunnel
# port) get distinct results dirs even when started in the same second.
RESULTS="${RESULTS:-$KIT_ROOT/data/tb_client_${ARM}_${STAMP}_p${PORT}}"
LOG_DIR="$RESULTS/logs"; mkdir -p "$LOG_DIR"

# Consolidate the driver's experiment output INTO this run's results dir, instead
# of a fresh data/experiments/<stamp>/runs/<leaf>/. Under fan-out, RESULTS is the
# per-cell dir, so every GPU's results land under the one fan-out run tree
# (RESULTS/runs/<leaf>/<run_id>/manifest.json) and the aggregator can find each
# arm's manifests. AB_RUN_DIR and AB_EXPERIMENT_DIR are strict aliases — keep equal.
export AB_EXPERIMENT_DIR="$RESULTS" AB_RUN_DIR="$RESULTS"

# --- decoupled serving wiring ---------------------------------------------
# Docker harness on this CPU node; no local server. The driver resolves the
# endpoint from the existing-endpoint serving profile (and these env vars).
export AB_DEPLOYMENT_CONTAINER_BACKEND=docker
# Hermes agent image (only used when HARNESS_CONFIG selects the hermes harness;
# ignored by tb_react). v2026.5.29.2 forwards the custom_providers extra_body
# (e.g. chat_template_kwargs knobs) to vLLM.
export AB_AGENT_RUNTIME_IMAGE="${AB_AGENT_RUNTIME_IMAGE:-agent-bench/hermes:v2026.5.29.2}"
export KIT_ROOT AB_REPO_ROOT="$KIT_ROOT"
export AB_BENCHMARK_CONFIG="$APP" AB_HARNESS_CONFIG="$HARNESS_CONFIG"
export AB_MODEL_CONFIG="$MODEL_CONFIG" AB_SERVING_CONFIG="$SERVING_CONFIG"
export AB_SERVING_BACKEND=existing_endpoint
export AB_SERVING_BASE_URL="${BASE}/v1"
export OPENAI_BASE_URL="${BASE}/v1"
# benchmark-data-root just needs to exist (driver uses it for scoring log layout).
AB_BENCHMARK_DATA_ROOT="${AB_BENCHMARK_DATA_ROOT:-$RESULTS/bench_root}"; mkdir -p "$AB_BENCHMARK_DATA_ROOT"

echo "[tb] arm=$ARM  node=$(hostname)"
echo "[tb] app=$APP"
echo "[tb] harness=$HARNESS_CONFIG"
echo "[tb] endpoint=$BASE/v1  (model-config=$MODEL_CONFIG)"
echo "[tb] results=$RESULTS"

# --- wait for the remote endpoint -----------------------------------------
# ENDPOINT_WAIT_SECS: how long to wait for /v1/models. A 35B FP8 MoE on 2xA100
# needs ~8 min just to load weights, then FP8 processing + torch.compile +
# CUDA-graph capture, so the default is generous. Bump for bigger models / TP.
ENDPOINT_WAIT_SECS="${ENDPOINT_WAIT_SECS:-1200}"
echo "[tb] waiting for $BASE/v1/models (up to $((ENDPOINT_WAIT_SECS / 60)) min)..."
ready=0
# Bound by WALL-CLOCK, not iteration count, and cap each curl: a firewall that
# DROPs packets to the endpoint makes a plain `curl` block for minutes per probe,
# so an iteration-counted loop runs far past ENDPOINT_WAIT_SECS and never times
# out (it just burns the whole fan-out budget). --connect-timeout/--max-time keep
# each probe short; the deadline keeps the total honest so reallocation can fire.
_wait_deadline=$(( $(date +%s) + ENDPOINT_WAIT_SECS ))
while (( $(date +%s) < _wait_deadline )); do
    curl -sf --connect-timeout 5 --max-time 10 "$BASE/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    sleep 5
done
[[ "$ready" == "1" ]] || {
    echo "[tb] FATAL: endpoint $BASE not reachable after $((ENDPOINT_WAIT_SECS / 60))min." >&2
    # The serve job (ccds srun / nscc PBS / baremetal) never answered /v1/models in
    # time. Emit the reallocate marker so the fan-out requeues this cell onto another
    # GPU; the wrapper's cleanup trap then tears the slow serve job down on exit.
    echo "[tb] reallocate: server not ready within $((ENDPOINT_WAIT_SECS / 60))min — releasing this GPU for another lane." >&2
    echo "[tb]   (is serve_vllm.sh up on the GPU host, and the tunnel running? scripts/serve_only/tunnel.sh)" >&2
    exit 1
}

# --- preflight: served model id must match what the agent will request -----
WANT_MODEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["name"])' "$MODEL_CONFIG")"
SERVED="$(curl -sf "$BASE/v1/models" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(m["id"] for m in d.get("data",[])))' 2>/dev/null || true)"
echo "[tb] served models: ${SERVED:-<none>}"
if [[ -n "$SERVED" && " $SERVED " != *" $WANT_MODEL "* ]]; then
    echo "[tb] FATAL: agent will request model '$WANT_MODEL' but the server serves: $SERVED" >&2
    echo "[tb]   Use the SAME MODEL_CONFIG on both sides, or set --served-model-name on the GPU host." >&2
    exit 1
fi

# --- instance-data: materialize from the vendored TB task tree -------------
DATASET="$RESULTS/instance_data.json"
mapfile -t TASK_IDS < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1])).get("instances") or []))' "$APP")
[[ ${#TASK_IDS[@]} -gt 0 ]] || { echo "[tb] FATAL: no instances in $APP" >&2; exit 1; }
echo "[tb] materializing ${#TASK_IDS[@]} task(s): ${TASK_IDS[*]}"
PYTHONPATH="$KIT_ROOT" python3 -m agent_bench.benchmarks.terminal_bench.loader "$DATASET" "${TASK_IDS[@]}"

# --- pull task docker images ----------------------------------------------
if [[ "${SKIP_PULL:-0}" != "1" ]]; then
    mapfile -t IMAGES < <(python3 -c 'import json,sys; print("\n".join(sorted({r["docker_image"] for r in json.load(open(sys.argv[1]))})))' "$DATASET")
    for img in "${IMAGES[@]}"; do
        echo "[tb] docker pull $img"
        docker pull "$img" || { echo "[tb] FATAL: docker pull failed for $img" >&2; exit 1; }
    done
fi

# --- engine metrics tailer (vLLM engine counters) --------------------------
python3 "$KIT_ROOT/agent_bench/serving/engine_log_tailer.py" --engine vllm \
    --base-url "$BASE" --interval 1.0 \
    --output "$LOG_DIR/engine_metrics.jsonl" >"$LOG_DIR/tailer.log" 2>&1 &
TAILER_PID=$!
cleanup() { [[ -n "${TAILER_PID:-}" ]] && kill "$TAILER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# --- run the driver --------------------------------------------------------
echo "[tb] launching driver ($ARM arm)..."
PYTHONPATH="$KIT_ROOT" python3 -m agent_bench.runtime.driver \
    --benchmark-config "$APP" \
    --harness-config "$HARNESS_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --serving-config "$SERVING_CONFIG" \
    --benchmark-instance-data "$DATASET" \
    2>&1 | tee "$LOG_DIR/run.log"
RUN_RC=${PIPESTATUS[0]}
echo "[tb] driver exit rc=$RUN_RC"
kill "$TAILER_PID" 2>/dev/null || true
echo "[tb] results in $RESULTS"
exit "$RUN_RC"
