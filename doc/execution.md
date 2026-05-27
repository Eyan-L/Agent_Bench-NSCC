# Execution

Two deployment modes — pick by environment. In **both**, the agent runtime
and benchmark task each run in their **own** isolated container/sif. The host
runs only vLLM/SGLang or another OpenAI-compatible serving agent_bench.serving.

| Mode | When to use | Harness backend |
|---|---|---|
| **Docker** | Workstation / docker-equipped nodes | Agent runtime container + task container, shared workspace |
| **Singularity** | HPC nodes without docker daemon | Agent runtime .sif + task .sif, shared workspace |

Selected via `HARNESS_BACKEND={docker,singularity}`. Default for `make eval`
is `docker`. There is **no** host mode — openclaw is always containerized.

---

## Common prerequisites

```bash
cp .env.example .env && $EDITOR .env
set -a && . .env && set +a
```

Common to all modes:

- Linux, Python 3.11+.
- `swebench` package importable (PBS Stage 0 installs it into a scratch lib
  cache; for local Docker / host modes, `pip install swebench`).
- A serving backend reachable at `http://localhost:8000` (vLLM default;
  SGLang via `SERVING_BACKEND=sglang`).

Per-mode infrastructure (only the row matching your mode applies):

| Mode | Infra needed |
|---|---|
| Docker (local) | Docker daemon, ability to pull from `docker.io/swebench/...` |
| PBS / singularity (cluster) | `singularity` ≥ 3.7 on PATH, PBS scheduler, `qsub` |

Benchmark-specific:

- SWE-bench scoring: per-instance image — Docker pulls lazily in Docker mode;
  pre-pulled `.sif` files under `$OC_SCORING_SIF_DIR` for singularity mode.
- Terminal-Bench: per-task `.sif` images pre-pulled by `pull_images.sh`.

---

## Mode 1 — Docker (local workstation)

Best for iterating on conditions or debugging a single instance.

### Setup

```bash
# Start a serving backend in the background
make serve &
# or directly:
SERVING_BACKEND=vllm bash agent_bench/serving/common/serve.sh &

# Verify
curl -s http://localhost:8000/v1/models | jq .
```

### Run a sweep

```bash
HARNESS_BACKEND=docker \
PYTHONPATH=. python3 -m agent_bench.runtime.driver \
    --app profiles/benchmarks/swebench/phase1.json \
    --harness profiles/harnesses/openclaw_swebench.json \
    --model profiles/models/qwen36_27b.json \
    --serving profiles/serving/vllm/default.json \
    --swebench-root "$SWEBENCH_ROOT" \
    --instance-data "$INSTANCE_DATA"
```

Or via Makefile:

```bash
make eval APP=swebench/phase1 HARNESS=openclaw_swebench MODEL_PROFILE=qwen36_27b SERVING=vllm/default
```

The Docker harness (`agent_bench/benchmarks/swebench/app.py:SWEBenchHarness`)
pulls the per-instance image lazily and `docker run`s it. First run on a new
instance pulls 1-3 GiB; subsequent runs reuse the cached image.

### Single-instance debug loop

```bash
INSTANCE=$(jq -r 'keys[0]' "$INSTANCE_DATA")
HARNESS_BACKEND=docker PYTHONPATH=. \
    python3 -m agent_bench.runtime.driver \
      --app profiles/benchmarks/swebench/phase0.json \
      --harness profiles/harnesses/openclaw_swebench.json \
      --model profiles/models/qwen36_27b.json \
      --serving profiles/serving/vllm/default.json \
      --swebench-root "$SWEBENCH_ROOT" \
      --instance-data "$INSTANCE_DATA"
```

Inspect `data/experiments/<stamp>/runs/phase0/<run_id>/{verdict.json,steps.jsonl,spans.jsonl}`.

---

## Mode 2 — PBS / singularity (cluster)

Production sweeps on HPC. `run.py --render-job` emits a short PBS wrapper; the
worker calls `agent_bench.deployments.pbs.run_job`, which materializes data,
pulls task/runtime SIFs, starts serving, starts the OTel collector, runs the
driver, and tears sidecars down.

### Submit

```bash
# SWE-bench smoke on NSCC
python3 run.py --config experiments/nscc_vllm.toml --render-job > nscc_vllm.pbs
qsub nscc_vllm.pbs

# Terminal-Bench smoke on NSCC
python3 run.py --config experiments/nscc_terminal_bench.toml --render-job > nscc_terminal_bench.pbs
qsub nscc_terminal_bench.pbs
```

### Stages

| Stage | What it does |
|---|---|
| `prepare_environment` | create run/log/cache directories and export runtime env |
| `materialize_benchmark` | materialize SWE-bench or Terminal-Bench instance JSON |
| `prepare_images` | pull task SIFs and prepare OpenClaw/OpenHands/Hermes runtime SIFs |
| `start_serving` | start vLLM, SGLang, or LiteLLM sidecar |
| `start_otel` | start local OTel collector that writes `otel-traces.jsonl` |
| `run_benchmark` | invoke the migrated runtime driver |
| `cleanup` | stop sidecars and collectors |

Disable a stage with `qsub -v AB_STAGE_PREPARE_IMAGES=0 ...`. Legacy
submission names such as `STAGE_PULL=0`, `STAGE_TB_PULL=0`, `STAGE_SERVING=0`,
and `STAGE_RUN=0` are still accepted as compatibility overrides.

### Outputs on the cluster

Per-run dirs land under the experiment `output_dir` plus stable run id, for
example `runs/nscc_vllm/<run_id>/`. Logs go to `<run_dir>/logs/`:

- `pbs.log` — top-level PBS job log
- `vllm.log` — engine stdout (parsed by `engine_log_tailer`)
- `litellm.log` — LiteLLM proxy stdout for `profiles/serving/litellm/*`
- `litellm_config.yaml` — per-job LiteLLM proxy config; readable only by the job user
- `engine_metrics.jsonl` — per-iteration scheduler state
- `engine_log_tailer.log` — tailer's own diagnostic output

### SWE-bench Scoring On NSCC

NSCC jobs cannot use Docker directly, so final SWE-bench scoring should be
run through the same per-instance `.sif` images used by the task sandbox.
The PBS stage runner exports `OC_SCORING_SIF_DIR="$SIF_DIR"` after environment
preparation, so
OpenClaw/OpenHands/Hermes score patches by:

1. copying `/testbed` out of `<instance_id>.sif` into a temporary writable dir,
2. applying the agent patch inside the SIF with the SWE-bench apply commands,
3. running the generated SWE-bench eval script with that writable `/testbed`
   bind-mounted back into the SIF,
4. parsing the official `get_eval_report` output into `verdict.json`.

If `OC_SCORING_SIF_DIR` is unset, the harnesses can still delegate to a
Docker-equipped scoring host as a fallback:

```bash
# On a machine that can run Docker + SWE-bench:
python -m agent_bench.benchmarks.swebench.scoring_server.main \
  --swebench-root /path/to/SWE-bench \
  --host 0.0.0.0 \
  --port 8765

# In the NSCC qsub environment, pass the scoring host to the generated job:
qsub -v OC_SCORING_URL=http://<docker-host>:8765 nscc_vllm.pbs
```

Scoring precedence is: `OC_SCORING_SIF_DIR` first, then `OC_SCORING_URL`, then
local `swebench.harness.run_evaluation` in Docker-capable environments.

---

## Mode 3 — Bare host

Removed. Agent harnesses are always containerized; running OpenClaw,
OpenHands, or Hermes directly on the host is no longer supported. For quick
local iteration, use `HARNESS_BACKEND=docker` with a small app config such as
`profiles/benchmarks/swebench/phase0.json`.

---

## Common operations (any mode)

```bash
# Inspect a run's verdict
jq . data/experiments/<stamp>/runs/<sweep>/<run_id>/verdict.json

# Count outcomes across a sweep
find data/experiments/<stamp>/runs/<sweep> -name verdict.json -exec jq -r .status {} \; | sort | uniq -c

# Replay analysis without re-running anything
PYTHONPATH=. python3 -m agent_bench.results.analysis.aggregate --runs-dir data/experiments/<stamp>/runs/<sweep> --out-dir data/analysis/<stamp>/<sweep>

# Render figures
PYTHONPATH=. python3 agent_bench/results/analysis/plots/plot_all.py \
    --runs-dir data/experiments/<stamp>/runs/<sweep_id> --out-dir data/figures/<stamp>/<sweep_id>

# Mock figures (no real runs needed)
PYTHONPATH=. python3 agent_bench/results/analysis/plots/plot_all.py --mock
```

---

## Self-validation in the agent loop

Both OpenClaw and TB agents stop early once the candidate fix passes:

- **SWE-bench** (any mode) — the agent prompt instructs it to run
  `bash _oc_validate.sh` before DONE. The host bridge round-trips through
  the harness's `run_tests()` and prints a JSON verdict. Configured
  automatically.
- **Terminal-Bench** — the agent runs `bash /tests/test.sh && cat
  /logs/verifier/reward.txt` before DONE; reward `1` = stop.

Each validation invocation appends to `<run_dir>/validate_verdicts.jsonl`.

---

## Engine telemetry

`agent_bench/serving/engine_log_tailer.py` polls vLLM/SGLang's Prometheus `/metrics`
endpoint at iteration cadence (default 100ms) and writes per-iteration JSONL
with running/waiting reqs, KV cache utilization, and token throughput.

Started automatically by the PBS script. For Docker / host mode, run it
manually:

```bash
python3 agent_bench/serving/engine_log_tailer.py \
    --engine vllm \
    --base-url http://127.0.0.1:8000 \
    --interval 0.1 \
    --output engine_metrics.jsonl &
```

---

## Stress / loadgen (separate pipeline)

For serving-side characterization (no agent loop, just request replay):

```bash
make -C stress/loadgen                        # build the loadgen binary
./stress/loadgen/loadgen-bin agent --config profiles/stress/agent.json
./stress/loadgen/loadgen-bin alibaba --config profiles/stress/alibaba.json
./stress/loadgen/loadgen-bin azure   --config profiles/stress/azure.json

# Cluster
qsub -v EXP_CONFIG=profiles/stress/alibaba_nscc_2a_singularity.json \
    stress/scripts/pbs_experiment.pbs
```

Outputs land in `stress/loadgen/result/`.

---

## Troubleshooting

| Symptom | First check |
|---|---|
| `spans.jsonl` has only one record (heartbeat) | OC patches not applied — see `agent_bench/harnesses/openclaw/patch/MANIFEST.json` markers and the patched runtime SIF build |
| `replayInvalid: true` in trajectory | Long-context formatting drift — slim the system prompt or raise per-call `max_tokens` |
| `ENOTEMPTY` on plugin extract | Run `bash agent_bench/deployments/pbs/oc_plugin_cache_clean.sh $OC_PLUGIN_CACHE` |
| vLLM never reaches ready | `tail -50 $LOG_DIR/vllm.log` — usually plugin npm install on a cold cache (network needed) |
| `spans.jsonl` missing entirely | OTel collector didn't start — check `bash agent_bench/deployments/pbs/otel_collector.sh start` and `$OC_OTEL_DATA_DIR/otelcol.log` |
| Docker mode: instance image pull fails | Pre-fetch with `docker pull swebench/sweb.eval.<arch>.<instance>:latest` from a network-equipped host |
| NSCC scoring tries Docker | Ensure Stage 1 pulled task SIFs and `OC_SCORING_SIF_DIR` points at that directory |
