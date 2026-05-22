# Execution

Two deployment modes — pick by environment. In **both**, the agent runtime
and benchmark task each run in their **own** isolated container/sif. The host
runs only vLLM/SGLang or another OpenAI-compatible serving backend.

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
SERVING_BACKEND=vllm bash backend/common/serve.sh &

# Verify
curl -s http://localhost:8000/v1/models | jq .
```

### Run a sweep

```bash
HARNESS_BACKEND=docker \
PYTHONPATH=. python3 -m orchestration.runner.driver \
    --app config/application/swebench/phase1.json \
    --harness config/harness/openclaw_swebench.json \
    --model config/model/qwen36_27b.json \
    --serving config/serving/vllm/default.json \
    --swebench-root "$SWEBENCH_ROOT" \
    --instance-data "$INSTANCE_DATA"
```

Or via Makefile:

```bash
make eval APP=swebench/phase1 HARNESS=openclaw_swebench MODEL_PROFILE=qwen36_27b SERVING=vllm/default
```

The Docker harness (`applications/swebench/app.py:SWEBenchHarness`)
pulls the per-instance image lazily and `docker run`s it. First run on a new
instance pulls 1-3 GiB; subsequent runs reuse the cached image.

### Single-instance debug loop

```bash
INSTANCE=$(jq -r 'keys[0]' "$INSTANCE_DATA")
HARNESS_BACKEND=docker PYTHONPATH=. \
    python3 -m orchestration.runner.driver \
      --app config/application/swebench/phase0.json \
      --harness config/harness/openclaw_swebench.json \
      --model config/model/qwen36_27b.json \
      --serving config/serving/vllm/default.json \
      --swebench-root "$SWEBENCH_ROOT" \
      --instance-data "$INSTANCE_DATA"
```

Inspect `data/experiments/<stamp>/runs/phase0/<run_id>/{verdict.json,steps.jsonl,spans.jsonl}`.

---

## Mode 2 — PBS / singularity (cluster)

Production sweeps on HPC. The PBS script orchestrates: pip install swebench,
pull instance SIFs, start vLLM, prepare the selected harness runtime SIF, start
the OTel collector, run the driver, tear everything down.

### Submit

```bash
# Default bench profile (4× A100, 24h walltime)
qsub scripts/execution_pbs/run_benchmark.pbs

# Override the app/harness/model/serving, skip stage 0 (assume pylib already populated)
qsub -v STAGE_SWEBENCH=0,APP=$KIT_ROOT/config/application/swebench/phase0.json \
    scripts/execution_pbs/run_benchmark.pbs

# Small local-model smoke with the four-axis configs
qsub -l select=1:ncpus=16:ngpus=1:mem=64gb \
    -v KIT_ROOT=$KIT_ROOT,APP=$KIT_ROOT/config/application/swebench/phase0.json,HARNESS_CONFIG=$KIT_ROOT/config/harness/openclaw_swebench.json,MODEL_CONFIG=$KIT_ROOT/config/model/qwen35_9b.json,SERVING_CONFIG=$KIT_ROOT/config/serving/vllm/nscc.json,VLLM_TP=1 \
    scripts/execution_pbs/run_benchmark.pbs

# Provider-backed model API through a local LiteLLM proxy (no GPU request)
qsub -l select=1:ncpus=16:mem=64gb \
    -v KIT_ROOT=$KIT_ROOT,APP=$KIT_ROOT/config/application/swebench/phase0.json,HARNESS_CONFIG=$KIT_ROOT/config/harness/openclaw_swebench.json,MODEL_CONFIG=$KIT_ROOT/config/model/minimax_m2_7.json,SERVING_CONFIG=$KIT_ROOT/config/serving/litellm/minimax_global.json,MINIMAX_API_KEY=$MINIMAX_API_KEY \
    scripts/execution_pbs/run_benchmark.pbs

# Terminal-Bench sweep
qsub scripts/execution_pbs/run_benchmark_tb.pbs
```

### Stages (each toggleable via `STAGE_*=0`)

| Stage | What it does |
|---|---|
| `STAGE_SWEBENCH` | pip-install swebench into a persistent scratch lib cache |
| `STAGE_PULL`     | pull SWE-bench instance .sif images (one-time, network needed) |
| `STAGE_SERVING`  | start the local serving sidecar: vLLM for local models, or LiteLLM proxy for provider-backed models |
| `STAGE_OC_CHECK` | prepare the selected harness runtime SIF: OpenClaw/OpenHands/Hermes all build locally on NSCC from a base/upstream SIF unless you explicitly set a pullable final `*_RUNTIME_DOCKER` |
| `STAGE_OTEL`     | start the local OTel collector that writes otel-traces.jsonl |
| `STAGE_RUN`      | invoke `orchestration/runner/driver` with the sweep |
| `STAGE_CLEANUP`  | kill the serving sidecar + OTel, free ports |

See the header comment in `scripts/execution_pbs/run_benchmark.pbs` for the
full env-var menu.

### Outputs on the cluster

Per-run dirs land in
`$KIT_ROOT/data/experiments/$AB_RUN_STAMP/runs/<sweep_id>/<run_id>/`.
Logs go to `$KIT_ROOT/data/experiments/$AB_RUN_STAMP/logs/`:

- `pbs.log` — top-level PBS job log
- `vllm.log` — engine stdout (parsed by `engine_log_tailer`)
- `litellm.log` — LiteLLM proxy stdout for `config/serving/litellm/*`
- `litellm_config.yaml` — per-job LiteLLM proxy config; readable only by the job user
- `engine_metrics.jsonl` — per-iteration scheduler state
- `engine_log_tailer.log` — tailer's own diagnostic output

### SWE-bench Scoring On NSCC

NSCC jobs cannot use Docker directly, so final SWE-bench scoring should be
run through the same per-instance `.sif` images used by the task sandbox.
`run_benchmark.pbs` exports `OC_SCORING_SIF_DIR="$SIF_DIR"` after Stage 1, so
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
python -m orchestration.scoring_server.main \
  --swebench-root /path/to/SWE-bench \
  --host 0.0.0.0 \
  --port 8765

# In the NSCC qsub environment:
qsub -v OC_SCORING_URL=http://<docker-host>:8765,... \
  scripts/execution_pbs/run_benchmark.pbs
```

Scoring precedence is: `OC_SCORING_SIF_DIR` first, then `OC_SCORING_URL`, then
local `swebench.harness.run_evaluation` in Docker-capable environments.

---

## Mode 3 — Bare host

Removed. Agent harnesses are always containerized; running OpenClaw,
OpenHands, or Hermes directly on the host is no longer supported. For quick
local iteration, use `HARNESS_BACKEND=docker` with a small app config such as
`config/application/swebench/phase0.json`.

---

## Common operations (any mode)

```bash
# Inspect a run's verdict
jq . data/experiments/<stamp>/runs/<sweep>/<run_id>/verdict.json

# Count outcomes across a sweep
find data/experiments/<stamp>/runs/<sweep> -name verdict.json -exec jq -r .status {} \; | sort | uniq -c

# Replay analysis without re-running anything
PYTHONPATH=. python3 -m orchestration.analysis.aggregate --runs-dir data/experiments/<stamp>/runs/<sweep> --out-dir data/analysis/<stamp>/<sweep>

# Render figures
PYTHONPATH=. python3 orchestration/analysis/plots/plot_all.py \
    --runs-dir data/experiments/<stamp>/runs/<sweep_id> --out-dir data/figures/<stamp>/<sweep_id>

# Mock figures (no real runs needed)
PYTHONPATH=. python3 orchestration/analysis/plots/plot_all.py --mock
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

`backend/engine_log_tailer.py` polls vLLM/SGLang's Prometheus `/metrics`
endpoint at iteration cadence (default 100ms) and writes per-iteration JSONL
with running/waiting reqs, KV cache utilization, and token throughput.

Started automatically by the PBS script. For Docker / host mode, run it
manually:

```bash
python3 backend/engine_log_tailer.py \
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
./stress/loadgen/loadgen-bin agent --config config/stress/agent.json
./stress/loadgen/loadgen-bin alibaba --config config/stress/alibaba.json
./stress/loadgen/loadgen-bin azure   --config config/stress/azure.json

# Cluster
qsub -v EXP_CONFIG=config/stress/alibaba_nscc_2a_singularity.json \
    stress/scripts/pbs_experiment.pbs
```

Outputs land in `stress/loadgen/result/`.

---

## Troubleshooting

| Symptom | First check |
|---|---|
| `spans.jsonl` has only one record (heartbeat) | OC patches not applied — see `harnesses/openclaw/patch/MANIFEST.json` markers and the patched runtime SIF build |
| `replayInvalid: true` in trajectory | Long-context formatting drift — slim the system prompt or raise per-call `max_tokens` |
| `ENOTEMPTY` on plugin extract | Run `bash scripts/execution_pbs/oc_plugin_cache_clean.sh $OC_PLUGIN_CACHE` |
| vLLM never reaches ready | `tail -50 $LOG_DIR/vllm.log` — usually plugin npm install on a cold cache (network needed) |
| `spans.jsonl` missing entirely | OTel collector didn't start — check `bash scripts/execution_pbs/otel_collector.sh start` and `$OC_OTEL_DATA_DIR/otelcol.log` |
| Docker mode: instance image pull fails | Pre-fetch with `docker pull swebench/sweb.eval.<arch>.<instance>:latest` from a network-equipped host |
| NSCC scoring tries Docker | Ensure Stage 1 pulled task SIFs and `OC_SCORING_SIF_DIR` points at that directory |
