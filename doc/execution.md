# Execution

Two deployment modes — pick by environment, not by preference. The
underlying eval pipeline is identical; only the harness backend changes.

| Mode | When to use | Harness backend |
|---|---|---|
| **Docker (local)** | Workstation development; iterating on a condition | `SWEBenchHarness` |
| **PBS / singularity (cluster)** | HPC nodes without docker daemon; batch sweeps | `SingularityHarness` |

Selected via `HARNESS_BACKEND={docker,singularity}`. Default in
`scripts/execution_pbs/run_benchmark.pbs` is `singularity`; default for
`make eval` / `make run_phase*` is `docker`. (Bare-host mode was removed —
openclaw must always be containerized.)

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

Via Makefile (preferred):

```bash
HARNESS_BACKEND=docker \
    make run_phase1 SWEBENCH_ROOT=/path/to/SWE-bench \
                    INSTANCE_DATA=/path/to/verified.json
```

Override the harness/backend JSONs from the CLI:

```bash
make run_phase1 HARNESS=openhands_27b BACKEND=vllm/tp2 \
                SWEBENCH_ROOT=... INSTANCE_DATA=...
```

The driver is `orchestration.runner.driver`, which composes three
configs: `--app config/application/swebench/<exp>.json`,
`--harness config/harness/<HARNESS>.json`,
`--backend config/backend/<BACKEND>.json`. The Makefile wires those for
you. Direct invocation:

```bash
HARNESS_BACKEND=docker \
PYTHONPATH=. python3 -m orchestration.runner.driver \
    --app config/application/swebench/phase1.json \
    --harness config/harness/openclaw_27b.json \
    --backend config/backend/vllm/tp1.json \
    --swebench-root /path/to/SWE-bench \
    --instance-data /path/to/verified.json
```

The Docker harness (`applications/swebench/app.py:SWEBenchHarness`)
pulls the per-instance image lazily and `docker run`s it. First run on a
new instance pulls 1-3 GiB; subsequent runs reuse the cached image.

### Single-instance debug loop

Drop a one-off application JSON next to `phase0.json`:

```bash
INSTANCE=astropy__astropy-12907   # any verified instance_id

cat > config/application/swebench/smoke.json <<EOF
{
  "sweep_name": "smoke",
  "plan_version": "v1.0",
  "application": "swebench",
  "instances": ["$INSTANCE"],
  "repo_mapping": {"$INSTANCE": {"repo_id": "astropy/astropy", "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607"}},
  "container_image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
  "wall_clock_budget_seconds": 1800,
  "output_dir": "data/runs/smoke",
  "seeds": [0],
  "parallel_limit_gpu": 1,
  "parallel_limit_cpu_cores": 8
}
EOF

HARNESS_BACKEND=docker PYTHONPATH=. python3 -m orchestration.runner.driver \
    --app config/application/swebench/smoke.json \
    --harness config/harness/openclaw_27b.json \
    --backend config/backend/vllm/tp1.json \
    --swebench-root /path/to/SWE-bench \
    --instance-data /path/to/verified.json
```

Inspect `data/runs/smoke/<run_id>/{verdict.json,step_trace.jsonl,spans.jsonl}`.

---

## Mode 2 — PBS / singularity (cluster)

Production sweeps on HPC. The PBS script orchestrates: pip install swebench,
pull instance SIFs, start vLLM, build OC patch overlay, start the OTel
collector, run the driver, tear everything down.

### Submit

```bash
# Default bench profile (4× A100, 24h walltime)
qsub scripts/execution_pbs/run_benchmark.pbs

# Override the application, skip stage 0 (assume pylib already populated)
qsub -v STAGE_SWEBENCH=0,APP=$KIT_ROOT/config/application/swebench/phase0.json \
    scripts/execution_pbs/run_benchmark.pbs

# Test profile — small model on 1 GPU, smokes the full pipeline cheaply
qsub -l select=1:ncpus=8:mem=32gb -v VLLM_PROFILE=test \
    scripts/execution_pbs/run_benchmark.pbs

# CPU-only smoke (no GPU node)
qsub -l select=1:ncpus=8:mem=32gb -v VLLM_PROFILE=test,VLLM_DEVICE=cpu \
    scripts/execution_pbs/run_benchmark.pbs

# Terminal-Bench sweep
qsub scripts/execution_pbs/run_benchmark_tb.pbs
```

### Stages (each toggleable via `STAGE_*=0`)

| Stage | What it does |
|---|---|
| `STAGE_SWEBENCH` | pip-install swebench into a persistent scratch lib cache |
| `STAGE_PULL`     | pull SWE-bench instance .sif images (one-time, network needed) |
| `STAGE_VLLM`     | start vLLM in singularity, serve the model |
| `STAGE_OC_CHECK` | pull alpine/openclaw sif + bake patch overlay |
| `STAGE_OTEL`     | start the local OTel collector that writes otel-traces.jsonl |
| `STAGE_RUN`      | invoke `orchestration/runner/driver` with the composed configs |
| `STAGE_CLEANUP`  | kill vLLM + OTel, free ports |

See the header comment in `scripts/execution_pbs/run_benchmark.pbs` for the
full env-var menu.

### Outputs on the cluster

Per-run dirs land in `$KIT_ROOT/data/runs/<sweep_id>/<run_id>/`. Logs go to
`$KIT_ROOT/logs/`:

- `pbs.log` — top-level PBS job log
- `vllm.log` — engine stdout (parsed by `engine_log_tailer`)
- `engine_metrics.jsonl` — per-iteration scheduler state
- `engine_log_tailer.log` — tailer's own diagnostic output

---

## Scoring offload (HTTP scoring server)

For singularity / restricted-network nodes that can't run the full
SWE-bench scoring image themselves, run the scoring server on a docker
host and point sweeps at it:

```bash
# On the docker host
make scoring-server SWEBENCH_ROOT=/path/to/SWE-bench INSTANCE_DATA=/path/to/verified.json

# On the sweep node
export OC_SCORING_URL=http://<docker-host>:8765
```

The reference implementation is `orchestration/scoring_server/main.py`.

---

## Common operations (any mode)

```bash
# Inspect a run's verdict
jq . data/runs/<sweep>/<run_id>/verdict.json

# Count outcomes across a sweep
find data/runs/<sweep> -name verdict.json -exec jq -r .status {} \; | sort | uniq -c

# Replay analysis without re-running anything
PYTHONPATH=. python3 -m orchestration.analysis.aggregate --runs data/runs/<sweep>

# Render figures
PYTHONPATH=. python3 orchestration/analysis/plots/plot_all.py \
    --runs data/runs/<sweep_id> --out data/figures/<sweep_id>

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
| `spans.jsonl` has only one record (heartbeat) | OC patches not applied — see `frontend/openclaw/patch/MANIFEST.json` markers and the patch overlay build |
| `replayInvalid: true` in trajectory | Long-context formatting drift — slim the system prompt or raise per-call `max_tokens` |
| `ENOTEMPTY` on plugin extract | Run `bash scripts/execution_pbs/oc_plugin_cache_clean.sh $OC_PLUGIN_CACHE` |
| vLLM never reaches ready | `tail -50 $LOG_DIR/vllm.log` — usually plugin npm install on a cold cache (network needed) |
| `spans.jsonl` missing entirely | OTel collector didn't start — check `bash scripts/execution_pbs/otel_collector.sh start` and `$OC_OTEL_DATA_DIR/otelcol.log` |
| Docker mode: instance image pull fails | Pre-fetch with `docker pull swebench/sweb.eval.<arch>.<instance>:latest` from a network-equipped host |
| Host mode: scoring writes "DEFERRED" | Set `OC_SCORING_SIF_DIR` or `OC_SCORING_URL` — host mode has no built-in scorer |
