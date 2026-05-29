# Execution Operations

Use [quickstart.md](quickstart.md) for first-run commands. This page covers
runtime behavior, outputs, telemetry, scoring, and troubleshooting once an
experiment is running.

## Execution Flow

Every public run starts from an experiment TOML:

```bash
python3 run.py --config experiments/<name>.toml
```

For batch or cloud deployments:

```bash
python3 run.py --config experiments/<name>.toml --render-job
```

The resolved experiment selects benchmark, harness, model, serving, and
deployment profiles. The runtime driver then creates one run per selected
instance and seed.

## Container Model

Agent harnesses are containerized. OpenClaw, OpenHands, and Hermes do not run
directly on the host as production harnesses.

- Local development uses Docker task and harness containers.
- NSCC uses Singularity SIFs for task and harness runtimes.
- Client-only changes serving behavior only; it does not mean the harness runs
  without isolation.

## PBS Stages

NSCC jobs are rendered as short wrappers around the Python PBS stage runner.

| Stage | Purpose |
|---|---|
| `prepare_environment` | create run/log/cache directories and export runtime env |
| `materialize_benchmark` | materialize benchmark instance data |
| `prepare_images` | pull task SIFs and prepare runtime SIFs |
| `start_serving` | start vLLM, SGLang, or LiteLLM sidecar |
| `start_otel` | start local OTel collector |
| `run_benchmark` | invoke the runtime driver |
| `cleanup` | terminate sidecars and collectors |

Disable a stage at submission time with `AB_STAGE_<NAME>=0`, for example:

```bash
qsub -v AB_STAGE_PREPARE_IMAGES=0 nscc_vllm.pbs
```

Legacy stage names are accepted only as compatibility overrides.

## Outputs

Run products land under the experiment output directory, typically:

```text
runs/<experiment>/<run_id>/
```

Common files:

- `manifest.json`: run metadata and terminal state
- `candidate.patch`: patch artifact when the benchmark uses patches
- `verdict.json`: benchmark scoring result
- `steps.jsonl`: LLM/tool step trace
- `spans.jsonl`: OpenTelemetry-derived spans when available
- `util.jsonl`: CPU/RSS/GPU samples when enabled
- `logs/`: serving, OTel, PBS, and sidecar logs

Useful commands:

```bash
jq . runs/<experiment>/<run_id>/verdict.json
find runs/<experiment> -name verdict.json -exec jq -r '.status // "unknown"' {} \; | sort | uniq -c
python3 analyze.py --run runs/<experiment>/<run_id>
```

## SWE-bench Scoring

SWE-bench scoring uses the benchmark environment for the selected deployment.

- Local Docker can score through Docker-backed SWE-bench images.
- NSCC prefers the per-instance SIFs prepared by the PBS job.
- A remote Docker scoring service can be used as a fallback by setting
  `OC_SCORING_URL`.

Scoring precedence:

```text
OC_SCORING_SIF_DIR -> OC_SCORING_URL -> local Docker SWE-bench evaluation
```

## Telemetry

OpenClaw emits OTel spans for model usage and tool calls when the patched
runtime and diagnostics plugin are active. The PBS stage runner starts the OTel
collector automatically. Local runs can start it through the same helper used by
PBS if needed:

```bash
bash agent_bench/deployments/pbs/otel_collector.sh start
```

Engine telemetry is collected from serving metrics where available and written
as JSONL. LiteLLM provider fallback still records agent-side steps, but it does
not expose vLLM/SGLang scheduler metrics.

## Analysis

Aggregate and plotting code lives under `agent_bench/results/`.

```bash
python3 -m agent_bench.results.analysis.aggregate \
  --runs-dir runs/<experiment> \
  --out-dir data/analysis/<experiment>

python3 agent_bench/results/analysis/plots/plot_all.py \
  --runs-dir runs/<experiment> \
  --out-dir data/figures/<experiment>
```

## Troubleshooting

| Symptom | First check |
|---|---|
| `validate.py` rejects an experiment | Confirm every referenced profile path exists and uses the current profile format. |
| vLLM never reaches ready | Tail the deployment log and verify model path, GPU visibility, and context length. |
| LiteLLM returns unauthorized | Confirm the upstream provider key env var named by the serving profile is exported. |
| `spans.jsonl` missing | Check OTel collector logs and OpenClaw runtime patch markers. |
| NSCC job requests a GPU for provider fallback | Use the CPU deployment profile, as in `experiments/nscc_litellm.toml`. |
| SWE-bench scoring tries Docker on NSCC | Confirm task SIFs were prepared and `OC_SCORING_SIF_DIR` is set in the job environment. |
| Generated PBS script looks too short | That is expected; it delegates execution back to `run.py`, which runs the shared stage coordinator. |
