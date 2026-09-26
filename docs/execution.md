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

## Execution Stages

Every run uses the shared Python stage coordinator. NSCC jobs are rendered as
short wrappers that call back into the same coordinator.

| Stage | Purpose |
|---|---|
| `prepare_environment` | create run/log/cache directories and export runtime env |
| `materialize_benchmark` | materialize benchmark instance data |
| `prepare_images` | pull task SIFs and prepare runtime SIFs |
| `start_serving` | start vLLM, SGLang, or LiteLLM sidecar |
| `start_otel` | start local OTel collector |
| `run_benchmark` | invoke the runtime driver |
| `cleanup` | terminate sidecars and collectors |

Stages can be controlled per experiment with a top-level `[stages]` table:

```toml
[stages]
start_otel = true
start_serving = true
prepare_images = false
```

Stage values must be booleans and stage names must match the table above.
Precedence is: built-in defaults, deployment profile `[deployment.stages]`,
experiment TOML `[stages]`, `AB_STAGE_<NAME>` environment variables, then
legacy compatibility variables such as `STAGE_OTEL`.

Override a stage at submission time with `AB_STAGE_<NAME>=0`, for example:

```bash
qsub -v AB_STAGE_PREPARE_IMAGES=0 nscc_vllm.pbs
```

`AB_TELEMETRY_MODE` controls how the OTel collector runs (`docker`, `host`, or
`off`). Use `[stages].start_otel` or `AB_STAGE_START_OTEL` as the canonical
on/off switch for the stage itself.

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

## Portable run analysis

The public analyzer is a read-only exporter and viewer for one run. It accepts
normal, failed, timed-out, partial, single-agent, and subagent executions
without applying paired-comparison selection:

```bash
python3 analyze.py \
  --run runs/<experiment>/<pbs-run>/runs/<benchmark>/<run-id> \
  --output-dir reports/analyze \
  --name run-id
```

The generated directory contains five artifact groups:

```text
01_durable_session_records/  complete conversation and durable session state
02_timeline_model_trace/     steps, spans, trajectory, token accounting, viewer
03_execution_evidence/       runtime/resource data and raw execution logs
04_final_workspace_score/    manifest, candidate, verifier result, run details
05_figures/                  report figures when available
```

`artifact-index.json` maps original paths to bundle-relative paths and records
the size, SHA-256, source kind, and export status of every artifact. Existing
files are not rewritten. If `spans_timeline.html` was not collected but valid
`spans.jsonl` exists, the analyzer uses the repository's existing span renderer
to create a viewer in the bundle. The report never changes the source run and
rejects output paths that overlap it.

For a Roadmap PBS job with only one experiment arm, use the single-arm analyzer.
Either arm accepts one or more PBS job IDs; no paired run is required:

```bash
python3 scripts/analyze_roadmap_runs.py <subagent-pbs-job...> \
  --arm subagent --name roadmap-subagent-analysis
python3 scripts/analyze_roadmap_runs.py <single-agent-pbs-job...> \
  --arm single-agent --name roadmap-single-agent-analysis
```

It writes the same five artifact groups, a report, per-task CSV, trajectory
index, and available single-arm figures below `reports/roadmap/<name>/`.
`--runs-root` defaults to `$SCRATCH/agent-bench/runs` on NSCC.

Roadmap paired comparison uses the same exporter:

```bash
python3 scripts/compare_roadmap_runs.py \
  --subagent <subagent-pbs-jobs...> \
  --single-agent <single-agent-pbs-jobs...> \
  --output-dir reports/roadmap \
  --name roadmap-comparison
```

Comparison selection and statistics are unchanged: duplicate precedence,
normal-completion pairing, and actual materialized-subagent requirements still
control formal metrics. The report records exclusion counts/reasons and links
each selected run to its portable details, conversation, span timeline,
resource samples, logs, candidate, and verifier output.

Measurement labels retain their source meaning:

- `steps.jsonl` is a coarse harness trace; `spans.jsonl` is per-call telemetry.
- logical trajectory tokens are not cumulative serving processed tokens.
- child intervals are lifecycle observations, not continuous computation.
- TTFB is not renamed to TTFT.
- inferred, synthesized, and censored values keep their original quality fields.

## SWE-bench Scoring

SWE-bench scoring uses the benchmark environment for the selected deployment.

- Local Docker can score through Docker-backed SWE-bench images.
- NSCC prefers the per-instance SIFs prepared by the PBS job.
- A remote Docker scoring service can be used as a fallback by setting
  `AB_SCORING_SERVICE_URL`.

Scoring precedence:

```text
AB_SCORING_IMAGE_DIR -> AB_SCORING_SERVICE_URL -> local Docker SWE-bench evaluation
```

## Telemetry

OpenClaw emits OTel spans for model usage and tool calls when the patched
runtime and diagnostics plugin are active. The shared `start_otel` stage starts
the collector when enabled, for both local and PBS runs. Set
`[stages].start_otel = true` in an experiment TOML to force it on, or use
`AB_STAGE_START_OTEL=0` for a one-off disable.

Engine telemetry is collected from serving metrics where available and written
as JSONL. LiteLLM provider fallback still records agent-side steps, but it does
not expose vLLM/SGLang scheduler metrics.

## Troubleshooting

| Symptom | First check |
|---|---|
| `validate.py` rejects an experiment | Confirm every referenced profile path exists and uses the current profile format. |
| vLLM never reaches ready | Tail the deployment log and verify model path, GPU visibility, and context length. |
| LiteLLM returns unauthorized | Confirm the upstream provider key env var named by the serving profile is exported. |
| `spans.jsonl` missing | Check OTel collector logs and OpenClaw runtime patch markers. |
| NSCC job requests a GPU for provider fallback | Use the CPU deployment profile, as in `experiments/nscc_litellm.toml`. |
| SWE-bench scoring tries Docker on NSCC | Confirm task SIFs were prepared and `AB_SCORING_IMAGE_DIR` is set in the job environment. |
| Generated PBS script looks too short | That is expected; it delegates execution back to `run.py`, which runs the shared stage coordinator. |
