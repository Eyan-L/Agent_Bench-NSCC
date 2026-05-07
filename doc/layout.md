# Repo layout

What each top-level folder is for, why it exists separately from its
neighbors, and where to look for what.

```
Agent_Bench/
├── frontend/        agent runtime images + state (currently: openclaw)
├── backend/         serving stack (vllm, sglang) + engine telemetry tailer
├── config/          inputs: benchmark instance data, sweeps, stress presets
├── eval/            backend code: benchmarks, conditions, runner, analysis
├── data/            outputs: per-run artifacts, figures
├── scripts/         top-level CLIs: PBS jobs, host samplers
├── stress/          load-generation stack (Go) for serving-side stress tests
├── doc/             development guides (this folder)
├── prompts/         operator handoff notes for Claude
├── skills/          Claude pipeline definitions
├── tests/           pipeline self-checks
└── analysis_tools/  ad-hoc plotting (separate from the eval/analysis lib)
```

The repo separates **inputs** (`config/`), **runtime images + state**
(`frontend/`, `backend/`), **code** (`eval/`, `stress/`, `scripts/`,
`backend/`, `frontend/`), and **outputs** (`data/`). Each top-level folder
owns one concern.

---

## `frontend/`

Agent runtime images and the configuration they read at startup. Currently
holds OpenClaw only.

```
frontend/openclaw/
├── apply_patch.py     applies vendored .patch files to OC's bundled JS
├── apply_profile.sh   profile selection at deploy time
├── configure.sh       env / dep checks
├── gateway_up.sh      bring OC's gateway up for a run
├── gateway_down.sh    tear it down
├── defaults.env       defaults for the above
├── patch/             vendored unified diffs against OC's bundle
└── state/             OC reads this at startup
    ├── openclaw.json  gateway, plugins, providers, agent definitions
    └── agents/swebench/{AGENTS,SOUL,IDENTITY,USER,TOOLS,HEARTBEAT}.md
```

Why separate from `eval/`: the agent runtime is image-shaped (build, deploy,
start, stop), not eval-shaped (load instance, run, score). Keeping them
apart means a new frontend (e.g. SWE-Agent) drops in as
`frontend/<name>/` without disturbing eval logic. See
[adding_a_frontend.md](adding_a_frontend.md).

## `backend/`

Serving stack — vLLM + SGLang launchers + the engine log tailer.

```
backend/
├── engine_log_tailer.py    polls /metrics, writes per-iteration JSONL
├── common/{serve.sh, endpoint.py}    dispatcher
├── vllm/{serve.sh, defaults.env}
└── sglang/{serve.sh, apply_patch.sh, defaults.env, Dockerfile, patch/}
```

The engines speak OpenAI-compat HTTP at `http://localhost:8000`. Pick
backend with `SERVING_BACKEND=vllm|sglang`; `bash backend/common/serve.sh`
dispatches.

The tailer is co-located here (rather than under `scripts/`) because it's
inherently bound to the engines' Prometheus endpoint shape — see
[usage.md → Engine telemetry](usage.md#engine-telemetry).

## `config/`

Inputs to the eval and stress pipelines. Pure data — no executable code.

```
config/
├── benchmarks/
│   ├── swebench/verified.json                     SWE-bench Verified instances
│   └── terminal_bench/<task>/{task.toml, environment/, tests/}
├── sweeps/                                        what to evaluate, how
│   └── *.yaml
└── stress/                                        loadgen presets + thresholds
    ├── agent.json                                 code_agent workload
    ├── alibaba.json                               doc_analysis (GPT-4 trace shape)
    ├── azure.json                                 websearch (Azure trace shape)
    ├── alibaba_nscc_2a_singularity.json           NSCC 2A cluster overrides
    ├── agent_dirs.example.json                    code_agent dataset selector
    ├── trace_thresholds.yaml                      trace pipeline self-test thresholds
    └── sglang/                                    SGLang-specific variants
```

Splitting inputs out of `eval/` and `stress/` means swapping a sweep is a
config change, not a code change.

## `eval/`

The eval pipeline code. The largest folder by line count and the one with
the most internal structure.

```
eval/
├── benchmarks/
│   ├── swebench/{harness.py, loader.py}          SWE-bench adapter
│   └── terminal_bench/{harness.py, loader.py}    TB adapter
├── conditions/                                    how the agent attempts a task
│   ├── _base.py                                   ConditionProtocol
│   ├── _llm_client.py                             OpenAI-compat caller (shared)
│   ├── _validate_bridge.py                        workspace IPC for self-validate
│   ├── code_agent/                                code-/automation-flavored
│   │   ├── ar_issue_only.py                       SWE-bench AR (one-shot)
│   │   ├── ar_static_context.py                   SWE-bench AR + retrieval
│   │   ├── openclaw_agent.py                      SWE-bench agent loop
│   │   └── terminal_bench/                        TB-specific (subfolder
│   │       └── react.py                            because TB will likely grow
│   │                                               more variants)
│   ├── doc_analysis/                              placeholder (no impl yet)
│   ├── websearch/                                 placeholder (no impl yet)
│   └── ablations/no_retry.py
├── runner/                                        orchestration
│   ├── driver.py                                  the dispatch table + run loop
│   ├── config.py                                  RunConfig, sweep YAML loading
│   ├── manifest.py                                per-run state tracking
│   └── fairness_check.py
├── analysis/                                      post-run metric computation
│   ├── aggregate.py, composition.py, evaluation.py, ...
│   └── plots/
├── trace/                                         instrumentation primitives
│   ├── logger.py                                  step-trace JSONL writer
│   ├── llm_wrapper.py                             TTFT/TPOT capture
│   ├── telemetry.py                               GPU/CPU/RSS sampler
│   ├── align.py                                   monotonic ↔ wall-clock
│   └── otel/
├── otel/otel_query.py                             OTel post-hoc query helper
└── scoring_server/                                external validation (HTTP)
```

The four submodules implement orthogonal axes:

- **benchmarks** = which task suite (one folder per benchmark)
- **conditions** = how the agent tries (one folder per workload, then per condition)
- **runner** = wiring (driver dispatches benchmark × condition)
- **analysis / trace** = instrumentation and post-processing

See [adding_a_benchmark.md](adding_a_benchmark.md) and
[adding_a_condition.md](adding_a_condition.md) for the extension surfaces.

## `data/`

Outputs only. Per-run artifacts, aggregated tables, figures.

```
data/
├── runs/<sweep_id>/<run_id>/         per-run output (verdict, traces, spans)
├── runs_mock/                         synthetic runs for plot dev
├── figures/<sweep_id>/rq{1,2,3}/      rendered RQ figures
└── figures_mock/rq{1,2,3}/            figures from mock_data.py
```

`data/` is the output sink. Re-running a sweep with the same `sweep_id`
appends new run dirs (driver checks manifest.status before re-running an
already-terminal run).

## `scripts/`

Top-level CLIs. Things you `qsub` or run by hand.

```
scripts/
├── util_sampler.py                    host CPU/GPU/RSS/per-PID-tree sampler
├── execution_pbs/                     PBS submission scripts
│   ├── run_benchmark.pbs              full-stack SWE-bench run on cluster
│   ├── run_benchmark_tb.pbs           TB equivalent
│   ├── pull_images.{pbs,sh}           one-time SIF pull for SWE-bench instances
│   ├── otel_collector.sh              start/stop the OTel collector sidecar
│   ├── build_oc_overlay.sh            apply OC patches into a singularity overlay
│   ├── warm_oc_plugins.sh             prime the OC plugin runtime-deps cache
│   └── oc_plugin_cache_clean.sh       sweep partial / disabled plugin extracts
└── helper_pbs/                        small helper jobs
```

Why separate from `eval/runner/`: these scripts are **operations**, not
eval logic. They orchestrate stage-by-stage setup (install, pull, patch,
serve, run, teardown) on the cluster.

## `stress/`

Load-generation stack. Go module that replays request traces against the
serving backend without an agent loop.

```
stress/
├── loadgen/                  Go: cmd/, client/, sampling/, result/, loadgen-bin
├── experiment/               Go: orchestration (sglang lifecycle + loadgen + plots)
├── workloads/                workload defs: code_agent, doc_analysis, websearch
├── scripts/                  pbs_experiment.pbs and replay drivers
└── README.md
```

Used to produce the AR baseline for RQ1/RQ2 — same backend, same model, but
request-organized rather than trajectory-organized.

## `doc/`

Development docs (this folder). See `doc/README.md` for the index.

## `prompts/`, `skills/`

Claude-specific pipeline definitions. `prompts/` holds operator handoff notes
(e.g. `OPENCLAW_GATEWAY_HANDOFF.md`); `skills/` defines reusable Claude
pipelines for common tasks (instrumentation, report generation, trace
analysis). Outside the eval pipeline.

## `tests/`

Trace-pipeline self-checks. Thresholds in
`config/stress/trace_thresholds.yaml`. Run with `make test`.

## `analysis_tools/`

Ad-hoc plotting helpers. Separate from `eval/analysis/` because these are
exploratory scripts that don't necessarily round-trip through the analysis
library — useful for one-off TTFT/TBT/GPU plots when prototyping.
