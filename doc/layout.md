# Repo layout

What each top-level folder is for, why it exists separately from its
neighbors, and where to look for what.

```
Agent_Bench/
├── applications/    Layer 1: application/dataset (swebench, terminal_bench, ...)
├── harnesses/       Layer 2: agent runtime (openclaw, future openhands, ...)
├── backend/         Layer 3: serving stack (vllm, sglang) + engine telemetry
├── config/          inputs: per-instance task data, sweeps, stress presets
├── eval/            orchestration: runner, conditions, scoring, analysis, trace
├── data/            outputs: per-run artifacts, figures
├── scripts/         top-level CLIs: PBS jobs, host samplers
├── stress/          load-generation stack (Go) for serving-side stress tests
├── doc/             development guides (this folder)
├── prompts/         operator handoff notes for Claude
├── skills/          Claude pipeline definitions (NOT OpenClaw skills — those live under applications/<app>/openclaw/skills/)
├── tests/           pipeline self-checks
└── analysis_tools/  ad-hoc plotting (separate from the orchestration/analysis lib)
```

## The three-layer abstraction

The top three dirs (`applications/`, `harnesses/`, `backend/`) are the load-bearing
abstraction:

- **applications/** — *which app/dataset is being evaluated.* Owns instance
  preparation, scoring, and **how the app wires up to each harness**. Per-(app ×
  harness) customization lives under `applications/<app>/<harness>/` (e.g.
  `applications/swebench/openclaw/`). Adding a new app/harness pair = one new
  directory.
- **harnesses/** — *which agent runtime drives the model.* Only harness-wide
  shared code lives here (gateway lifecycle scripts, patches, common config
  templates). OpenClaw is the only harness today.
- **backend/** — *which inference engine serves the model.* vLLM or SGLang;
  selection is `SERVING_BACKEND={vllm|sglang}`.

`eval/` is layer-agnostic glue: the run driver dispatches across (app, harness,
backend), the conditions implement how an agent attempts a task, the scoring
server provides remote evaluation, and analysis processes results.

---

## `applications/` (Layer 1)

```
applications/
├── conditions/                           cross-app condition base classes + utilities
│   ├── openclaw_agent.py                 OpenClawAgentCondition base (agent_id + prompt as hooks)
│   ├── _llm_client.py                    OpenAI-compat HTTP caller (shared by AR baselines + TB ReAct)
│   ├── util_sampler.py                   CPU/RSS/GPU sampler sidecar (shared by all condition paths)
│   └── gateway_helpers.py                gateway port pool + singularity argv wrapper (shared by both harnesses + OC condition)
├── swebench/
│   ├── __init__.py
│   ├── app.py                            SWEBenchHarness (two-container docker), SingularityHarness (stub)
│   ├── conditions/                       SWE-bench specific conditions
│   │   ├── ar_issue_only.py              one-shot AR baseline
│   │   ├── ar_static_context.py          one-shot + BM25 retrieval baseline
│   │   └── openclaw_agent.py             SWEBenchOpenClawCondition (subclass; agent_id=swebench, git-diff patch collection)
│   └── openclaw/                         OpenClaw customization for SWE-bench
│       ├── agents/{AGENTS,SOUL,...}.md   prompts mounted into /root/.openclaw/agents/swebench/
│       └── skills/verify/SKILL.md        the `verify` OpenClaw skill (agent runs `python -m pytest` in /testbed)
└── terminal_bench/
    ├── __init__.py, app.py, loader.py, README.md, tasks/
    ├── conditions/
    │   ├── react.py                      TerminalBenchReactCondition (no OpenClaw — direct docker exec)
    │   └── openclaw_agent.py             TerminalBenchOpenClawCondition (subclass; agent_id=terminal_bench, no patch)
    └── openclaw/                         OpenClaw customization for Terminal-Bench
        ├── agents/{AGENTS,SOUL,...}.md   prompts mounted into /root/.openclaw/agents/terminal_bench/
        ├── agents/BOOTSTRAP.md           seeded into /app at run start (silences OC's "[Bootstrap pending]" injection)
        └── skills/verify/{SKILL.md, _oc_validate.sh}
                                          the `verify` skill + helper; harness copies _oc_validate.sh into /app at run start
```

Adding a new application: see [benchmarks.md](benchmarks.md). The `<app>/openclaw/`
subdir is the documented per-(app × harness) slot.

## `harnesses/` (Layer 2)

```
harnesses/
└── openclaw/
    ├── apply_patch.py     applies vendored .patch files to OC's bundled JS
    ├── apply_profile.sh   profile selection at deploy time
    ├── configure.sh       env / dep checks
    ├── gateway_up.sh      bring OC's gateway up for a run
    ├── gateway_down.sh    tear it down
    ├── defaults.env       defaults for the above
    ├── patch/             vendored unified diffs against OC's bundle
    └── state/openclaw.json   the gateway/plugin/provider config template
```

Per-app prompts and skills do **not** live here — those live under
`applications/<app>/openclaw/`. A new harness drops in as `harnesses/<name>/`
plus per-app glue under `applications/<app>/<name>/`. See [harnesses.md](harnesses.md).

## `backend/` (Layer 3)

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
```

Splitting inputs out of code dirs means swapping a sweep is a config change,
not a code change.

## `eval/`

Orchestration: drives runs, implements conditions, scores SWE-bench
remotely, post-processes traces. Layer-agnostic.

```
eval/
├── runner/                                        orchestration
│   ├── driver.py                                  the dispatch table + run loop
│   ├── config.py                                  RunConfig, sweep YAML loading
│   ├── manifest.py                                per-run state tracking
│   └── fairness_check.py
│                                                  (no condition code under orchestration/ — all moved to applications/conditions/)
├── analysis/                                      post-run metric computation
├── trace/                                         instrumentation primitives
├── otel/                                          OTel post-hoc query helpers
└── scoring_server/                                external SWE-bench scoring (HTTP)
```

The driver dispatches on (application × condition × backend) without knowing
about OpenClaw specifics — those live in `applications/<app>/openclaw/`.

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

Why separate from `orchestration/runner/`: these scripts are **operations**, not
eval logic. They orchestrate stage-by-stage setup (install, pull, patch,
serve, run, teardown) on the cluster.

## `stress/`

Load-generation stack. Go module that replays request traces against the
serving backend without an agent loop.

```
stress/
├── loadgen/, experiment/, workloads/, scripts/
└── README.md
```

Used to produce the AR baseline for RQ1/RQ2 — same backend, same model, but
request-organized rather than trajectory-organized.

## `doc/`

Development docs (this folder). See `doc/README.md` for the index.

## `prompts/`, `skills/`

`prompts/` holds free-form operator handoff notes for Claude
(e.g. `OPENCLAW_GATEWAY_HANDOFF.md`); `skills/` defines reusable Claude
pipeline definitions. Note: these are **Claude pipeline skills**, not
OpenClaw skills — the latter live under `applications/<app>/openclaw/skills/`.
