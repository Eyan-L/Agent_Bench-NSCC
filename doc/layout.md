# Repo Layout

What each top-level folder is for after the hard-break refactor.

```
Agent_Bench/
├── agent_bench/       implementation package
├── experiments/       runnable experiment TOMLs
├── profiles/          reusable benchmark, harness, model, serving, deployment profiles
├── data/              generated datasets, run artifacts, figures, and local caches
├── scripts/           small host utilities that are still invoked directly
├── doc/               development and user documentation
└── tests/             repository tests
```

## Public Surface

The open-source-facing run interface is:

```bash
python3 validate.py --config experiments/client_only.toml
python3 run.py --config experiments/local_vllm.toml
python3 run.py --config experiments/nscc_vllm.toml --render-job
python3 analyze.py --run runs/<run_id>
```

Users should normally edit only `experiments/` and `profiles/`.

## `agent_bench/`

The package contains the moved implementation.

```
agent_bench/
├── core/          experiment/profile parsing, validation, run IDs, manifests
├── runtime/       main driver, dispatch contract, exec bridge socket protocol
├── deployments/  local/client/PBS/cloud adapters and container helpers
├── benchmarks/   SWE-bench, Terminal-Bench, DeepResearch, LongBench
├── harnesses/     OpenClaw, OpenHands, Hermes, ReAct, AR runtime assets
├── serving/      vLLM, SGLang, LiteLLM, endpoint helpers, engine telemetry
├── telemetry/    OTel helpers, trace processing, resource sampler
└── results/      analysis, aggregation, plots, reports
```

Benchmark-specific harness glue lives under
`agent_bench/benchmarks/<benchmark>/<harness>/`. Harness-wide runtime assets
live under `agent_bench/harnesses/<harness>/`.

## `profiles/`

Profiles are reusable fragments referenced by experiment TOMLs.

```
profiles/
├── benchmarks/   task suites and sweep definitions
├── harnesses/    agent runtime behavior and condition lists
├── models/       model aliases, context windows, generation limits
├── serving/      vLLM, SGLang, LiteLLM, existing endpoint definitions
├── deployments/  client-only, local Docker, NSCC PBS, CCDS cloud
└── stress/       stress/loadgen presets
```

Experiment files compose these profiles into a complete run.

## `scripts/`

Only small host utilities remain here, such as config env emitters, model
prefetch helpers, and MIG management scripts. Benchmark execution scripts moved
under `agent_bench/deployments/`.

## Data And Outputs

`data/` is generated state: materialized benchmark datasets, run artifacts,
logs, figures, and local caches. Re-running the same experiment can append new
run directories; manifests decide whether a specific run is terminal.
