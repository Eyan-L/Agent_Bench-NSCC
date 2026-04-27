# bench-platform

Measurement platform for **trajectory-centric agentic workflows** vs
**request-centric autoregressive (AR) serving**. Provides the data and
infrastructure needed to answer three research questions about how agent
workloads differ from traditional LLM serving.

## Motivation

LLM serving today is studied through a request/turn lens — TTFT, TPOT,
throughput per token. That lens was built around chatbot-shaped traces and
assumes the workload is GPU-bound, request-organized, and adequately
summarized by token-level metrics.

Agentic workflows break all three assumptions:

- They are **trajectory-organized**, not request-organized. One task spans
  many LLM calls, tool invocations, retries, and stalls.
- Their **resource shape is not purely GPU-centric**. Tool execution, CPU
  work, dependency stalls, and retried attempts move bottlenecks off the
  accelerator and inflate effective resource cost.
- Their **success criterion is task-level**, not token-level. TTFT and TPOT
  improvements do not necessarily translate to better task completion.

This platform exists to measure those gaps systematically, on a shared
serving backend, across multiple workload classes (code agent, web search,
document analysis).

## Research questions

### RQ1. How does a trajectory-based workflow change the **execution composition** compared with traditional AR serving?

- Aggregate step-level traces into the right unit:
  - **Trajectory** — one full workflow instance for a task.
  - **Attempt** — one individual try within a trajectory.
- Compare execution composition (LLM vs tool vs waiting/stall) between AR
  and agentic workflow, and across the three workload classes.
- Characterize input/output token ratios per workload (prefill- vs
  decode-heavy).

Reference metrics: step type breakdown, tool-call ratio, non-LLM step
fraction, waiting/stall fraction, input/output token ratio.

### RQ2. Do these structural differences change the **resource bottlenecks** of the workload?

- Compare CPU / GPU / memory / PCIe / network composition between AR and
  agentic workflow.
- Measure how dependency stalls and fragmented execution reduce hardware
  utilization.
- Quantify resource cost of retries and repeated attempts.
- Compare whether code agent, web search, and document analysis exhibit
  distinct bottleneck patterns.

Reference metrics: GPU busy ratio, HBM bandwidth utilization, CPU
utilization, DRAM usage, PCIe traffic, external wait ratio, end-to-end
latency, wasted-work ratio, compaction/fragmentation score.

### RQ3. Are **token-centric metrics** still sufficient for evaluating agentic workflow?

- Report TTFT / TPOT per LLM step (token-level baseline).
- Report end-to-end task latency and completion rate (task-level baseline).
- Test whether token-level wins agree with task-level wins.

Reference metrics: TTFT, TPOT, end-to-end task latency, task completion
rate, p95 / p99 task latency.

## How the platform answers them

| RQ  | Data source in this repo                                                          |
|-----|-----------------------------------------------------------------------------------|
| RQ1 | `eval/` produces full trajectories (step trace + attempts); `stress/` produces matched AR baselines from synthetic / replayed request traces. Aggregation in `eval/analysis/`. |
| RQ2 | `eval/trace/telemetry.py` collects GPU / CPU / memory samples aligned to step boundaries; `analysis_tools/draw/` plots resource composition. `stress/` provides isolated load curves for the AR baseline. |
| RQ3 | `eval/runner/` records both token-level metrics (per LLM step) and task-level outcomes (verdict, end-to-end latency). Same model + same backend on both pipelines makes the comparison apples-to-apples. |

## Expected outcomes

1. A workload model that distinguishes request-centric AR from
   trajectory-centric agentic workflow.
2. An execution-composition comparison (LLM / tool / waiting).
3. A resource-bottleneck comparison showing where agentic workflow shifts
   load off the GPU.
4. An evaluation comparison demonstrating where task-level metrics diverge
   from TTFT / TPOT.

## Two pipelines, one backend

- **`eval/`** — real end-to-end agent evaluation. Currently: code agent via
  SWE-bench + OpenClaw. Outputs `candidate.patch` + step trajectories +
  pass/fail. Bench-type interfaces (`conditions/_base.py`) are pluggable;
  `doc_analysis/` and `websearch/` slots wired but unfilled.
- **`stress/`** — simulated trace replay (Go loadgen) against the same
  backend. Three workload classes (code_agent, doc_analysis, websearch)
  replayed from recorded or synthetic traces. Outputs latency / throughput
  / GPU metrics + plots.

## Bench-type support matrix

| Bench type    | eval (real) | stress (simulated trace) |
|---------------|-------------|--------------------------|
| code_agent    | ✅ SWE-bench + OpenClaw | ✅ |
| doc_analysis  | ❌ TODO (interface placeholder) | ✅ |
| websearch     | ❌ TODO (interface placeholder) | ✅ |

`eval` placeholders live at `eval/runner/conditions/{doc_analysis,websearch}/`.
Implement `ConditionProtocol` from `eval/runner/conditions/_base.py` to fill in.

## Backends

Both pipelines speak OpenAI-compatible HTTP. Choose backend with
`SERVING_BACKEND`:

- `vllm` — `backend/vllm/serve.sh`
- `sglang` — `backend/sglang/serve.sh` (ships local patch overlay; see
  `backend/sglang/patch/` for the 6-file diff vs upstream)

Dispatcher: `bash backend/common/serve.sh` selects launcher by env.

## Layout

```
bench-platform/
├── backend/
│   ├── common/{serve.sh, endpoint.py}
│   ├── vllm/{serve.sh, defaults.env}
│   └── sglang/{serve.sh, apply_patch.sh, defaults.env, Dockerfile, patch/{VERSION, MANIFEST.yaml, sglang/...}}
├── eval/
│   ├── runner/{driver.py, harness.py, config.py, fairness_check.py, manifest.py,
│   │           conditions/{_base.py, code_agent/, doc_analysis/, websearch/, ablations/}}
│   ├── frontend/openclaw/{apply_patch.py, configure.sh, gateway_*.sh, defaults.env,
│   │                       patch/{VERSION, MANIFEST.yaml, *.patch}}
│   ├── trace/{schema, logger, writers, telemetry, llm_wrapper, align,
│   │           openclaw_hooks, otel/{otel_collector_*.sh, otel_query.py, otel_reset_traces.sh}}
│   ├── analysis/, report/
├── stress/
│   ├── loadgen/             # Go module (cmd, client, sampling, result, loadgen-bin)
│   ├── experiment/          # Go orchestration (sglang lifecycle + loadgen + plots)
│   ├── scripts/             # replay_*.sh, run_*_trace_experiment.sh, pbs_experiment.pbs
│   └── workloads/{code_agent, doc_analysis, websearch}/   # READMEs only
├── configs/
│   ├── trace_thresholds.yaml
│   ├── eval/, stress/, sglang/, vllm/
├── tests/trace/test_trace_pipeline.py
├── analysis_tools/draw/     # plotting (TTFT/TBT/GPU)
├── utils/                   # synthetic trace generators + extract_files
├── skills/, prompts/        # Claude pipeline definitions
├── scripts/                 # ops only: check_env, status, prefetch_model
├── Makefile
└── .env.example
```

## Run

```bash
cp .env.example .env && $EDITOR .env
set -a && . .env && set +a

# 1. start backend (background)
make serve &

# 2a. real eval
make eval EVAL_ARGS="--sweep configs/eval/your_sweep.yaml"

# 2b. or simulated stress
make stress STRESS_ARGS="agent --config configs/stress/agent.json"

# 3. aggregate / report
make analyze
make report
```

## Tests

```bash
make test     # trace pipeline self-checks (thresholds in configs/trace_thresholds.yaml)
```

## Pending migration items (Phase 8 follow-up)

- Sglang patch deploy mode (`bind`) currently expects bind-mount: validate
  path under Docker (`SGLANG_SITE_PACKAGES` auto-detect).
- OpenClaw patch pinned to `2026.4.22` (commit `00bd2cf`); re-pin and
  refresh `patch/*.patch` whenever upstream releases bundle hashes change.
```
