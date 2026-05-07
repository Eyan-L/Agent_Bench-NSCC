# bench-platform

Measurement platform for **trajectory-centric agentic workflows** vs
**request-centric autoregressive (AR) serving**. Provides the data and
infrastructure to compare how agent workloads differ from traditional LLM
serving across three axes — execution composition, resource bottlenecks,
and the adequacy of token-level metrics.

## Motivation

LLM serving today is studied through a request/turn lens — TTFT, TPOT,
throughput per token. That lens was built around chatbot-shaped traces
and assumes the workload is GPU-bound, request-organized, and adequately
summarized by token-level metrics.

Agentic workflows break all three assumptions:

- They are **trajectory-organized**, not request-organized. One task spans
  many LLM calls, tool invocations, retries, and stalls.
- Their **resource shape is not purely GPU-centric**. Tool execution,
  CPU work, dependency stalls, and retried attempts move bottlenecks off
  the accelerator and inflate effective resource cost.
- Their **success criterion is task-level**, not token-level. TTFT and
  TPOT improvements do not necessarily translate to better task
  completion.

This platform exists to measure those gaps systematically, on a shared
serving backend, across multiple workload classes (code agent, web
search, document analysis).

## Quick start

```bash
cp .env.example .env && $EDITOR .env
set -a && . .env && set +a
```

Pick the mode that matches your environment.

**Local Docker** — needs a running Docker daemon (per-instance SWE-bench
images are pulled lazily on first use, ~1–3 GiB each). Start a serving
backend, then run a sweep:

```bash
make serve &                                                 # vLLM/SGLang in the background
HARNESS_BACKEND=docker \
    make eval EVAL_ARGS="--sweep config/sweeps/phase1.yaml"
```

**HPC cluster (PBS)** — the PBS script provisions everything itself
(vLLM, OC overlay, OTel collector, sweep, teardown). Don't pre-start
serving; the job spins it up on the compute node.

```bash
qsub scripts/execution_pbs/run_benchmark.pbs
```

**Bare host (no container)** — start a serving backend, then run with
`HARNESS_BACKEND=host` (scoring is deferred unless `OC_SCORING_SIF_DIR`
or `OC_SCORING_URL` is set):

```bash
make serve &
HARNESS_BACKEND=host PYTHONPATH=. python3 -m eval.runner.driver \
    --sweep config/sweeps/phase1.yaml --max-runs 1
```

Then aggregate + render:

```bash
make analyze
make report
```

Outputs land in `data/runs/<sweep_id>/<run_id>/` (per-run verdicts,
trajectories, spans, telemetry) and `data/figures/<sweep_id>/`.

## Tests

```bash
make test     # trace pipeline self-checks
```

## Documentation

Detailed guides live under [`doc/`](doc/):

- [doc/execution.md](doc/execution.md) — running benchmarks (Docker, PBS, host modes), inputs/outputs, common operations, troubleshooting.
- [doc/layout.md](doc/layout.md) — what each top-level folder is for.
- [doc/benchmarks.md](doc/benchmarks.md) — adding a new benchmark.
- [doc/conditions.md](doc/conditions.md) — adding a new condition.
- [doc/frontends.md](doc/frontends.md) — adding a new agent runtime / frontend.
