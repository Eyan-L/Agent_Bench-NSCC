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

Two modes — `docker` and `singularity`. In both, the agent runtime and the
benchmark task each run in their **own** container (or .sif). The host runs
only the serving backend.

**Local Docker** (default) — needs a running Docker daemon and the
patched OpenClaw image:

```bash
bash harnesses/openclaw/docker/build.sh   # one-time: agent-bench/openclaw:<v>
make serve &                              # vLLM on the host
HARNESS_BACKEND=docker \
    make eval APP=swebench/phase1 HARNESS=openclaw_swebench MODEL_PROFILE=qwen36_27b SERVING=vllm/default
```

Per-run docker layout: a private docker network, a `task-<run_id>` container
(SWE-bench instance image), an `oc-<run_id>` container (`agent-bench/openclaw`
running `gateway`). Workspace is bind-mounted into both at `/testbed`.

**HPC cluster (PBS)** — the PBS script provisions everything itself
(vLLM, OC sif, OTel collector, sweep, teardown). Don't pre-start
serving; the job spins it up on the compute node.

```bash
qsub scripts/execution_pbs/run_benchmark.pbs
```

**Singularity** — analogous to docker but using `singularity instance`s for
the agent runtime and task .sif on the same private mount tree.

Then aggregate + render:

```bash
make analyze
make report
```

Outputs land in `data/experiments/<stamp>/runs/<sweep_id>/<run_id>/`
(per-run verdicts, trajectories, spans, telemetry). Engine/PBS logs live in
`data/experiments/<stamp>/logs/`.

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
