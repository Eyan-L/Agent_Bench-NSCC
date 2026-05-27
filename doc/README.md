# Development docs

Detailed documentation for using and extending the framework. Pick the guide
that matches what you're trying to do.

## Using the framework

| Guide | What's in it |
|---|---|
| [execution.md](execution.md) | How to run a benchmark (Docker, PBS, host modes), inspect outputs, replay analysis, troubleshoot. |
| [layout.md](layout.md)       | What each top-level folder is for and where to find what. |
| [clean-break-refactor.md](clean-break-refactor.md) | New open-source-facing experiment/profile interface and migration map. |

## Extending the framework

| Guide | When to use |
|---|---|
| [benchmarks.md](benchmarks.md) | New application/dataset (the *what to test*) — e.g. SWE-bench Lite, HumanEval, a custom internal task set. Lives under `agent_bench/benchmarks/<name>/`. |
| [conditions.md](conditions.md) | New way the agent attempts a task (the *how* the model is invoked) — e.g. a different retrieval strategy, an ablation, a tool-use variant. |
| [harnesses.md](harnesses.md)   | New agent runtime / orchestrator (the *who runs the agent*) — e.g. add OpenHands alongside OpenClaw, or swap in claude-code. Lives under `agent_bench/harnesses/<name>/`. |

## Quick orientation

The repo's load-bearing abstraction is three layers:

- **agent_bench/benchmarks/** — *which app/dataset is being evaluated.* Owns instance
  prep, scoring, and per-(app x harness) glue (e.g. `agent_bench/benchmarks/swebench/openclaw/`).
- **agent_bench/harnesses/** — *which agent runtime drives the model.* Harness-wide shared
  parts only.
- **agent_bench/serving/** — *which inference engine serves the model.* vLLM or SGLang.

The three extension axes — application, condition, harness — are
orthogonal: a new application is a new task set scored against any
condition; a new condition runs against any application its conditions list
supports; a new harness adds a new runtime with its own per-app glue.
When in doubt about which axis a change lives on, see [layout.md](layout.md).
