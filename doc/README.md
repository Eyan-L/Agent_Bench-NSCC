# Development docs

Detailed documentation for using and extending the framework. Pick the guide
that matches what you're trying to do.

## Using the framework

| Guide | What's in it |
|---|---|
| [usage.md](usage.md)             | How to run a benchmark (local + PBS), inspect outputs, replay analysis, troubleshoot. |
| [repo_layout.md](repo_layout.md) | What each top-level folder is for and where to find what. |

## Extending the framework

| Guide | When to use |
|---|---|
| [adding_a_benchmark.md](adding_a_benchmark.md) | New evaluation suite (the *what to test*) — e.g. SWE-bench Lite, HumanEval, a custom internal task set. |
| [adding_a_condition.md](adding_a_condition.md) | New way the agent attempts a task (the *how* the model is invoked) — e.g. a different retrieval strategy, an ablation, a tool-use variant. |
| [adding_a_frontend.md](adding_a_frontend.md)   | New agent runtime image / orchestrator (the *who runs the agent*) — e.g. swap OpenClaw for SWE-Agent, claude-code, or a custom CLI. |

## Quick orientation

The three extension axes — benchmark, condition, frontend — are *orthogonal*.
A new benchmark is a new task set scored against any condition. A new
condition runs against any benchmark its harness supports. A new frontend
changes how the agent is launched but not what it's solving. When in doubt
about which axis a change lives on, check which subfolder of `eval/` the
change *most* mutates; that's usually the right axis.
