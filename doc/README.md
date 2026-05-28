# Documentation

Use these guides from top to bottom when getting oriented.

## Users

| Guide | Purpose |
|---|---|
| [quickstart.md](quickstart.md) | Run OpenClaw-based examples through client-only, local Docker, NSCC PBS, and CCDS cloud deployments. |
| [deployments.md](deployments.md) | Understand deployment adapters and how each environment launches serving and benchmark execution. |
| [execution.md](execution.md) | Inspect outputs, telemetry, scoring behavior, and troubleshooting steps. |
| [layout.md](layout.md) | Understand the repository layout after the refactor. |

## Developers

| Guide | Purpose |
|---|---|
| [benchmarks.md](benchmarks.md) | Add a new benchmark/task suite. |
| [conditions.md](conditions.md) | Add a new task-attempt procedure. |
| [harnesses.md](harnesses.md) | Add a new agent runtime or extend an existing one. |
| [exec-bridge.md](exec-bridge.md) | Understand the OpenClaw execution bridge. |
| [clean-break-refactor.md](clean-break-refactor.md) | Architecture overview for the hard-break package/profile layout. |

## Public Interface

The user-facing commands are:

```bash
python3 validate.py --config experiments/<name>.toml
python3 run.py --config experiments/<name>.toml
python3 run.py --config experiments/<name>.toml --render-job
python3 analyze.py --run runs/<run_id>
```

Most users should edit only `experiments/` and `profiles/`.
