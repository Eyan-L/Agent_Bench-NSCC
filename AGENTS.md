# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python benchmark platform for agent workloads. Core package code lives under `agent_bench/`: benchmark apps in `agent_bench/benchmarks/`, runtime orchestration in `agent_bench/runtime/`, container and deployment code in `agent_bench/deployments/`, serving backends in `agent_bench/serving/`, and telemetry utilities in `agent_bench/telemetry/`. Experiment configs live in `config/`. Tests live in `tests/`, docs in `docs/`, scripts in `scripts/`, and generated artifacts in `data/` and `logs/`.

## Build, Test, and Development Commands

- `cp .env.example .env && $EDITOR .env`: create local environment settings.
- `make help`: list supported targets and current composed config defaults.
- `make serve`: launch the configured vLLM or sglang backend.
- `make eval APP=swebench/phase0 HARNESS=openclaw_swebench SERVING=sglang/default`: run the composed benchmark driver. SWE-bench also requires `SWEBENCH_ROOT` and `INSTANCE_DATA`.
- `make run_phase0`, `make run_phase1`, `make run_phase2`: SWE-bench shortcuts.
- `make run_tb`: run the Terminal-Bench `kv_store` task.
- `make analyze` and `make report`: aggregate results and generate reports.
- `make test`: run trace pipeline self-checks.

## Coding Style & Naming Conventions

Use Python 3.10+ and dependencies from `pyproject.toml`. Follow existing style: 4-space indentation, type hints for public or nontrivial functions, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Prefer structured config parsing over ad hoc strings. Keep Make targets lowercase and JSON config names descriptive, for example `config/harness/openclaw_swebench.json`.

## Testing Guidelines

The current test suite uses plain Python assertions and pytest-compatible files under `tests/`. Name new tests `test_*.py` and place them near the feature area, such as `tests/trace/`. Run `make test` for the trace self-checks; for targeted work, run modules directly, for example `python -m tests.trace.test_trace_pipeline`. Add focused tests when changing trace schema, telemetry alignment, runner behavior, or config loading.

## Commit & Pull Request Guidelines

Git history uses concise Conventional Commits-style subjects such as `feat: ...`, `fix: ...`, `tests: ...`, `docs: ...`, and `refactor(swebench): ...`. Keep subjects specific and one line when possible. Pull requests should include the benchmark area touched, commands run, config files changed, and expected impact on outputs under `data/runs/` or `data/figures/`. Link related issues and include screenshots only for report or plot changes.

## Security & Configuration Tips

Do not commit `.env`, credentials, model tokens, or generated benchmark artifacts. Treat `data/`, `logs/`, container images, and backend patch outputs as local/runtime assets unless a change is explicitly part of the benchmark definition.
