# Agent Bench Architecture Surface

This document describes the open-source-facing shape of Agent Bench after the
hard-break refactor. It is an architecture overview; use
[quickstart.md](quickstart.md) for current run commands.

## Five Public Nouns

- **Benchmark**: the task suite and scorer, such as SWE-bench or Terminal-Bench.
- **Harness**: the agent runtime adapter, such as OpenClaw, OpenHands, Hermes, ReAct, or AR.
- **Model**: benchmark-facing model identity and generation limits.
- **Serving**: how model requests reach an OpenAI-compatible endpoint, such as vLLM, SGLang, LiteLLM, or an existing endpoint.
- **Deployment**: where the run happens, such as client-only, local Docker, NSCC PBS, or CCDS cloud.

## Entry Points

Validate an experiment:

```bash
python3 validate.py --config experiments/client_only.toml
```

Run an experiment:

```bash
python3 run.py --config experiments/local_vllm.toml
```

Render a batch/cloud script when the deployment adapter supports one:

```bash
python3 run.py --config experiments/nscc_vllm.toml --render-job
```

Inspect a run directory:

```bash
python3 analyze.py --run runs/client_only
```

Use `--dry-run` to print the resolved execution plan without launching work.

## Deployment Examples

```bash
python3 run.py --config experiments/client_only.toml
python3 run.py --config experiments/local_vllm.toml
python3 run.py --config experiments/nscc_vllm.toml --render-job
python3 run.py --config experiments/ccds_vllm.toml --render-job
```

## Experiment Files

An experiment TOML owns the whole run and references reusable profiles:

```toml
name = "client_only"
output_dir = "runs/client_only"
seeds = [42]

[benchmark]
name = "swebench"
profile = "profiles/benchmarks/swebench/phase0.json"

[harness]
name = "openclaw"
profile = "profiles/harnesses/openclaw_swebench.yaml"

[model]
profile = "profiles/models/gemini_3_5_flash.json"

[serving]
profile = "profiles/serving/litellm_gemini.toml"

[deployment]
profile = "profiles/deployments/client_only.toml"
```

Model, serving, and deployment profiles live under `profiles/` so users can add a new provider or deployment without editing central runner code.

## Migration Map

- Runner/config code lives in `agent_bench/core` and `agent_bench/runtime`.
- Container helpers live in `agent_bench/deployments/containers`.
- Shared dispatcher protocol lives in `agent_bench/runtime/exec_bridge`.
- Harness-specific dispatchers live in `agent_bench/harnesses/<name>/dispatcher.py`.
- Telemetry code lives in `agent_bench/telemetry`.
- Analysis and reports live in `agent_bench/results`.
- SWE-bench scoring services live under `agent_bench/benchmarks/swebench/scoring_server`.
