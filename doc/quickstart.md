# Quick Start

This guide runs the OpenClaw + SWE-bench smoke profile through each supported
deployment path. The public interface is the experiment TOML:

```bash
python3 validate.py --config experiments/<name>.toml
python3 run.py --config experiments/<name>.toml
```

Experiment files compose reusable profiles from `profiles/`:

- benchmark profile: task set and scoring target
- harness profile: agent runtime and condition list
- model profile: model alias, context window, token limits
- serving profile: vLLM, SGLang, LiteLLM, or endpoint details
- deployment profile: client-only, local Docker, NSCC PBS, or CCDS cloud

## Setup

```bash
cp .env.example .env
$EDITOR .env
set -a && . .env && set +a
```

For local Docker and CCDS/NSCC vLLM examples, make sure the selected model is
available to the serving runtime. For LiteLLM/API examples, export the provider
API key named by the serving profile.

## Client-Only API Fallback

Client-only runs do not start vLLM or SGLang. They use a local LiteLLM proxy, or
an already reachable OpenAI-compatible endpoint, and keep provider secrets out
of harness profiles.

```bash
python3 validate.py --config experiments/client_only.toml
python3 run.py --config experiments/client_only.toml --dry-run
python3 run.py --config experiments/client_only.toml
```

The bundled `client_only` example uses OpenClaw + SWE-bench with
`profiles/serving/litellm_gemini.toml`.

## Local Docker + vLLM

Use this path for workstation development. The benchmark task and OpenClaw
runtime run in Docker containers; vLLM runs through the shared serving planner.

```bash
python3 validate.py --config experiments/local_vllm.toml
python3 run.py --config experiments/local_vllm.toml --dry-run
python3 run.py --config experiments/local_vllm.toml
```

If you still use Makefile helpers, treat them as shortcuts over the same
profiles:

```bash
make validate-configs
```

## NSCC PBS + vLLM

Render a short PBS wrapper. The worker calls the Python PBS stage runner, which
prepares directories, materializes benchmark data, prepares SIFs, starts
serving and OTel sidecars, runs the benchmark, and cleans up.

```bash
python3 validate.py --config experiments/nscc_vllm.toml
python3 run.py --config experiments/nscc_vllm.toml --render-job > nscc_vllm.pbs
qsub nscc_vllm.pbs
```

For provider/API fallback on NSCC without a GPU request:

```bash
python3 validate.py --config experiments/nscc_litellm.toml
python3 run.py --config experiments/nscc_litellm.toml --render-job > nscc_litellm.pbs
qsub nscc_litellm.pbs
```

## CCDS Cloud + vLLM

CCDS is the cloud deployment path for GPU-backed vLLM runs.

```bash
python3 validate.py --config experiments/ccds_vllm.toml
python3 run.py --config experiments/ccds_vllm.toml --render-job > ccds_vllm.sh
bash ccds_vllm.sh
```

The generated script is intentionally small: it sets deployment-scoped
environment and calls the same `run.py --config` path used locally.

## Outputs

Runs write under the experiment `output_dir`, usually `runs/<experiment>/<run_id>/`.
Set either `AB_RUN_DIR` or `AB_EXPERIMENT_DIR` to choose the run directory;
they are strict aliases and must match if both are set.
Common files:

- `manifest.json`: run metadata and terminal state
- `verdict.json`: benchmark scoring result
- `steps.jsonl`: LLM/tool step trace
- `spans.jsonl`: OpenTelemetry-derived spans when available
- `logs/`: serving, PBS, OTel, and sidecar logs depending on deployment
