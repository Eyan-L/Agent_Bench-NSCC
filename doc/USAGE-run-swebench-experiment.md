# Run SWE-bench Experiments

This guide is SWE-bench-specific. For a cross-deployment quick start, see
[quickstart.md](quickstart.md).

## Current Interface

Run from an experiment TOML:

```bash
python3 validate.py --config experiments/local_vllm.toml
python3 run.py --config experiments/local_vllm.toml
```

For batch/cloud deployments, render a job script:

```bash
python3 run.py --config experiments/nscc_vllm.toml --render-job > nscc_vllm.pbs
qsub nscc_vllm.pbs
```

SWE-bench examples use these profile axes:

| Axis | Example | Owns |
|---|---|---|
| benchmark | `profiles/benchmarks/swebench/phase0.json` | instance ids, repo mapping, budgets |
| harness | `profiles/harnesses/openclaw_swebench.yaml` | OpenClaw behavior and conditions |
| model | `profiles/models/qwen35_9b.json` | model id, context window, token limits |
| serving | `profiles/serving/vllm/nscc.yaml` | endpoint, image, SIF, proxy settings |
| deployment | `profiles/deployments/nscc_pbs.toml` | execution environment and resource shape |

## OpenClaw SWE-bench Examples

Client-only API fallback through LiteLLM:

```bash
python3 run.py --config experiments/client_only.toml
```

Local Docker + vLLM:

```bash
python3 run.py --config experiments/local_vllm.toml
```

NSCC PBS + vLLM:

```bash
python3 run.py --config experiments/nscc_vllm.toml --render-job > nscc_vllm.pbs
qsub nscc_vllm.pbs
```

NSCC PBS + LiteLLM/provider, without a GPU request:

```bash
python3 run.py --config experiments/nscc_litellm.toml --render-job > nscc_litellm.pbs
qsub nscc_litellm.pbs
```

CCDS cloud + vLLM:

```bash
python3 run.py --config experiments/ccds_vllm.toml --render-job > ccds_vllm.sh
bash ccds_vllm.sh
```

## Changing The Sweep

To run a smaller or larger SWE-bench subset, edit or copy a benchmark profile
under `profiles/benchmarks/swebench/` and update the experiment TOML to point at
it. Do not encode task sets in harness, model, serving, or deployment profiles.

To change the agent runtime, switch the harness profile. Common SWE-bench
harness profiles:

```text
profiles/harnesses/openclaw_swebench.yaml
profiles/harnesses/openhands_swebench.yaml
profiles/harnesses/hermes_swebench.yaml
profiles/harnesses/ar_swebench.yaml
```

To change the model, switch the model profile. Model profiles remain JSON
because they are compact machine-readable aliases.

To change vLLM runtime parallelism, pass launch-time env vars such as `VLLM_TP`
or `TP`; tensor parallelism is not stored in serving profiles.

## Outputs

Run outputs usually land under:

```text
runs/<experiment>/<run_id>/
```

Important files:

- `candidate.patch`
- `manifest.json`
- `verdict.json`
- `steps.jsonl`
- `spans.jsonl`
- `validate_verdicts.jsonl`

## Scoring On NSCC

NSCC jobs cannot assume Docker is available on compute nodes. The PBS stage
runner prepares task SIFs and exports `OC_SCORING_SIF_DIR` so SWE-bench scoring
uses Singularity first.

Fallback scoring through a Docker-capable remote host is still available:

```bash
python3 -m agent_bench.benchmarks.swebench.scoring_server.main \
  --swebench-root /path/to/SWE-bench \
  --host 0.0.0.0 \
  --port 8765

qsub -v OC_SCORING_URL=http://<docker-host>:8765 nscc_vllm.pbs
```

Scoring precedence:

```text
OC_SCORING_SIF_DIR -> OC_SCORING_URL -> local Docker SWE-bench evaluation
```

## Common Pitfalls

- Use experiment TOMLs as the public run surface; direct driver commands are
  for low-level debugging.
- Harness and serving profiles are YAML/TOML, not the old JSON profile copies.
- Provider keys belong to LiteLLM launcher environment variables, not harness
  profiles.
- `context_window` and `max_tokens` belong to model profiles.
- Tensor parallelism and GPU visibility are launch-time environment, not model
  identity.
