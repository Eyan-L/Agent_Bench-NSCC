# USAGE — Run SWE-bench Experiments

This document describes the current composed-config workflow for SWE-bench.
It intentionally uses the four independent config axes introduced by this
branch:

```text
APP             profiles/benchmarks/swebench/<phase>.json
HARNESS_CONFIG  profiles/harnesses/<harness>_swebench.yaml
MODEL_CONFIG    profiles/models/<model>.json
SERVING_CONFIG  profiles/serving/<backend>/<profile>.yaml
```

Do not use the old mixed harness configs or retired backend-config paths. Model
identity, context window, and output token cap live in `profiles/models/*`.
Serving profiles only describe endpoint/image/SIF/proxy deployment.

---

## 1. Pick the four axes

| Axis | Example | Owns |
|---|---|---|
| APP | `profiles/benchmarks/swebench/phase1.json` | instance ids, app budget, output leaf |
| HARNESS_CONFIG | `profiles/harnesses/openclaw_swebench.yaml` | OpenClaw/OpenHands/Hermes behavior and conditions |
| MODEL_CONFIG | `profiles/models/qwen35_9b.json` | model id/path, provider, `context_window`, `max_tokens` |
| SERVING_CONFIG | `profiles/serving/vllm/nscc.yaml` | serving backend endpoint, image, SIF, proxy settings |

Common harness choices:

```text
profiles/harnesses/openclaw_swebench.yaml
profiles/harnesses/openhands_swebench.yaml
profiles/harnesses/hermes_swebench.yaml
```

Common model choices:

```text
profiles/models/qwen35_9b.json
profiles/models/qwen36_27b.json
profiles/models/minimax_m2_7.json
```

Common serving choices:

```text
profiles/serving/vllm/nscc.yaml
profiles/serving/vllm/default.yaml
profiles/serving/litellm/minimax_global.yaml
```

Tensor parallelism is runtime state, not config-file state. Set `VLLM_TP`,
`SGLANG_TP`, or `TP` when submitting the job.

---

## 2. NSCC: Qwen3.5-9B with vLLM

This requests one GPU and starts vLLM through Singularity.

```bash
cd /home/users/ntu/boxi001/Agent_Bench

export KIT_ROOT=$(pwd)
export APP="$KIT_ROOT/profiles/benchmarks/swebench/phase1.json"
export MODEL_CONFIG="$KIT_ROOT/profiles/models/qwen35_9b.json"
export SERVING_CONFIG="$KIT_ROOT/profiles/serving/vllm/nscc.yaml"

python3 run.py --config experiments/nscc_vllm.toml --render-job > nscc_vllm.pbs
qsub -v KIT_ROOT=$KIT_ROOT,VLLM_TP=1 nscc_vllm.pbs
```

Before first use on NSCC, ensure the harness runtime SIFs are available:

```bash
bash agent_bench/deployments/pbs/build_oc_sif.sh
bash agent_bench/deployments/pbs/build_oh_sif.sh
bash agent_bench/deployments/pbs/build_hm_sif.sh
```

The generated PBS job also prepares the selected runtime SIF during the
`prepare_images` stage, but explicit build commands are useful for debugging.

---

## 3. NSCC: MiniMax through LiteLLM

Provider-backed models go through a local LiteLLM proxy. The agents connect to
the proxy's OpenAI-compatible `/v1/chat/completions`; provider API keys are
only passed to the proxy.

This does not need a GPU request:

```bash
cd /home/users/ntu/boxi001/Agent_Bench

export KIT_ROOT=$(pwd)
export APP="$KIT_ROOT/profiles/benchmarks/swebench/phase1.json"
export MODEL_CONFIG="$KIT_ROOT/profiles/models/minimax_m2_7.json"
export SERVING_CONFIG="$KIT_ROOT/profiles/serving/litellm/minimax_global.yaml"
export MINIMAX_API_KEY="<your-minimax-key>"

python3 run.py --config experiments/nscc_litellm.toml --render-job > nscc_litellm.pbs
qsub -v KIT_ROOT=$KIT_ROOT,MINIMAX_API_KEY=$MINIMAX_API_KEY nscc_litellm.pbs
```

LiteLLM logs are written to:

```text
data/experiments/<AB_RUN_STAMP>/logs/litellm.log
data/experiments/<AB_RUN_STAMP>/logs/litellm_config.yaml
```

---

## 4. Run a smaller subset

Use a smaller app config such as `phase0.json`:

```bash
export APP="$KIT_ROOT/profiles/benchmarks/swebench/phase0.json"
```

Or create a new application config under `profiles/benchmarks/swebench/` with
only the desired `instances`. The app config controls the task set; do not
encode the task set in harness/model/serving configs.

---

## 5. Stages

Each stage can be disabled with `STAGE_*=0`.

| Stage | Meaning |
|---|---|
| `prepare_environment` | create run/log/cache directories and export runtime env |
| `materialize_benchmark` | materialize SWE-bench instance JSON |
| `prepare_images` | pull task SIFs and prepare OpenClaw/OpenHands/Hermes runtime SIFs |
| `start_serving` | start vLLM, SGLang, or LiteLLM sidecar |
| `start_otel` | start local OTel collector |
| `run_benchmark` | invoke `agent_bench.runtime.driver` |
| `cleanup` | stop sidecars and collectors |

Use `AB_STAGE_START_SERVING=0` to skip the model-serving sidecar. Legacy names
such as `STAGE_SERVING=0` and `STAGE_PULL=0` are still accepted as submission
overrides.

---

## 6. Outputs and logs

All run products are grouped by experiment stamp:

```text
data/experiments/<AB_RUN_STAMP>/
  logs/
    pbs.log
    vllm.log or litellm.log
    engine_metrics.jsonl
    otel/
  runs/
    phase0/
    phase1/
      <run_id>/
        candidate.patch
        verdict.json
        steps.jsonl
        spans.jsonl
```

Useful commands:

```bash
tail -f data/experiments/<AB_RUN_STAMP>/logs/pbs.log
tail -f data/experiments/<AB_RUN_STAMP>/logs/vllm.log
tail -f data/experiments/<AB_RUN_STAMP>/logs/litellm.log

find data/experiments/<AB_RUN_STAMP>/runs -name verdict.json \
  -exec jq -r '.status // .summary // "unknown"' {} \; | sort | uniq -c
```

---

## 7. NSCC scoring

NSCC jobs cannot use Docker directly. The PBS stage runner exports
`OC_SCORING_SIF_DIR="$SIF_DIR"` after task SIFs are available, so
OpenClaw/OpenHands/Hermes score patches through the same per-instance SIFs.

Fallback scoring through a Docker-capable remote host is still available with:

```bash
OC_SCORING_URL=http://<docker-scoring-host>:8765
```

Scoring precedence:

```text
OC_SCORING_SIF_DIR -> OC_SCORING_URL -> local Docker SWE-bench evaluation
```

---

## 8. Common pitfalls

- Pass `SERVING_CONFIG` for the serving profile.
- Do not use old harness files with model names embedded in the filename; use
  `openclaw_swebench.json`, `openhands_swebench.json`, or
  `hermes_swebench.json` plus a separate `MODEL_CONFIG`.
- Do not put `context_window`, `max_tokens`, or tensor parallelism in serving
  configs. `context_window` and `max_tokens` belong to model configs; TP is a
  launch variable.
- Provider keys should go to LiteLLM, not directly to agent harness configs.
- Use absolute paths in `qsub -v`; PBS does not expand nested shell variables
  inside the `-v` payload.
