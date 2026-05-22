# USAGE — Run SWE-bench Experiments

This document describes the current composed-config workflow for SWE-bench.
It intentionally uses the four independent config axes introduced by this
branch:

```text
APP             config/application/swebench/<phase>.json
HARNESS_CONFIG  config/harness/<harness>_swebench.json
MODEL_CONFIG    config/model/<model>.json
SERVING_CONFIG  config/serving/<backend>/<profile>.json
```

Do not use the old mixed harness configs or retired backend-config paths. Model
identity, context window, and output token cap live in `config/model/*`.
Serving profiles only describe endpoint/image/SIF/proxy deployment.

---

## 1. Pick the four axes

| Axis | Example | Owns |
|---|---|---|
| APP | `config/application/swebench/phase1.json` | instance ids, app budget, output leaf |
| HARNESS_CONFIG | `config/harness/openclaw_swebench.json` | OpenClaw/OpenHands/Hermes behavior and conditions |
| MODEL_CONFIG | `config/model/qwen35_9b.json` | model id/path, provider, `context_window`, `max_tokens` |
| SERVING_CONFIG | `config/serving/vllm/nscc.json` | serving backend endpoint, image, SIF, proxy settings |

Common harness choices:

```text
config/harness/openclaw_swebench.json
config/harness/openhands_swebench.json
config/harness/hermes_swebench.json
```

Common model choices:

```text
config/model/qwen35_9b.json
config/model/qwen36_27b.json
config/model/minimax_m2_7.json
```

Common serving choices:

```text
config/serving/vllm/nscc.json
config/serving/vllm/default.json
config/serving/litellm/minimax_global.json
```

Tensor parallelism is runtime state, not config-file state. Set `VLLM_TP`,
`SGLANG_TP`, or `TP` when submitting the job.

---

## 2. NSCC: Qwen3.5-9B with vLLM

This requests one GPU and starts vLLM through Singularity.

```bash
cd /home/users/ntu/boxi001/Agent_Bench

export KIT_ROOT=$(pwd)
export APP="$KIT_ROOT/config/application/swebench/phase1.json"
export MODEL_CONFIG="$KIT_ROOT/config/model/qwen35_9b.json"
export SERVING_CONFIG="$KIT_ROOT/config/serving/vllm/nscc.json"

for h in openclaw openhands hermes; do
  qsub -l select=1:ncpus=16:ngpus=1:mem=64gb \
    -l walltime=24:00:00 \
    -v "KIT_ROOT=$KIT_ROOT,AB_RUN_STAMP=swebench_phase1_${h}_qwen35_9b,APP=$APP,HARNESS_CONFIG=$KIT_ROOT/config/harness/${h}_swebench.json,MODEL_CONFIG=$MODEL_CONFIG,SERVING_CONFIG=$SERVING_CONFIG,VLLM_TP=1,STAGE_SERVING=1" \
    scripts/execution_pbs/run_benchmark.pbs
done
```

Before first use on NSCC, ensure the harness runtime SIFs are available:

```bash
bash scripts/execution_pbs/build_oc_sif.sh
bash scripts/execution_pbs/build_oh_sif.sh
bash scripts/execution_pbs/build_hm_sif.sh
```

`run_benchmark.pbs` also prepares the selected runtime SIF when
`STAGE_OC_CHECK=1`, but explicit build commands are useful for debugging.

---

## 3. NSCC: MiniMax through LiteLLM

Provider-backed models go through a local LiteLLM proxy. The agents connect to
the proxy's OpenAI-compatible `/v1/chat/completions`; provider API keys are
only passed to the proxy.

This does not need a GPU request:

```bash
cd /home/users/ntu/boxi001/Agent_Bench

export KIT_ROOT=$(pwd)
export APP="$KIT_ROOT/config/application/swebench/phase1.json"
export MODEL_CONFIG="$KIT_ROOT/config/model/minimax_m2_7.json"
export SERVING_CONFIG="$KIT_ROOT/config/serving/litellm/minimax_global.json"
export MINIMAX_API_KEY="<your-minimax-key>"

for h in openclaw openhands hermes; do
  qsub -l select=1:ncpus=16:mem=64gb \
    -l walltime=24:00:00 \
    -v "KIT_ROOT=$KIT_ROOT,AB_RUN_STAMP=swebench_phase1_${h}_minimax_m2_7,APP=$APP,HARNESS_CONFIG=$KIT_ROOT/config/harness/${h}_swebench.json,MODEL_CONFIG=$MODEL_CONFIG,SERVING_CONFIG=$SERVING_CONFIG,MINIMAX_API_KEY=$MINIMAX_API_KEY,STAGE_SERVING=1" \
    scripts/execution_pbs/run_benchmark.pbs
done
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
export APP="$KIT_ROOT/config/application/swebench/phase0.json"
```

Or create a new application config under `config/application/swebench/` with
only the desired `instances`. The app config controls the task set; do not
encode the task set in harness/model/serving configs.

---

## 5. Stages

Each stage can be disabled with `STAGE_*=0`.

| Stage | Meaning |
|---|---|
| `STAGE_SWEBENCH` | install/import SWE-bench into the shared scratch Python cache |
| `STAGE_PULL` | pull per-instance SWE-bench task SIFs |
| `STAGE_SERVING` | start vLLM or LiteLLM sidecar |
| `STAGE_OC_CHECK` | prepare the selected OpenClaw/OpenHands/Hermes runtime SIF |
| `STAGE_OTEL` | start local OTel collector |
| `STAGE_RUN` | invoke `orchestration.runner.driver` |
| `STAGE_CLEANUP` | stop sidecars and collectors |

Use `STAGE_SERVING` for the model-serving sidecar.

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

NSCC jobs cannot use Docker directly. `run_benchmark.pbs` exports
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
