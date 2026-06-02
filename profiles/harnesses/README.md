# Composed Config Layout

Agent_Bench now composes four independent profile axes:

```text
profiles/benchmarks/<benchmark>/<sweep>.json  # tasks, instances, budgets, output dirs
profiles/harnesses/<harness>_<benchmark>.yaml    # agent framework behavior + conditions
profiles/models/<model>.json              # model id/path, context window, output cap
profiles/serving/<backend>/<profile>.yaml # serving endpoint / engine or proxy deployment
```

Examples:

```bash
make run_phase0 \
  AB_HARNESS_CONFIG=openclaw_swebench \
  AB_MODEL_CONFIG=qwen25_05b \
  AB_SERVING_CONFIG=vllm/debug

make run_phase1 \
  AB_HARNESS_CONFIG=hermes_swebench \
  AB_MODEL_CONFIG=qwen36_27b \
  AB_SERVING_CONFIG=vllm/default
```

Important boundaries:

- Benchmark configs own tasks/budgets and a run-set leaf like `phase0`;
  the experiment root is `data/experiments/<AB_RUN_STAMP>/`.
- Harness configs must not contain `model`. They only describe the framework
  and conditions.
- Model configs own `context_window` and `max_tokens`. `context_window` is
  also the default `--max-model-len` passed to vLLM/SGLang; `max_tokens` is
  the single-response completion cap.
- Serving configs own endpoint, image/SIF, scheduler, and engine knobs. They
  must not contain context length or tensor-parallel size.
- Provider-backed serving should normally use `backend: "litellm"`. The PBS
  launcher starts a local LiteLLM OpenAI-compatible proxy, passes the real
  upstream key only to that proxy container via `serving.provider.*.api_key_env`,
  and exposes only a local proxy key to OpenClaw/OpenHands/Hermes. Direct
  `backend: "provider"` / `backend: "api"` profiles are not supported.
- Tensor parallelism is launch state: set `AB_SERVING_TENSOR_PARALLEL_SIZE` or backend-specific `VLLM_TP`/`SGLANG_TP`
  when submitting/running the job.
- A valid run must satisfy:

```text
model.max_tokens <= model.context_window
```

The harness adapters render OpenClaw/OpenHands/Hermes runtime config from the
same model + serving pair, so the agent metadata and served engine cannot drift.
