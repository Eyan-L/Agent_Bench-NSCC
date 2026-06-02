# Legacy Name Migration

This document records the legacy names removed from Agent Bench and their canonical replacements. The migration intentionally does not keep backward-compatible aliases. New names use the `AB_*` namespace and put the axis or subsystem first: benchmark, harness, agent runtime, model, serving, deployment, telemetry, scoring, or run.

Provider-native names are still allowed when they belong to an external API or backend contract, such as `OPENAI_*`, `OPENCLAW_*`, `VLLM_*`, `SGLANG_*`, and `LITELLM_*`.

## Experiment And CLI Names

| Old representation | New representation | Meaning | Used for |
|---|---|---|---|
| `KIT_ROOT` | `AB_REPO_ROOT` | Repository root. | Resolving profiles, scripts, assets, and runtime resources. |
| `APP`, `APP_PROFILE_PATH`, `AB_APP_CONFIG` | `AB_BENCHMARK_PROFILE`, `AB_BENCHMARK_PROFILE_PATH`, `AB_BENCHMARK_CONFIG` | Benchmark profile selector and resolved benchmark config path. | Selecting SWE-bench, Terminal-Bench, LongBench, or DeepResearch sweeps. |
| `HARNESS` | `AB_HARNESS_CONFIG` | Harness profile selector. | Selecting OpenClaw, OpenHands, Hermes, ReAct, or other agent runtimes. |
| `MODEL_PROFILE` | `AB_MODEL_CONFIG` | Model profile selector or path. | Selecting model id/path, context window, tool parser, and token limits. |
| `SERVING` | `AB_SERVING_CONFIG` | Serving profile selector or path. | Selecting vLLM, SGLang, LiteLLM, existing endpoint, or OpenAI-compatible serving. |
| `--app` | `--benchmark-config` | Runtime driver benchmark config path. | Passing the benchmark sweep config into `agent_bench.runtime.driver`. |
| `--model-profile` | `--model-config` | Model config path. | Passing model config into serving launchers and runtime drivers. |
| `--serving` when it meant a config path | `--serving-config` | Serving config path. | Passing serving config into serving launchers and runtime drivers. |
| `--instance-data` | `--benchmark-instance-data` | Materialized benchmark instance JSON. | Giving the driver the concrete task/instance file for a run. |
| `--swebench-root` | `--benchmark-data-root` | Benchmark data root. | Locating benchmark-specific data such as SWE-bench checkouts/evaluation assets. |
| `--app-leaf` | `--benchmark-leaf` | Output leaf for a benchmark sweep. | Re-judging or locating persisted DeepResearch outputs. |
| `--kit-root` | `--repo-root` | Repository root CLI argument. | PBS image-build helpers that need the source tree. |

## Benchmark Data Names

| Old representation | New representation | Meaning | Used for |
|---|---|---|---|
| `SWEBENCH_ROOT`, `AB_SWEBENCH_ROOT`, `swebench_root` | `AB_BENCHMARK_DATA_ROOT`, `benchmark_data_root` | Root directory for benchmark data. | SWE-bench evaluation data and any benchmark that needs a shared data root. |
| `INSTANCE_DATA`, `AB_INSTANCE_DATA`, `instance_data` config key | `AB_BENCHMARK_INSTANCE_DATA`, `benchmark_instance_data` | Materialized instance/task JSON path. | All benchmark drivers after dataset materialization. |
| `OC_TB_TASKS_ROOT` | `AB_BENCHMARK_TASKS_ROOT` | Terminal-Bench task snapshot root. | Loading Terminal-Bench tasks from a local dataset snapshot. |
| `OC_DR_QUERY_FILE` | `AB_BENCHMARK_QUERY_FILE` | DeepResearch query JSONL path. | Materializing DeepResearch-Bench instances. |
| `DR_SANDBOX_IMAGE`, `LONGBENCH_SANDBOX_IMAGE` | `AB_BENCHMARK_SANDBOX_IMAGE` | Benchmark sandbox image. | DeepResearch and LongBench task sandboxes. |
| `DR_AGENT_TIMEOUT_SEC`, `LONGBENCH_AGENT_TIMEOUT_SEC` | `AB_BENCHMARK_AGENT_TIMEOUT_SEC` | Per-instance agent timeout. | Benchmark harness execution limits. |
| `DR_SKIP_EVAL` | `AB_BENCHMARK_SKIP_EVAL` | Evaluation skip flag. | Running DeepResearch agent-only passes. |
| `DR_PASS_THRESHOLD` | `AB_BENCHMARK_PASS_THRESHOLD` | Pass/fail threshold. | DeepResearch judge verdicts. |
| `DRB_JUDGE_WORKERS` | `AB_BENCHMARK_JUDGE_WORKERS` | Judge concurrency. | DeepResearch host-side evaluation. |
| `LONGBENCH_DATA_DIR` | `AB_BENCHMARK_DATA_ROOT` | LongBench dataset root/cache. | Loading LongBench examples. |
| `LONGBENCH_EXAMPLE_CAP` | `AB_BENCHMARK_EXAMPLE_CAP` | Per-subtask example cap. | LongBench smoke/subset runs. |
| `LONGBENCH_GPU_DEVICE` | `AB_BENCHMARK_GPU_DEVICE` | GPU device selector. | LongBench sandbox/ClawRAG service wiring. |
| `LONGBENCH_HF_CACHE` | `AB_BENCHMARK_HF_CACHE` | Hugging Face cache path. | LongBench dataset/model cache mounting. |
| `LONGBENCH_CONCURRENCY` | `AB_BENCHMARK_CONCURRENCY` | Benchmark concurrency. | LongBench RAG loop scheduling. |
| `LONGBENCH_RAG_SUBTASK_LOOP` | `BENCHMARK_RAG_SUBTASK_LOOP` | Telemetry step subtype for the RAG subtask loop. | LongBench step trace records without a benchmark-specific legacy prefix. |

## Serving And Model Names

| Old representation | New representation | Meaning | Used for |
|---|---|---|---|
| `SERVING_BACKEND` | `AB_SERVING_BACKEND` | Serving backend name. | Selecting vLLM, SGLang, LiteLLM, existing endpoint, or OpenAI-compatible serving. |
| `SERVING_MODE` | `AB_SERVING_MODE` | Serving launch mode. | Native, Docker, Singularity, or external endpoint serving. |
| `MODEL`, `VLLM_MODEL` when used as the Agent Bench model id | `AB_MODEL_NAME` | Served model name or path. | Rendering launch plans and agent runtime configs. |
| `MODEL_ENDPOINT`, `OC_VLLM_BASE_URL` | `AB_SERVING_BASE_URL` | OpenAI-compatible serving base URL. | Agent frameworks and scoring clients. |
| `MODEL_API_KEY`, `OC_VLLM_API_KEY` | `AB_SERVING_API_KEY` | Agent-facing API key. | LiteLLM proxy auth and OpenAI-compatible clients. |
| `MODEL_PROVIDER` | `AB_MODEL_PROVIDER` | Agent-facing provider label. | Hermes/OpenClaw provider routing. |
| `SERVING_PROVIDER_BASE_URL` | `AB_SERVING_PROVIDER_BASE_URL` | Upstream provider base URL. | LiteLLM rendering and provider proxy setup. |
| `SERVING_PROVIDER_API_KEY_ENV` | `AB_SERVING_PROVIDER_API_KEY_ENV` | Name of the upstream provider secret env var. | LiteLLM secret injection without exposing the secret to agents. |
| `PORT` | `AB_SERVING_PORT` | Serving HTTP port. | Launch plans and readiness checks. |
| `TP` | `AB_SERVING_TENSOR_PARALLEL_SIZE` | Tensor parallel size. | vLLM/SGLang launch-time parallelism. |
| `MAX_MODEL_LEN` | `AB_SERVING_MAX_MODEL_LEN` | Engine maximum context length. | vLLM/SGLang server launch flags. |
| `MAX_TOKENS` | `AB_MODEL_MAX_OUTPUT_TOKENS` | Per-response output token cap. | Agent model config and generation settings. |
| `GPU_MEM_UTIL` | `AB_SERVING_GPU_MEMORY_UTILIZATION` | GPU memory utilization fraction. | vLLM/SGLang launch plans. |
| OpenClaw provider id `vllm` and model string `vllm/<model>` | `agent_model` and `agent_model/<model>` | Logical agent model provider. | Generated OpenClaw runtime state independent of the serving backend. |

## Harness And Agent Runtime Names

| Old representation | New representation | Meaning | Used for |
|---|---|---|---|
| `HARNESS_BACKEND` | `AB_DEPLOYMENT_CONTAINER_BACKEND` | Container backend for harness/task execution. | Selecting Docker or Singularity execution. |
| `OC_IMAGE_TAG`, `OH_IMAGE`, `HM_IMAGE` | `AB_AGENT_RUNTIME_IMAGE` | Concrete agent runtime image. | OpenClaw, OpenHands, and Hermes harness containers. |
| `OC_RUNTIME_*`, `OH_RUNTIME_*`, `HM_RUNTIME_*` | `AB_AGENT_RUNTIME_*` | Agent runtime image/build settings. | Version, base image/SIF, patched SIF, rebuild flag, overlay, and inside package path. |
| `OC_AGENT_ID` | `AB_AGENT_RUNTIME_AGENT_ID` | Agent id/name inside the runtime framework. | Rendering runtime config and state. |
| `HERMES_AGENT_BENCH_EVENTS` | `AB_AGENT_RUNTIME_EVENTS_FILE` | Agent runtime event output file. | Hermes telemetry event capture. |
| `OC_WORKSPACE` | `AB_AGENT_WORKSPACE` | Agent workspace path. | OpenClaw runtime workspace configuration. |
| `OC_HOST_WORKSPACE_BASE` | `AB_HARNESS_WORKSPACE_BASE` | Host workspace base directory. | Per-run workspace creation. |
| `OC_HOST_KEEP_WORKSPACE` | `AB_HARNESS_KEEP_WORKSPACE` | Workspace retention flag. | Debugging failed runs without deleting workspaces. |
| `OC_HOST_STATE_TEMPLATE` | `AB_HARNESS_STATE_TEMPLATE` | Rendered harness state template path. | OpenClaw plugin warmup and runtime config generation. |
| `OC_PLUGIN_CACHE` | `AB_HARNESS_PLUGIN_CACHE` | OpenClaw plugin dependency cache. | PBS plugin warmup and per-run plugin staging. |
| `OC_GATEWAY_PORT_MIN`, `OC_GATEWAY_PORT_MAX`, `OC_GATEWAY_PORT_STRIDE` | `AB_HARNESS_GATEWAY_PORT_MIN`, `AB_HARNESS_GATEWAY_PORT_MAX`, `AB_HARNESS_GATEWAY_PORT_STRIDE` | Gateway port allocation range. | Avoiding port collisions across concurrent harness runs. |
| `OC_BIN` | `AB_HARNESS_OPENCLAW_BIN` | OpenClaw CLI path. | OpenClaw configure/run scripts. |
| `OC_PATCH_*` | `AB_HARNESS_OPENCLAW_PATCH_*` | OpenClaw patch marker variables. | Preventing duplicate in-place runtime patches. |
| `HM_PATCH_*` | `AB_HARNESS_HERMES_PATCH_*` | Hermes patch marker variables. | Preventing duplicate in-place runtime patches. |
| `OC_PYLIB` | `AB_RUNTIME_PYLIB` | Shared Python library/cache root. | Runtime dependency materialization on PBS. |
| `OC_PIP_CACHE` | `AB_RUNTIME_PIP_CACHE` | Pip cache path. | Faster repeated runtime setup. |

## Deployment, Scoring, And Telemetry Names

| Old representation | New representation | Meaning | Used for |
|---|---|---|---|
| `AGENT_BENCH_PBS_SITE` | `AB_PBS_SITE` | PBS site identifier. | Site-specific PBS defaults and job metadata. |
| `AGENT_BENCH_PROJECT` | `AB_DEPLOYMENT_PROJECT` | Deployment project/account name. | Cloud/PBS deployment metadata. |
| `AGENT_BENCH_RUN_DIR` | `AB_RUN_DIR` | Active run directory. | Runtime artifacts, manifests, and per-run outputs. |
| `AGENT_BENCH_LOG_DIR` | `AB_LOG_DIR` | Active log directory. | Stage logs, telemetry files, and serving logs. |
| `AGENT_BENCH_STAGES` | `AB_STAGE_SUMMARY` | Enabled/disabled stage summary. | Rendered job metadata and debugging. |
| `STAGE_SERVING`, `STAGE_OTEL`, `STAGE_SWEBENCH`, `STAGE_CLEANUP` | `AB_STAGE_START_SERVING`, `AB_STAGE_START_OTEL`, `AB_STAGE_RUN_BENCHMARK`, `AB_STAGE_CLEANUP` | Per-stage enable flags. | Controlling composed execution stages. |
| `EVAL_MAX_WORKERS` | `AB_RUN_MAX_WORKERS` | Driver concurrency. | Parallel run execution. |
| `SIF_DIR`, `OC_SIF_DIR`, `OC_SCORING_SIF_DIR` | `AB_SCORING_IMAGE_DIR` | SWE-bench scoring/task SIF directory. | Pulling and locating scoring images for SWE-bench. |
| `OC_SCORING_URL` | `AB_SCORING_SERVICE_URL` | SWE-bench scoring service URL. | Calling the scoring server from harness adapters. |
| `OC_SCORING_*` | `AB_SCORING_*` | Scoring service/image/runtime settings. | SWE-bench scoring server and Singularity scoring paths. |
| `OC_HF_SPLIT` | `AB_SCORING_DATASET_SPLIT` | Dataset split for scoring. | SWE-bench scoring/evaluation selection. |
| `OC_OTEL_*` | `AB_TELEMETRY_*` | Telemetry collector/export settings. | OTel collector, trace files, ports, and service names. |
| `OC_UTIL_SAMPLE` | `AB_TELEMETRY_UTIL_SAMPLE` | Utilization sampling enable flag. | Optional serving-process telemetry. |
| `OC_UTIL_SAMPLE_INTERVAL` | `AB_TELEMETRY_UTIL_SAMPLE_INTERVAL` | Utilization sampling interval. | Optional CPU/GPU/process sampling cadence. |
| `OC_LLM_PID` | `AB_TELEMETRY_SERVING_PID` | Serving process id. | Optional serving-process utilization sampling. |
