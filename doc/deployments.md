# Deployment Developer Guide

Deployment adapters decide where a resolved experiment runs. They do not define
the benchmark, harness, model, or serving backend; those come from the
experiment and profiles.

The four deployment kinds are:

- `client_only`
- `local_docker`
- `nscc_pbs`
- `ccds_cloud`

## Client-Only

Profile: `profiles/deployments/client_only.toml`

Use client-only when GPU resources are unavailable or the model is already
reachable through an API. The benchmark and OpenClaw harness still run normally,
but local GPU serving is not started. Provider-backed examples should use
LiteLLM so all agents talk to one OpenAI-compatible `/v1` endpoint.

Canonical example:

```bash
python3 run.py --config experiments/client_only.toml
```

Implementation expectations:

- The serving profile must be LiteLLM, existing endpoint, or another
  OpenAI-compatible route.
- Provider API keys stay in launcher environment variables.
- Agents receive only the local proxy key and local endpoint.

## Local Docker

Profile: `profiles/deployments/local_docker.toml`

Use local Docker for development. The task container and OpenClaw runtime
container share a workspace bind, while serving is started through the shared
serving launch planner.

Canonical example:

```bash
python3 run.py --config experiments/local_vllm.toml
```

Implementation expectations:

- Docker must be available on the host.
- vLLM, SGLang, or LiteLLM can be launched by the serving planner.
- The runtime driver owns benchmark execution and result writing.
- The serving context starts only for local deployment kinds; batch/cloud
  wrappers own their sidecars.

## NSCC PBS

Profiles:

- `profiles/deployments/nscc_pbs.toml` for GPU-backed serving
- `profiles/deployments/nscc_pbs_cpu.toml` for provider/API fallback

Use NSCC PBS for HPC execution. `run.py --render-job` emits a short wrapper with
PBS directives and calls `python3 run.py --config ...` inside the allocation, so the same core stage coordinator drives local and PBS execution.

Canonical vLLM example:

```bash
python3 run.py --config experiments/nscc_vllm.toml --render-job > nscc_vllm.pbs
qsub nscc_vllm.pbs
```

Provider/API fallback example:

```bash
python3 run.py --config experiments/nscc_litellm.toml --render-job > nscc_litellm.pbs
qsub nscc_litellm.pbs
```

Stages:

- `prepare_environment`
- `materialize_benchmark`
- `prepare_images`
- `start_serving`
- `start_otel`
- `run_benchmark`
- `cleanup`

Implementation expectations:

- Stage flags are compatibility overrides, not canonical configuration.
- SIF preparation and OTel lifecycle live in the PBS stage runner.
- vLLM/SGLang/LiteLLM sidecars are created from the shared serving launch plan.
- LiteLLM/provider jobs should use CPU deployment profiles unless the
  experiment explicitly needs GPUs.

## CCDS Cloud

Profile: `profiles/deployments/ccds_cloud.toml`

CCDS cloud is the cloud deployment path for GPU-backed vLLM serving. It should
not be documented as the provider/API fallback path; use `client_only` or
`nscc_pbs_cpu` for that.

Canonical example:

```bash
python3 run.py --config experiments/ccds_vllm.toml --render-job > ccds_vllm.sh
bash ccds_vllm.sh
```

Implementation expectations:

- The CCDS profile marks `requires_gpu = true`.
- The generated script is a small wrapper around `python3 run.py --config`.
- Cloud-specific project or secret wiring belongs in the deployment layer, not
  in harness or benchmark profiles.

## Adding A Deployment

To add a new deployment kind:

1. Add a deployment profile in `profiles/deployments/`.
2. Add or extend a deployment adapter that returns a `DeploymentPlan`.
3. Add at least one experiment TOML using OpenClaw + SWE-bench as a smoke path.
4. Add tests for `validate.py`, `run.py --dry-run`, and `run.py --render-job`
   if the deployment renders a job script.
