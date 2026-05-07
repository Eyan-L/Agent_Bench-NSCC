# run/ — experiment orchestration

One Go binary (`experiment`) + a handful of bash helpers that launch sglang.
Everything needed to reproduce a replay experiment end-to-end lives here;
plotting and analysis scripts live in [`../draw/`](../draw/), and the load
generator itself lives in [`../loadgen/`](../loadgen/).

## Layout

```
run/
  experiment/                 # Go module — orchestration logic
    cmd/experiment/main.go    # CLI entry point
    config.go                 # JSON config schema + defaults + validation
    sglang.go                 # wraps sglang_launcher.sh as a subprocess
    loadgen.go                # builds /loadgen binary and dispatches subcommand
    plots.go                  # invokes draw/*.py after the run
    run.go                    # Run() top-level orchestration
  config/stress/                    # example JSON configs for alibaba / azure / agent
    alibaba.json
    azure.json
    agent.json
  sglang_launcher.sh             # single-instance sglang launcher (bare-metal python3)
  sglang_launcher_singularity.sh # same shape, wraps python3 in `singularity run` (NSCC 2A)
  sglang_launch_multiple.sh      # router + N-worker sglang
  run_alibaba_trace_experiment.sh  # legacy shell orchestrators (kept for quick
  run_azure_trace_experiment.sh    # one-off invocations and backward compat)
  run_agents.sh
  replay_alibaba_trace.sh     # loadgen-only (no sglang lifecycle)
  replay_agent_trace.sh
```

## Usage

```bash
# Build the experiment binary once.
cd run/experiment
go build -o experiment-bin ./cmd/experiment

# Run a canned config.
./experiment-bin -config ../../config/stress/alibaba.json
```

The binary:
1. loads the config, fills in defaults, and writes the fully-resolved version
   to `<output_dir>/run_config.json`;
2. launches sglang via `run/sglang_launcher.sh` (or `sglang_launch_multiple.sh`);
3. builds `/loadgen/loadgen-bin` if it's missing or out of date;
4. runs the matching subcommand (`alibaba` | `azure` | `agent`) with flags
   derived from the config;
5. stops sglang (and `dcgmi dmon`, if `plots.sm_trace=true`);
6. invokes every enabled analysis script under [`../draw/`](../draw/).

Project root is auto-detected by walking up until it finds
`loadgen/go.mod` + `run/sglang_launcher.sh`; pass `-project-root <path>` to
override.

## Config schema

See [`config/stress/alibaba.json`](config/stress/alibaba.json) for a fully populated
example. The top-level layout:

```json
{
  "name": "short experiment label",
  "output_dir": "experiment/{timestamp}_{trace_type}",
  "sglang": { /* port, GPUs, model path, mem fraction, etc. */ },
  "loadgen": {
    "trace_type": "alibaba",
    "speedup": 1.0,
    "duration": 900,
    "alibaba": { /* trace path + session sampling */ }
  },
  "plots": { "enabled": true, "sm_trace": false }
}
```

Two tokens are expanded in `output_dir`: `{timestamp}` (start time) and
`{trace_type}` (copied from `loadgen.trace_type`).

### NSCC 2A — singularity launcher

Set `sglang.launcher="singularity"` and provide the container knobs. See
[`config/stress/alibaba_nscc_2a_singularity.json`](config/stress/alibaba_nscc_2a_singularity.json):

```json
"sglang": {
  "launcher": "singularity",
  "cuda_visible_devices": "0,1,2,3",
  "model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",      // HF model name; resolved via $HF_HOME
  "singularity": {
    "sif_image": "/scratch/users/ntu/USERNAME/sglang.sif",
    "mounts":   ["/scratch/users/ntu/USERNAME/models:/scratch/users/ntu/USERNAME/models"],
    "modules":  ["singularity"]                          // default if omitted
  }
}
```

This routes to [`sglang_launcher_singularity.sh`](sglang_launcher_singularity.sh)
which runs `module load <modules> && singularity run --nv --env
CUDA_VISIBLE_DEVICES=... --bind <host>:<container> <sif> python3 -m
sglang.launch_server ...`. Everything else in the experiment config
(loadgen, plots, output_dir) works unchanged.

Same-path binds (`/scratch/.../models:/scratch/.../models`) let the HF
cache inside the container keep the same paths as outside. The launcher
auto-forwards `HF_HOME`, `HF_HUB_OFFLINE`, `HF_TOKEN`, and
`TRANSFORMERS_CACHE` via `--env` when they're set on the host — so setting
`export HF_HOME=/scratch/.../models` in your PBS job is enough for sglang
to resolve the HF model name.

Typical use on NSCC 2A is inside a PBS job — the job script launches
`experiment-bin -config ...json` on the head node; `experiment-bin` then
starts sglang in the same shell (not over SSH), waits for `/health`, runs
loadgen, and teardown is handled automatically.

**Post-run plotting on NSCC**: set `plots.env_prelude` to whatever shell
snippet puts `python3` + `matplotlib` on PATH before each plot script runs.
On NSCC 2A login/compute nodes:

```json
"plots": {
  "enabled": true,
  "env_prelude": "module load miniforge3",
  "python_bin": "python3"
}
```

When `env_prelude` is non-empty, every plot invocation runs as
`bash -c "<prelude> && <python_bin> draw/<script>.py ..."`, so the module
load happens in the same shell that calls python. Leave `env_prelude`
empty on machines where `python3` + deps are already on PATH (e.g. a
local dev box with conda already activated).

`python_bin` defaults to `python3`; override it when a specific conda env
interpreter is required, e.g. `"python_bin": "/home/users/.../envs/agentbench/bin/python"`.

## Auto-plot

When `plots.enabled=true`, each of these draw scripts runs unless the
matching per-field toggle is explicitly false:

| Plot                  | Script                                   | Input                                  |
|-----------------------|------------------------------------------|----------------------------------------|
| Experiment result     | `draw/plot_experiment_result.py`         | `<output_dir>`                         |
| sglang TFLOPS         | `draw/plot_sglang_tflops_over_time.py`   | `<output_dir>/sglang.log`              |
| sglang token usage    | `draw/plot_sglang_token_usage_over_time.py` | `<output_dir>/sglang.log`           |
| GPU SM metrics        | `draw/plot_sm_metrics_over_time.py`      | `<output_dir>/sm.log` (only if sm_trace=true) |
| SLO attainment + goodput | `draw/slo_attainment.py`              | `<output_dir>/<trace>_result.json`     |
| Cache hit vs theoretical (alibaba only) | `draw/plot_cache_hit_vs_theoretical.py` | `<output_dir>/responses.json` + raw trace |

Plot failures are logged but do not fail the experiment.
