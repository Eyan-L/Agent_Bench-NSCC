# Agent Bench

Agent Bench is a research platform for measuring trajectory-centric agent
workloads on shared LLM serving infrastructure. It compares full task
trajectories, tool execution, resource use, and task success rather than only
request-level serving metrics such as TTFT and TPOT.

The public run interface is experiment-based:

```bash
python3 validate.py --config experiments/client_only.toml
python3 run.py --config experiments/local_vllm.toml
python3 run.py --config experiments/nscc_vllm.toml --render-job
python3 analyze.py --run runs/<run_id>
```

`analyze.py` writes a portable, read-only report bundle under
`reports/analyze/` by default. Use `--output-dir` or `--name` to choose another
location, and `--json` for the same structured summary on stdout:

```bash
python3 analyze.py \
  --run runs/<experiment>/<run_id> \
  --output-dir reports/analyze \
  --name my-run
```

The bundle preserves the source run artifacts byte-for-byte, records SHA-256
and export status in `artifact-index.json`, and provides `index.html` plus the
existing span timeline. Missing optional telemetry remains explicitly marked
as not collected rather than being represented as zero.

Experiments compose reusable profiles from `profiles/`:

- benchmark: SWE-bench, Terminal-Bench, RoadmapBench, DeepResearch-Bench, or LongBench
- harness: OpenClaw, OpenHands, Hermes, ReAct, or AR
- model: model alias, context window, and token limits
- serving: vLLM, SGLang, LiteLLM, or an existing OpenAI-compatible endpoint
- deployment: client-only, local Docker, NSCC PBS, or CCDS cloud

## Quick Start

```bash
cp .env.example .env
$EDITOR .env
set -a && . .env && set +a
```

OpenClaw + SWE-bench smoke examples:

```bash
# API fallback through LiteLLM
python3 run.py --config experiments/client_only.toml

# Local Docker containers + vLLM
python3 run.py --config experiments/local_vllm.toml

# NSCC PBS + vLLM
python3 run.py --config experiments/nscc_vllm.toml --render-job > nscc_vllm.pbs
qsub nscc_vllm.pbs

# CCDS cloud + vLLM
python3 run.py --config experiments/ccds_vllm.toml --render-job > ccds_vllm.sh
bash ccds_vllm.sh
```

See [docs/quickstart.md](docs/quickstart.md) for prerequisites, provider/API
fallback, and output locations.

## Documentation

- [docs/quickstart.md](docs/quickstart.md): shortest path to run OpenClaw-based examples.
- [docs/deployments.md](docs/deployments.md): deployment adapter behavior and examples.
- [docs/layout.md](docs/layout.md): repository layout and public surfaces.
- [docs/execution.md](docs/execution.md): runtime operations, outputs, telemetry, and troubleshooting.
- [docs/benchmarks.md](docs/benchmarks.md): adding a benchmark.
- [docs/conditions.md](docs/conditions.md): adding a condition.
- [docs/harnesses.md](docs/harnesses.md): adding an agent runtime.

## Tests

```bash
conda run -n ab python3 -m pytest -q
conda run -n ab python3 -m compileall agent_bench run.py validate.py analyze.py
```
