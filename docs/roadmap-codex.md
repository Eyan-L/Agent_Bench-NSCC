# Codex CLI and RoadmapBench

The harness uses the official downloadable `@openai/codex` **0.157.1** release,
without patching Codex. `profiles/harnesses/codex_terminal_bench.yaml` supplies
the generic CLI configuration; `codex_roadmapbench.yaml` supplies the Roadmap
persona and Qwen protocol settings. The single-agent control disables native
multi-agent tools in `codex_roadmapbench_single_agent.yaml`.

## Runtime and model interface

Build the portable CLI runtime on local Ubuntu:

```bash
bash scripts/build_codex_sif_local.sh
```

Upload `dist/codex_0.157.1.sif` to NSCC's `$SCRATCH/sif_images` before submitting
a run. The local builder verifies the official binaries, preserves an existing
SIF as `.back.<timestamp>`, and atomically replaces its unchanged filename.

Codex's portable binaries run inside the benchmark task image, so commands
use the task's own dependencies. Each task has a fresh `CODEX_HOME`; personal
Codex authentication and configuration are never copied. NSCC runs use the
existing Qwen/SGLang server and do not require OpenAI connectivity.

Codex speaks the Responses API. The generic harness can forward Responses
directly; Roadmap uses a concurrent Responses/Chat Completions adapter for the
existing Qwen endpoint. The adapter preserves tool call IDs, native namespaces,
freeform tool input, text, usage, and truncation status. It does not implement
agent scheduling. This path supports text software tasks; images and other
unsupported input types fail explicitly.

## Subagents and completion

Native `multi_agent_v1.spawn_agent` returns an agent ID without waiting for the
child. The parent can continue work; `wait_agent` waits for specified `targets`
when a dependency requires their results. The configuration allows sixteen
children and one level of delegation. The Roadmap prompt follows OpenClaw's
delegation policy, with only native-tool substitutions.

Verify scheduling against the real CLI with a scripted model:

```bash
module load singularity/4.3.1
python3 scripts/verify_codex_subagents.py \
  --runtime-sif "$SCRATCH/sif_images/codex_0.157.1.sif" \
  --output-dir "$SCRATCH/agent-bench/probes/codex-async-$(date +%s)"
```

The child executes a four-second command while the parent writes its own
marker, then waits. The probe retains native rollouts and checks actual overlap.
This establishes runtime scheduling behavior; a real model may still choose
serial execution.

Roadmap uses a 7200-second execution deadline. Codex has no configured native
tool-call iteration cap in this harness. Normal completion requires exit zero,
a native completed parent turn, a nonempty final assistant message without
incomplete truncation (or explicit `DONE`), and available child timing evidence.
Exit zero alone is insufficient. The verifier reward is recorded separately.

All Codex task writers and the API recorder are stopped before candidate
freezing. The task runtime is restarted for the existing isolated Roadmap
verifier. Agent latency ends at writer fencing and excludes candidate copying
and verification.

## Run and inspect

```bash
python3 validate.py --config experiments/qwen36_35b_a3b_roadmapbench_codex_fyn_2_5_0_smoke.toml
python3 run.py --config experiments/qwen36_35b_a3b_roadmapbench_codex_fyn_2_5_0_smoke.toml --render-job > /tmp/roadmap-codex.pbs
mkdir -p "$SCRATCH/agent-bench/pbs-logs/roadmap-codex"
qsub -o "$SCRATCH/agent-bench/pbs-logs/roadmap-codex" /tmp/roadmap-codex.pbs
python3 scripts/analyze_roadmap_codex_runs.py <pbs-job-id> \
  --arm subagent --output-dir "$SCRATCH/agent-bench/reports/roadmap"
```

PBS IDs resolve through the shared Roadmap lookup under `$SCRATCH/agent-bench/runs`.
`compare_roadmap_codex_runs.py` accepts paired `--subagent` and `--single-agent`
arms. Both exporters retain durable native sessions, API/model/tool traces,
execution logs, the candidate patch, verifier evidence, and figures, without
copying full candidate workspace trees.

`codex_sessions.json` records thread lineage and active turn intervals.
`runtime_observability.json` records child counts, runtime, concurrency, and
parent work during child execution. Resumed child idle time is excluded from
active-runtime metrics. Spawn/wait coordination calls are not counted as
independent parent work. Missing child rollouts are reported as unknown.
