# RoadmapBench

RoadmapBench uses Terminal-Bench task definitions and verifier isolation while
providing its own OpenClaw condition, lifecycle policy, prompts, and
measurement artifacts.

Use `benchmark.name = "roadmap_bench"` with one of the Roadmap harness
profiles:

- `profiles/harnesses/openclaw_roadmapbench.yaml`
- `profiles/harnesses/openclaw_roadmapbench_single_agent.yaml`

Download pinned task definitions with:

```bash
python scripts/download_roadmapbench_tasks.py \
  --output-root /path/to/roadmapbench-tasks \
  <task-id> [<task-id> ...]
```

Set `tasks_root` in the benchmark profile to that directory. Legacy profiles
that use `benchmark.name = "terminal_bench"` and identify
`UnipatAI/RoadmapBench` as their source are still materialized as RoadmapBench
instances.

The harness keeps verifier tests out of the task runtime until scoring. It
freezes the candidate after the parent trajectory ends and after remaining
child writers have been quiesced, then runs the verifier against that frozen
workspace.

OpenClaw v2026.9.2 stores sessions in SQLite. The Roadmap runtime starts a
read-only exporter so lifecycle and trajectory measurement can consume a
stable JSONL projection without coupling the shared OpenClaw condition to the
runtime's private database schema.

The repository's default OpenClaw runtime remains the pinned v2026.4.22
release. To use v2026.9.2, explicitly set `AB_AGENT_RUNTIME_VERSION=2026.9.2`
and configure `AB_AGENT_RUNTIME_SIF` or `AB_AGENT_RUNTIME_DOCKER` to point at a
runtime prepared with the semantic compatibility patch below. Version-specific
bounded empty/length continuation, context-pressure recovery, loop handling,
and SQLite export are enabled only for v2026.9.2. Gateway and fallback runtime
modes are recorded independently from execution success; formal RoadmapBench
validity still requires the configured gateway runtime.

The semantic runtime compatibility patch operates on an extracted OpenClaw
tree and fails closed when release anchors are missing:

```bash
python agent_bench/harnesses/openclaw/v2026_9_2_patch.py --root /path/to/openclaw
python agent_bench/harnesses/openclaw/v2026_9_2_patch.py \
  --root /path/to/openclaw --verify-only
```
