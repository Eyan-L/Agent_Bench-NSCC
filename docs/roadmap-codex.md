# Codex CLI and RoadmapBench

The harness uses the official downloadable `@openai/codex` **0.157.1** release
without patching Codex. `codex_terminal_bench.yaml` is the generic profile;
`codex_roadmapbench.yaml` adds the Roadmap persona and Chat Completions adapter
settings. `codex_roadmapbench_single_agent.yaml` disables native subagents.
Use `benchmark.name = "roadmap_bench"` for Roadmap tasks.

## Runtime and model interface

Build the Docker runtime, or convert it to a SIF on a Linux build host:

```bash
bash agent_bench/harnesses/codex/docker/build.sh
bash scripts/build_codex_sif_local.sh
```

For Singularity deployments, set `AB_AGENT_RUNTIME_SIF` to the installed image.
The SIF builder verifies the official binaries, preserves an existing image
as `.back.*`, and atomically replaces its unchanged filename. Portable binaries
run inside the task image and use that task's dependencies. Every task gets a
fresh `CODEX_HOME`; personal authentication and configuration are not copied.

Codex speaks the Responses API. The generic harness can forward Responses
directly or adapt Chat Completions. Roadmap uses the concurrent Chat adapter
for the existing Qwen endpoint. It preserves tool IDs, native namespaces,
freeform arguments, text, phases, usage, and explicit `tool_choice` restrictions.
Unsupported inputs or tool restrictions fail explicitly. Length truncation
emits `response.incomplete` and is never reported as completed. Both Roadmap
profiles send `chat_template_kwargs.enable_thinking=false` on each model call.

## Native turns and subagents

The Roadmap profiles enable `require_done_marker`. Initial developer
instructions require implementation and validation followed by a final DONE.
The adapter sends `response.end_turn=false` before that marker, keeping the
unmodified CLI in its own tools-enabled sampling loop within the same turn.
Text and accompanying tool calls share one Chat assistant message. Each run
starts one `codex exec`; no automatic resume or extra user prompt is injected.

Native `spawn_agent` returns an agent ID without waiting for completion.
The parent can continue work and use `wait_agent` when results are needed.
The profile allows sixteen children and one delegation level. The Roadmap
persona follows OpenClaw's delegation policy with native-tool substitutions.
Actual overlap is measured from native turn intervals and independent parent
model/tool activity, excluding coordination calls.

## Completion and artifacts

Normal completion requires exit zero, matching CLI/rollout/API evidence for
the final parent turn, no pending calls, and no unfinished children. Old DONE
messages and completed turns cannot supply evidence for a later turn. Roadmap
Chat responses close the native turn after final DONE, or an explained blocker
ending in BLOCKED. BLOCKED is not normal completion. A child ends its completed
assigned scope with DONE; only the parent declares the full roadmap complete.
All calls remain subject to the original 7200-second task deadline.

The marker uses OpenClaw's durable-transcript rule: a line beginning with the
standalone word DONE, case insensitive. The generic Codex profile also accepts
valid nonempty terminal stops without that marker. Native phases, protocol
statuses, and truncation remain available for auditing.

Child execution state and timing quality are independent. Completed native
status can establish completion even if timing is unavailable. Running,
queued, interrupted, failed, or unknown child work prevents normal completion;
a child receiving new work needs new completion evidence.

All task and API writers stop before candidate freezing. Symbolic links,
including dangling links, are preserved. Session/API/timing evidence and the
patch are saved before copying the snapshot, so a copy failure retains the
execution evidence and invalidates the sample. Agent latency ends at writer
fencing and excludes copying and verification. Reward is recorded separately.

## Run and report

The [minimal Roadmap example](../agent_bench/benchmarks/roadmap_bench/README.md#minimal-example)
provides a pinned task, model profile, and existing-endpoint configuration:

```bash
python3 validate.py --config experiments/roadmapbench.example.toml
python3 run.py --config experiments/roadmapbench.example.toml
python3 scripts/analyze_roadmap_codex_runs.py /path/to/run --arm subagent
python3 scripts/compare_roadmap_codex_runs.py \
  --subagent /path/to/subagent-run --single-agent /path/to/single-agent-run
```

The analyzer accepts one or more runs for either arm. Both exporters accept
run paths or PBS IDs and retain durable sessions, API/model/tool traces,
execution logs, the candidate patch, verifier evidence, and figures without
copying full candidate workspace trees. Use `--runs-root` or `AB_RUNS_ROOT`
for another run collection; PBS lookup also recognizes `$SCRATCH/agent-bench/runs`.
