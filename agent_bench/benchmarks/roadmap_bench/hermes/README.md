# RoadmapBench with Hermes

This is a separate RoadmapBench harness. It does not replace the existing
RoadmapBench/OpenClaw or ordinary Terminal-Bench/Hermes path.

The execution chain is SGLang (Qwen3.6) -> patched Hermes runtime SIF ->
Agent_Bench exec bridge -> prepared Roadmap task SIF. Hermes runs as a lead
engineer with the native `delegate_task` toolset. A batch of Hermes children
executes in parallel **synchronously**: the parent receives their results
after the batch finishes. This is not OpenClaw `sessions_yield`/announce.

The task workspace is primed from its prepared SIF and shared with the Hermes
runtime. `/tests` is an empty private bind during the agent run. The harness
stops agent/bridge/task writers, restarts the task against the frozen workspace,
extracts `candidate.patch` relative to the initial Git HEAD, then injects
`/tests` for scoring. Verifier reward remains independent of agent exit status.

## Local runtime SIF build

On a local Linux/WSL machine with Docker and Apptainer or Singularity:

```bash
cd Agent_Bench
bash agent_bench/benchmarks/roadmap_bench/hermes/build_sif.sh "$PWD/hermes_runtime.sif"
```

The script uses the existing pinned Hermes Dockerfile and patch manifest,
converts the patched Docker image to SIF, checks the CLI/import, and refuses
to overwrite an existing output. It does not build task SIFs.

Upload the SIF to NSCC, then atomically install it:

```bash
scp hermes_runtime.sif nscc:/scratch/users/ntu/n2602695/sif_images/hermes_runtime.sif.uploading
ssh nscc 'mv /scratch/users/ntu/n2602695/sif_images/hermes_runtime.sif.uploading /scratch/users/ntu/n2602695/sif_images/hermes_runtime.sif'
```

The PBS `prepare_images` stage checks both the uploaded runtime SIF and each
pre-existing task SIF. It never builds/pulls them for this Hermes Roadmap path.

## Configuration and smoke

Use the new smoke config first:

```bash
ssh nscc 'cd /home/users/ntu/n2602695/Agent_Bench && python3 run.py --config experiments/qwen36_35b_a3b_roadmapbench_hermes_fal_4_1_0_smoke.toml --dry-run'
```

After reviewing the generated job and SIF preflight, submit with the usual
`run.py` submission flow. The new subset10a/10b experiment configs are for
later runs. No GPU/PBS job is submitted by preparing this chain.

The profile sets `max_concurrent_children=16`, `max_iterations=600`, and
`child_timeout_seconds=7200`. These are Hermes delegation settings, not
OpenClaw subagent settings. Hermes child work is synchronous and children use
isolated context, so latency and fan-out are not directly comparable to
OpenClaw's asynchronous `sessions_yield` without accounting for that difference.
