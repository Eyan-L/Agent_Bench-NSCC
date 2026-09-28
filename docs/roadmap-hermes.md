# RoadmapBench with Hermes

`profiles/harnesses/hermes_roadmapbench.yaml` runs the pinned Hermes
`v2026.5.16` release on the same task workspace, Git baseline, and isolated
verifier as OpenClaw. Use `benchmark.name = "roadmap_bench"`; legacy
Terminal-Bench profiles identifying RoadmapBench remain supported.

## Runtime

Build the Docker runtime, or convert it to a SIF on a Linux build host:

```bash
bash agent_bench/harnesses/hermes/roadmap/build.sh
bash scripts/build_roadmap_hermes_sif_local.sh
```

For Singularity deployments, install the image at `AB_AGENT_RUNTIME_SIF`.
To replace an existing runtime without changing its filename, upload the new
image as `<target>.incoming`, then run:

```bash
bash scripts/install_roadmap_hermes_sif.sh /path/to/hermes_roadmap_v2026.5.16.sif
```

The installer verifies patch status, preserves the old image as `.back.*`,
and atomically replaces the target. The runtime includes Agent_Bench's exec
bridge, telemetry, and oneshot budget patch. The profile gives the parent and
children 2000 iterations within the 7200-second task deadline.

The configuration permits up to sixteen children. The pinned Hermes
delegation tool runs them serially and blocks the parent while they execute;
the harness preserves that scheduling behavior.

## Completion and artifacts

The harness stops task writers, freezes the candidate, then runs the verifier
in the isolated task runtime. Normal completion requires exit zero and a final
durable assistant turn with explicit `DONE` or a nonempty normal stop, without
pending tool calls or a fatal end reason. Empty or truncated stops without
DONE remain incomplete. Verifier reward is recorded independently.

Native sessions, model/tool events, execution and resource logs, the candidate
patch, trajectory, and verdict are retained. Child timing quality is recorded
separately from execution state; unavailable timestamps remain unknown.

## Run and report

Choose the Roadmap harness profile in your experiment, then use the shared
entrypoints:

```bash
python3 validate.py --config /path/to/experiment.toml
python3 run.py --config /path/to/experiment.toml
python3 scripts/analyze_roadmap_hermes_runs.py /path/to/run --arm subagent
python3 scripts/compare_roadmap_hermes_runs.py \
  --subagent /path/to/subagent-run --single-agent /path/to/single-agent-run
```

The analyzer accepts one or more runs for either arm. The comparator requires
paired arms. Both accept run paths or PBS IDs and export the five artifact
groups, metrics, and figures without copying full candidate workspaces.
Use `--runs-root` or `AB_RUNS_ROOT` for the run collection; PBS lookup also
recognizes `$SCRATCH/agent-bench/runs`.
