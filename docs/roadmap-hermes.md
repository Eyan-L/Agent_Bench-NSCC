# RoadmapBench with Hermes on NSCC

This harness uses the existing Roadmap task dataset, Git baseline, task SIF,
SGLang TP4 service, isolated verifier, and Roadmap result schema. Hermes runs
in a separate patched SIF and reaches the task container through the existing
exec bridge. The existing OpenClaw experiment results are unchanged.
The Roadmap profile enables Hermes' pinned `delegation` toolset with up to four
parallel children; the parent must integrate their results before completion.

## Build and upload the runtime

On **local Ubuntu**, use the local `~/Agent_Bench` checkout. Docker and
Apptainer (or Singularity) are required. The script builds the pinned
`v2026.5.16` Hermes image, applies the existing Agent_Bench bridge and
telemetry patches, converts the image to a SIF, and checks patch status.

```bash
cd ~/Agent_Bench
bash scripts/build_roadmap_hermes_sif_local.sh
scp dist/hermes_roadmap_v2026.5.16.sif \
  nscc:/scratch/users/ntu/n2602695/sif_images/
```

The PBS image preparation stage checks the uploaded file before launching the
agent. It does not pull or rebuild the Roadmap Hermes SIF on NSCC. The expected
path is `/scratch/users/ntu/n2602695/sif_images/hermes_roadmap_v2026.5.16.sif`.

## Smoke configuration

The smoke uses `fyn-2.5.0-roadmap`, seed 42, Qwen3.6-35B-A3B, and SGLang
TP4. This is the task retained in the final OpenClaw paired report after
`fal-2.0.0-roadmap` was replaced. On NSCC, inspect the plan first:

```bash
cd /home/users/ntu/n2602695/Agent_Bench
python3 run.py --config \
  experiments/qwen36_35b_a3b_roadmapbench_hermes_fyn_2_5_0_smoke.toml \
  --dry-run
```

After the SIF is uploaded, render and submit the PBS job from the NSCC
repository:

```bash
python3 run.py --config \
  experiments/qwen36_35b_a3b_roadmapbench_hermes_fyn_2_5_0_smoke.toml \
  --render-job > /tmp/roadmap-hermes-smoke.pbs
qsub /tmp/roadmap-hermes-smoke.pbs
```

The fixed 20-task Hermes arm uses the task IDs and seed 42 from the final
OpenClaw paired report. Its experiment file is
`experiments/qwen36_35b_a3b_roadmapbench_hermes_report20.toml` and its task
profile is `profiles/benchmarks/roadmap/roadmapbench_report20_hermes.json`.
Run the smoke first, then render this experiment with the same `--render-job`
command when the SIF and serving path have been checked on NSCC.

The harness preserves `hermes_events.jsonl`, exported
conversation state, `hermes_sessions.json` (session lineage and cumulative
token counters), `runtime_observability.json` (materialized child sessions and
their lifetimes), `trajectory.jsonl`, resource and OTel traces,
`candidate.patch`, `candidate_workspace/`, the Git baseline, and verifier
result. It stops the Hermes runtime and bridge, restarts the task container,
then freezes the candidate before verifier files are introduced.

The existing `scripts/analyze_roadmap_runs.py` can read the resulting Roadmap
manifest and trajectory. Token totals use the sum of Hermes session counters
when its state database is available; otherwise `token_accounting.json` marks
the event fallback as parent only. The pinned event patch instruments the
parent agent, so child model calls lack individual request latency. The full
session export and state database are retained in `hermes_state/` for review.
