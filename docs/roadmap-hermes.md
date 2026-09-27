# RoadmapBench with Hermes on NSCC

Hermes uses the fixed RoadmapBench tasks and Git baselines, the existing task
SIF and isolated verifier, and Qwen3.6-35B-A3B served by SGLang TP4. Its
patched SIF reaches the task container through the exec bridge. The Roadmap
profile allows up to 16 parallel delegated agents.

## Build and upload the runtime

On **local Ubuntu**, use the local `~/Agent_Bench` checkout. Docker and
Apptainer (or Singularity) are required. The script builds the pinned
`v2026.5.16` Hermes image, applies the existing Agent_Bench bridge and
telemetry patches, converts the image to a SIF, and checks patch status.

```bash
cd ~/Agent_Bench
bash scripts/build_roadmap_hermes_sif_local.sh
scp dist/hermes_roadmap_v2026.5.16.sif \
  nscc:/scratch/users/ntu/n2602695/sif_images/hermes_roadmap_v2026.5.16.sif.incoming
ssh nscc 'module load singularity/4.3.1 && bash /home/users/ntu/n2602695/Agent_Bench/scripts/install_roadmap_hermes_sif_nscc.sh'
```

PBS uses `/scratch/users/ntu/n2602695/sif_images/hermes_roadmap_v2026.5.16.sif`.
The install script verifies the incoming SIF, keeps the old image as `.back`,
and atomically replaces the unchanged path. The SIF patch makes oneshot honor
`agent.max_turns`. The current Roadmap harness profile sets `max_turns` and
`max_steps` to 2000, with a 7200-second task wall limit. Earlier Hermes runs
used 600 iterations; retain that distinction when analyzing results.

## Run and analyze

For new experiments, set an absolute `output_dir` below
`$SCRATCH/agent-bench/runs` in the experiment TOML. On NSCC:

```bash
cd /home/users/ntu/n2602695/Agent_Bench
python3 validate.py --config <experiment.toml>
python3 run.py --config <experiment.toml> --dry-run
python3 run.py --config <experiment.toml> --render-job > /tmp/roadmap-hermes.pbs
mkdir -p "$SCRATCH/agent-bench/pbs-logs/roadmap-hermes"
qsub -o "$SCRATCH/agent-bench/pbs-logs/roadmap-hermes" \
  -e "$SCRATCH/agent-bench/pbs-logs/roadmap-hermes" \
  /tmp/roadmap-hermes.pbs
```

The harness freezes `candidate.patch` and `candidate_workspace/` before
running the isolated verifier. It retains Hermes sessions and model/tool
events, execution and resource logs, `trajectory.jsonl`, and the verdict.
Normal completion requires oneshot exit 0 and a final durable assistant turn
with either explicit `DONE` or a nonempty normal `stop`, without a pending tool
call or fatal end reason. An empty stop or truncated turn without `DONE` is
incomplete. Verifier reward is recorded separately.

The Hermes analyzer accepts one or more PBS IDs for one arm; the comparator
requires both paired arms:

```bash
python3 scripts/analyze_roadmap_hermes_runs.py <subagent-pbs-jobs...> \
  --arm subagent --output-dir "$SCRATCH/agent-bench/reports/roadmap"
python3 scripts/compare_roadmap_hermes_runs.py \
  --subagent <subagent-pbs-jobs...> \
  --single-agent <single-agent-pbs-jobs...> \
  --output-dir "$SCRATCH/agent-bench/reports/roadmap"
```

PBS IDs resolve under `AB_RUNS_ROOT` when set, otherwise under
`$SCRATCH/agent-bench/runs` when it exists; use `--runs-root` for another
location. Both tools use materialized child sessions for subagent usage and
`agent_execution_seconds` for latency. Their portable bundles include durable
sessions, timelines and traces, execution evidence, final candidate and score,
and figures.
