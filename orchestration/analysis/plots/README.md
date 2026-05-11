# RQ1 / RQ2 / RQ3 plotting scripts

Renders matplotlib figures for the three research questions from per-run
trace data already produced by the eval driver
(`data/runs/<sweep>/<run_id>/`). Reuses the existing metric computers in
`orchestration/analysis/{composition,resource,evaluation}.py` — these scripts
only handle file I/O, grouping, and plotting.

## Layout

```
orchestration/analysis/plots/
├── _util.py              # shared loaders + style + workload-class resolver
├── rq1_composition.py    # 6 figures: AR vs trajectory composition
├── rq2_resources.py      # 6 figures: GPU/CPU/memory/retry bottlenecks
├── rq3_metrics.py        # 6 figures: TTFT/TPOT vs task latency, completion rate
└── plot_all.py           # runs all three, writes to <out>/{rq1,rq2,rq3}/
```

## Usage

```bash
# Render everything in one shot.
python -m orchestration.analysis.plots.plot_all \
    --runs-dir data/runs \
    --out-dir  data/figures

# Or run a single RQ.
python -m orchestration.analysis.plots.rq1_composition --runs-dir data/runs --out-dir data/figures/rq1
python -m orchestration.analysis.plots.rq2_resources   --runs-dir data/runs --out-dir data/figures/rq2
python -m orchestration.analysis.plots.rq3_metrics     --runs-dir data/runs --out-dir data/figures/rq3

# Add --also-pdf to emit vector copies alongside PNGs.
```

`--runs-dir` is walked recursively for any directory containing
`steps.jsonl`, so `data/runs/phase1/` and `data/runs/tb_kv_store/` mix
fine.

## What each figure shows

### RQ1 — execution composition

| File                                          | Says                                                                  |
| --------------------------------------------- | --------------------------------------------------------------------- |
| `rq1_time_fraction_by_step_type.png`          | Stacked bars, one per condition, fraction of wall time per step type. |
| `rq1_step_count_by_step_type.png`             | Same but step count instead of wall time.                             |
| `rq1_tool_and_stall_fractions.png`            | Grouped bars: tool-call ratio, non-LLM time fraction, wait/stall fraction. |
| `rq1_token_ratio.png`                         | Box plot of `prompt_tokens / completion_tokens` per trajectory (log y). |
| `rq1_composition_by_workload_class.png`       | Faceted version of F1.1 by workload class (only emitted when ≥2 classes are present). |
| `rq1_ar_vs_agent_rollup.png`                  | Headline contrast: AR conditions collapsed into one bar, agent into the other. |

### RQ2 — resource bottlenecks

| File                              | Says                                                                       |
| --------------------------------- | -------------------------------------------------------------------------- |
| `rq2_utilization_timelines.png`   | Per-condition GPU/CPU utilization over trajectory time; one line per run.  |
| `rq2_gpu_summary.png`             | Mean GPU util / p95 GPU util / GPU idle fraction per condition.            |
| `rq2_cpu_and_memory.png`          | Mean CPU util / p95 CPU util / peak host RSS per condition.                |
| `rq2_wasted_work.png`             | Mean attempts per trajectory + wasted-work fraction per condition.         |
| `rq2_task_latency.png`            | End-to-end task latency boxplot per condition.                             |
| `rq2_cpu_vs_gpu_scatter.png`      | Per-run scatter of mean CPU vs mean GPU utilization, colored by condition. |

### RQ3 — token-centric vs task-centric metrics

| File                                       | Says                                                                              |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| `rq3_ttft_mean.png`                        | Box of TTFT_mean per condition.                                                   |
| `rq3_tpot_mean.png`                        | Box of TPOT_mean per condition.                                                   |
| `rq3_task_latency.png`                     | Box of end-to-end task latency per condition.                                     |
| `rq3_ttft_mean_vs_task_latency.png`        | Scatter with Spearman ρ per condition + overall — does TTFT predict task latency? |
| `rq3_tpot_mean_vs_task_latency.png`        | Same for TPOT.                                                                    |
| `rq3_completion_rate.png`                  | Pass rate overall + pass rate within ≤300s/600s/1800s budgets.                   |

## Notes & caveats

- **TTFT/TPOT are approximated** when the trace doesn't carry a per-token
  timestamp. `orchestration/analysis/evaluation.py` falls back to
  `duration / (1 + completion_tokens)` for TTFT and `duration / completion_tokens`
  for TPOT. The shape per condition is still meaningful; the absolute
  values aren't comparable across providers.
- **`rss_MB` is host-wide**, not per-process — the sidecar sampler reports
  `cpu.mem_used_mb` from `/proc/meminfo`-style snapshots. Treat the
  memory metric as a coarse proxy until a per-process sampler lands.
- **OpenClaw step traces collapse** the agent's internal multi-step
  trajectory into one coarse `LLM_INFER` step per run (subtype
  `LLM_INFER_OPENCLAW_AGENTIC_COARSE`). RQ1 composition therefore looks
  100% LLM_INFER for OpenClaw runs until finer-grained step instrumentation
  is added (TODO referenced in `openclaw_agent.py`).
- **Workload class** defaults to `code_agent` for every existing condition.
  Once web-search / doc-analysis conditions are added, set
  `manifest["workload_class"]` (or extend `_util.workload_class_for`) so
  `rq1_composition_by_workload_class.png` becomes informative.
- **Low-N warning**: with fewer than 3 runs per condition, distributions
  and CIs are unreliable. Scripts still render but emit a warning.

## Dependencies

`pandas`, `numpy`, `matplotlib` are required. `seaborn`, `scipy` are
optional — without them you lose seaborn theming and the per-condition
Spearman annotations on the RQ3 scatter (annotations fall back to
"ρ N/A").
