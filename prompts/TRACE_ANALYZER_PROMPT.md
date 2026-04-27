# OpenClaw + SWE-bench Trace Analyzer Prompt

Please use the `openclaw_trace_analyzer` skill.

Raw data is collected. I need the analysis pipeline that computes metrics answering **RQ1 (composition)**, **RQ2 (resource bottlenecks)**, and **RQ3 (evaluation)**.

## Inputs
- Step-trace JSONL per `run_id`.
- Telemetry Parquet/CSV per `run_id`, aligned on monotonic clock.
- SWE-bench verdict per `run_id`.

## Metric families to compute

### Composition (RQ1)
Step-type counts and time share; tool-call ratio; non-LLM time fraction; waiting/stall fraction; attempts per trajectory; input/output token ratio.

### Resource (RQ2)
GPU utilization mean + p95; GPU idle fraction during trajectory (threshold-based, mark as inferred with threshold value); SM occupancy / MFU / MBU / HBM BW **only if counters are available — otherwise `NaN` with `reason=counter_unavailable`, do not fabricate**; CPU utilization mean + p95 (total and per-subprocess); DRAM/RSS peak + time-averaged; PCIe / network I/O if available.

### Evaluation (RQ3)
TTFT mean/p50/p95/p99; TPOT similarly; end-to-end task latency; p95/p99 task latency; pass rate; pass rate under latency budgets of 5, 10, and 30 min; **ranking-alignment test** between TTFT/TPOT ordering and end-to-end ordering (this is the core RQ3 probe).

### Retry / fragmentation
Failed-attempt cost fraction; wasted resource fraction from retries; LLM burst count and avg duration (default T = 1s, recorded as inferred parameter); inter-step idle-gap histogram.

## Hard requirements

1. Every metric is labeled `measured` or `inferred` in the output schema; inferred metrics carry their parameter values in a `params` dict.
2. Do **not** fabricate unavailable hardware counters — mark `NaN` with `reason`.
3. **Paired comparisons by `instance_id`** when comparing conditions; bootstrapped 95% CIs across instances.
4. Long-format Parquet: one row per `(run_id, metric_name, metric_value, kind)` plus metadata.
5. Wide CSV tables per metric family for quick inspection.
6. Diagnostic figures:
   - step-type stacked bar per condition,
   - GPU util timeline for one sample trajectory per condition,
   - TTFT-vs-e2e scatter per condition.

## Stratification
Post hoc stratification from trace-derived features (search-heavy, edit-test-loop, retry-heavy) is fine. Document the cutoffs in `strata.md` alongside the tables.

## Validation

Describe how these checks will pass automatically on every aggregate invocation:

- Sum of step-type durations ≈ trajectory wall time within 1%.
- AR trajectory wall time ≈ `LLM_INFER` duration (within tool-overhead budget).
- OpenClaw non-LLM time fraction > 0.
- Telemetry coverage ≥ 95% per trajectory; flag trajectories below this and exclude only from resource aggregates, not from composition/evaluation aggregates.

Emit a short `data_quality.md` next to the tables.

## Out of scope
- New data collection.
- Running experiments.
- Prose / report writing.

## Output format
- Python modules with precise metric definitions as docstrings.
- DataFrame-returning functions; `NaN` with `reason` for unavailable data.

## Style
- No silent zeros for missing data.
- No means-only summaries; include distribution summaries (median, p95, p99, CI).
- No hard-coded paths — everything flows from `run_id` + a root dir.
