---
name: openclaw_trace_analyzer
description: Compute the metrics required by the OpenClaw + SWE-bench validation plan from raw step traces, telemetry time-series, and benchmark verdicts. Use this skill when raw data has been collected and the user now needs analysis — composition breakdowns, resource utilization, TTFT/TPOT vs end-to-end latency comparisons, retry cost accounting, per-condition aggregation, and statistical comparison between AR and OpenClaw. Triggers on phrases like "analyze the traces", "compute the metrics", "break down step composition", "compare AR vs OpenClaw", "quantify wasted retries", "build the metric tables", "produce figures from traces", or any request for numbers that answer RQ1 / RQ2 / RQ3. Use this whenever the user has trace files and wants results, even if they don't name the skill.
---

# Purpose

Produce a metric-computation library and a reproducible analysis pipeline that turns raw traces into per-condition and per-task numbers addressing **RQ1 (composition)**, **RQ2 (resource bottlenecks)**, and **RQ3 (evaluation metrics)**.

The analyzer consumes artifacts from `openclaw_instrumentation` and `openclaw_experiment_runner` and produces CSVs, Parquet tables, and a small set of diagnostic figures that `openclaw_report_generator` can ingest.

# Core Deliverables

- Loaders for step-trace JSONL and telemetry Parquet/CSV, keyed by `run_id`.
- Four metric modules, one per metric family: composition, resource, evaluation, retry/fragmentation.
- A join utility that produces a long-format Parquet table with one row per `(run_id, metric_name, metric_value, kind)`.
- Per-condition aggregation (mean, median, p95, p99, stddev) with per-instance bootstrapping.
- A small set of diagnostic figures useful for sanity-checking the raw data.

# Required Metric Families (all must be implemented)

## Composition (RQ1)
- step count by `step_type`, per trajectory and per condition.
- time share by `step_type` (sum of durations / trajectory wall time).
- tool-call ratio = `count(TOOL_*) / count(all steps)`.
- non-LLM time fraction = `sum(non-LLM duration) / trajectory wall time`.
- waiting/stall fraction = `sum(WAIT_STALL duration) / trajectory wall time`.
- attempts per trajectory.
- input/output token ratio = `sum(prompt_tokens) / sum(completion_tokens)` over LLM steps.

## Resource (RQ2)
- GPU utilization: mean and p95 over trajectory wall time.
- SM occupancy / MFU / MBU: **only if DCGM or an equivalent counter source is available; otherwise mark as `NaN` with `reason=counter_unavailable`. Do not fabricate.**
- HBM bandwidth utilization: same conditional handling.
- GPU idle fraction while trajectory is active = fraction of trajectory wall time when `GPU_util < threshold` (default threshold 10%; threshold value is recorded as the inferred parameter).
- CPU utilization: mean and p95, total and per-subprocess.
- DRAM / RSS: peak and time-averaged.
- PCIe or network I/O: if telemetry is available.
- External wait ratio: fraction of trajectory time with high CPU wait or low GPU+CPU activity.

## Evaluation (RQ3)
- TTFT per LLM span, aggregated per condition: mean, p50, p95, p99.
- TPOT per LLM span, aggregated similarly.
- end-to-end task latency = trajectory wall time, per condition.
- p95 / p99 task latency per condition.
- pass rate per condition.
- pass rate under latency budget (e.g. 5, 10, 30 min — report all three).
- **Ranking-alignment test**: for each serving configuration, does the ranking of conditions by TTFT/TPOT match the ranking by end-to-end latency? This is the direct RQ3 probe — if they disagree, the headline is "token-level metrics are insufficient", not a side note.

## Retry / fragmentation
- failed-attempt cost fraction = `duration of non-final attempts / trajectory wall time`.
- wasted resource fraction from retries = `(GPU-time + CPU-time of non-final attempts) / (total GPU-time + CPU-time)`.
- LLM burst count per trajectory = count of contiguous runs of `LLM_INFER` steps separated by > T seconds of non-LLM time (default T = 1s; T is recorded as the inferred parameter).
- average LLM burst duration.
- inter-step idle gaps: histogram of gaps between consecutive steps within the same trajectory.

# Measured vs Inferred — Mark in Code

Each metric must carry a `kind` flag: `measured` (directly derived from a logged quantity) or `inferred` (computed via an assumption, e.g. a GPU-idle threshold, a burst window T, or a tokenizer-based arithmetic-intensity proxy). Inferred metrics must record the parameter values that produced them in a companion `params` dict.

Fabricating unavailable hardware counters (MFU without DCGM, HBM BW without profiler) is a severe bug, not a convenience. Downstream prose will inherit whatever confidence level the analyzer emits — overclaiming here propagates into overclaiming in the paper.

# Implementation Guidance — Concrete Modules

- `analysis/load.py` — `load_run(run_id) -> (steps: DataFrame, telemetry: DataFrame, verdict: dict)`.
- `analysis/composition.py` — step-type breakdowns, token ratios, tool-call ratio.
- `analysis/resource.py` — takes aligned telemetry and step frames, produces per-trajectory resource roll-ups.
- `analysis/evaluation.py` — TTFT/TPOT aggregation, e2e latency, ranking-alignment check.
- `analysis/retry.py` — retry cost and fragmentation metrics.
- `analysis/aggregate.py` — per-condition aggregation with bootstrap CIs.
- `analysis/tables.py` — emits the long-format Parquet plus a set of wide CSV tables, one per metric family.
- `analysis/figures.py` — a minimal set of diagnostic plots:
  - step-type stacked bar per condition,
  - GPU utilization timeline for one sampled trajectory per condition,
  - TTFT-vs-e2e-latency scatter per condition.

# Required Statistical Handling

- Do not rely on means alone. Report median, p95, p99, and bootstrapped 95% CIs over instances.
- Stratify tasks (post hoc is fine): search-heavy, edit-test-loop, retry-heavy. Stratification cuts must be defined from **trace-derived features**, not from external task metadata, and the cut definitions must be documented in a `strata.md` alongside the tables.
- **Paired comparisons by `instance_id`** when comparing conditions; avoid unpaired tests when paired data is available. The same instance may be much harder or easier than average, so unpaired comparisons conflate condition effects with instance-difficulty effects.

# Required Sanity Checks

- Sum of step-type durations within a trajectory matches trajectory wall time within 1%. (If it doesn't, the trace has gaps or overlaps and resource roll-ups will be wrong.)
- For AR conditions, trajectory wall time ≈ `LLM_INFER` duration (within tool-overhead budget). Large deviations mean the AR baseline is doing something it shouldn't.
- For OpenClaw, non-LLM time fraction > 0; if it's ~0, something is wrong with instrumentation on the tool path.
- Telemetry coverage per trajectory ≥ 95%; flag trajectories below this threshold and exclude them from resource aggregates (but keep them for composition and evaluation metrics that don't depend on telemetry).

These checks must run automatically on every `aggregate.py` invocation and produce a short `data_quality.md` next to the output tables.

# Out of Scope

- Collecting new data (in `openclaw_instrumentation`).
- Running experiments (in `openclaw_experiment_runner`).
- Narrative prose (in `openclaw_report_generator`).

# Writing Style

- Every metric has a precise definition at the top of its module as a docstring. A reviewer should be able to reconstruct the metric from the docstring alone.
- Every function returns a DataFrame with named columns; no tuples of numbers.
- Unavailable data is `NaN` with a `reason` field, never a silent zero.
- No hard-coded file paths — everything flows from a `run_id` plus a root directory.
