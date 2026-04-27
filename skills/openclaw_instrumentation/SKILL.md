---
name: openclaw_instrumentation
description: Implement the instrumentation and tracing layer for an OpenClaw + SWE-bench code-agent validation study. Use this skill when the user has a validation plan and now needs concrete code that produces the raw traces the plan calls for — step-trace loggers, LLM wrappers that capture TTFT/TPOT, resource telemetry collectors for GPU/CPU/memory, and timestamp alignment across the three observability layers. Triggers on phrases like "instrument OpenClaw", "build the trace logger", "capture step traces", "collect GPU telemetry for SWE-bench runs", "emit events in the required schema", or requests to produce the JSONL / Parquet writers described in the validation plan. Use this even when the user says things like "the tracing code", "the logger", or "the telemetry pipeline" without full context — assume they mean the trace pipeline for the code-agent study unless they clearly say otherwise.
---

# Purpose

Use this skill to produce a working instrumentation stack for the OpenClaw + SWE-bench code-agent validation study. The output is **code plus a small amount of integration documentation**, not a plan.

This skill assumes the validation plan already exists (built via `openclaw_swebench_code_agent_plan`). Your job is to turn the plan's **Instrumentation Plan**, **Raw Trace Schema**, and **Metrics** sections into runnable software that emits artifacts which the downstream analyzer can consume without any schema negotiation.

# Core Deliverables

Your response must produce, or be structured so the user can ask for, the following artifacts:

1. A step-trace logger library (Python) implementing the raw schema from the plan.
2. A wrapper around the LLM serving client that captures TTFT, TPOT, prompt/completion token counts, and streams cleanly.
3. An OpenClaw integration layer that emits events with the required step taxonomy.
4. A resource telemetry collector (GPU via NVML, CPU/memory via psutil, subprocess via /proc) that runs as a sidecar.
5. A timestamp-alignment utility that merges step events with resource time-series.
6. Writer backends: JSONL for step events, Parquet (fallback CSV) for telemetry time-series.

# Required Step Taxonomy (re-state before coding)

Before writing code, re-state the step taxonomy so events are consistent across AR and OpenClaw conditions:

- `LLM_INFER`
- `TOOL_READ`
- `TOOL_WRITE`
- `TOOL_EXEC`
- `WAIT_STALL`

Subtypes are permitted (e.g. `TOOL_EXEC_TEST`, `TOOL_READ_SEARCH`, `WAIT_TOOL_RETURN`) but must roll up to the top-level categories above. The logger must reject unknown top-level types at runtime.

# Required Raw Schema Fields

Every step event must carry at minimum:

- `trajectory_id`, `attempt_id`, `step_id`, `parent_step_id`
- `step_type`, `step_subtype` (optional)
- `tool_name`, `command` (if applicable)
- `start_ts_mono`, `end_ts_mono`, `start_ts_wall`, `end_ts_wall`, `duration_ms`
- `prompt_tokens`, `completion_tokens`, `ttft_ms`, `tpot_ms` (LLM only)
- `exit_code`, `error_type`
- `repo_id`, `instance_id`, `config_id`, `model_name`
- `schema_version`

Use `CLOCK_MONOTONIC_RAW` for duration math; keep wall-clock separately for alignment with external telemetry and log correlation. Never use wall-clock for latency math — containers, NTP corrections, and namespace boundaries make it unreliable.

# Key Design Decisions to State Explicitly

When producing the code, call out these decisions in a short section at the top of the integration README:

- **Clock source**: `time.monotonic_ns()` for durations, `time.time_ns()` for wall-clock.
- **Buffering**: in-memory ring buffer flushed to disk every N events or every T seconds, whichever first. Defaults: N=64, T=1s.
- **Crash safety**: fsync on trajectory termination; partial files must be recoverable and carry a `partial=true` marker on the last open attempt.
- **Placement**: one telemetry sidecar process per SWE-bench task container, pinned via cgroup where possible.
- **Schema evolution**: bump `schema_version` on any breaking change, never mutate old files.

# Implementation Guidance — Concrete Modules

Produce the following files unless the user narrows the scope:

- `trace/schema.py` — dataclasses or Pydantic models for events and telemetry samples; validator for step types; single source of truth for field names.
- `trace/logger.py` — `StepLogger` with `start_step(...)`, `end_step(...)`, and a `step(...)` context manager. Thread-safe. Writes JSONL.
- `trace/llm_wrapper.py` — wraps an OpenAI-compatible streaming client. Captures TTFT from the first non-empty token event and TPOT as mean inter-token time across the completion. Reports `prompt_tokens` and `completion_tokens` from the API's usage block when available; falls back to tokenizer-based counting with a `token_source=tokenizer` flag when not.
- `trace/openclaw_hooks.py` — integration surface for OpenClaw. Provide **two paths**: a hook-based path if OpenClaw exposes lifecycle callbacks (tool dispatch, LLM call, retry boundary), and a wrapper-based path that wraps the tool dispatcher and LLM client at the adapter boundary. Document which assumptions each path depends on.
- `trace/telemetry.py` — `ResourceCollector` sidecar. NVML via `pynvml` for GPU utilization, SM occupancy where available, memory used, HBM bandwidth if DCGM is available; `psutil` for CPU utilization, RSS, per-subprocess CPU time; `/proc/self/status` supplement for RSS peaks. Poll at 100–500 ms (default 200 ms).
- `trace/align.py` — merges step events with telemetry samples on the monotonic clock; computes per-step and per-trajectory resource roll-ups (means, maxes, idle fraction).
- `trace/writers.py` — JSONL step writer, Parquet telemetry writer (via `pyarrow`); graceful fallback to CSV if `pyarrow` is unavailable, with a warning logged once.

# AR-Side Instrumentation

The same logger must serve AR baselines. For `AR-issue-only` and `AR-static-context`:

- One turn = one `LLM_INFER` step nested in a degenerate one-attempt trajectory.
- Static-context build, if it runs offline before serving, is logged as a separate `TOOL_READ` step with `offline=true` **or** is excluded entirely with a note. State the choice in the integration README; downstream analysis must not have to guess.
- Trajectory and attempt IDs must still be populated so downstream analysis treats AR as a degenerate trajectory rather than a separate data model. This is what makes per-condition comparisons clean in `openclaw_trace_analyzer`.

# Telemetry Alignment

State explicitly in the code and docs:

- Monotonic clock is the join key.
- The telemetry process must stamp samples with the same clock source as the step logger. If it runs in a separate process, publish a sync beacon (a no-op step event at startup) and record the measured offset.
- On the SWE-bench container boundary, confirm `CLOCK_MONOTONIC_RAW` is not reset. PID/mount namespacing usually preserves it; call out and document if the user's environment differs.

# Required Sanity Checks

Before claiming the instrumentation is ready, the code or docs must support these checks:

- A synthetic trajectory with known step durations rolls up to the expected totals within 5 ms error.
- TTFT on a streaming LLM call matches `first_token_ts - request_ts` measured by an external stopwatch to within 50 ms.
- Telemetry samples and step events align: no step should have an empty resource window when the telemetry collector was running.
- A forced crash mid-trajectory leaves a recoverable JSONL with a `partial=true` marker on the last attempt.

Include these as a small `tests/` directory or as a `validate_instrumentation.py` script.

# Out of Scope

Do NOT produce:

- Analysis or metrics computation (handled by `openclaw_trace_analyzer`).
- Experiment driver logic (handled by `openclaw_experiment_runner`).
- Report prose (handled by `openclaw_report_generator`).

# Writing Style

- Produce runnable code with type hints and docstrings.
- State assumptions about OpenClaw's public surface explicitly; if unknown, provide both the hook-based and wrapper-based path and let the user pick.
- Separate directly measurable quantities (TTFT, step duration, prompt tokens) from inferred ones (arithmetic intensity, idle-fraction thresholds). Keep the inferred ones out of the raw trace — they belong in the analyzer.
- Prefer stdlib + `psutil` + `pynvml` (+ `pyarrow` optional); avoid pulling heavy frameworks unless the user asks.
