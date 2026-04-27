# OpenClaw + SWE-bench Instrumentation Prompt

Please use the `openclaw_instrumentation` skill.

I already have (or am building) a validation plan for an OpenClaw + SWE-bench code-agent study. I need the **instrumentation layer** implemented — the code that produces the raw traces the plan describes.

## Scope
- Code-agent workstream only; no web-search agent or document-analysis agent.
- Target runtime: OpenClaw inside the SWE-bench task container.
- Target LLM: a controllable serving endpoint used for both AR and agentic conditions.
- Output is **code (Python)**, not a plan.

## Hard requirements

1. Implement the step-trace logger with the full schema:
   - `trajectory_id`, `attempt_id`, `step_id`, `parent_step_id`
   - `step_type` (one of `LLM_INFER`, `TOOL_READ`, `TOOL_WRITE`, `TOOL_EXEC`, `WAIT_STALL`), `step_subtype`
   - `tool_name`, `command`
   - `start_ts_mono`, `end_ts_mono`, `start_ts_wall`, `end_ts_wall`, `duration_ms`
   - `prompt_tokens`, `completion_tokens`, `ttft_ms`, `tpot_ms`
   - `exit_code`, `error_type`
   - `repo_id`, `instance_id`, `config_id`, `model_name`, `schema_version`

2. LLM wrapper for an OpenAI-compatible streaming client that captures TTFT from the first non-empty token and TPOT as mean inter-token time. Use API `usage` block for token counts when available; fall back to tokenizer with a `token_source=tokenizer` flag.

3. OpenClaw integration: **provide both a hook-based path and a wrapper-based path**, and state the assumptions each depends on. If OpenClaw's lifecycle API is uncertain to you, pick the wrapper path as the default.

4. Sidecar resource telemetry: NVML GPU, psutil CPU/memory, `/proc` supplement, poll at 100–500 ms (default 200 ms), aligned on `CLOCK_MONOTONIC_RAW`.

5. Writers: JSONL for step events, Parquet (fallback CSV) for telemetry.

6. Alignment utility: join step events with telemetry time-series on monotonic clock; emit per-step resource roll-ups.

7. AR baselines (`AR-issue-only`, `AR-static-context`) must use the **same** logger with a degenerate one-attempt trajectory. Downstream analysis must not branch on condition type to read the data.

## Validation

Before returning, describe how these sanity checks will pass:

- Synthetic trajectory durations roll up correctly within 5 ms.
- TTFT matches an external stopwatch within 50 ms.
- Telemetry coverage ≥ 95% per trajectory.
- Forced crash mid-trajectory leaves a recoverable JSONL with `partial=true` on the last attempt.

## Out of scope
- Metric computation, experiment driver, report generation.

## Output format
- Python modules with type hints and docstrings.
- A short `README.md` summarizing integration steps, clock/buffer defaults, and open assumptions about OpenClaw's surface.

## Style
- Stdlib + `psutil` + `pynvml` (+ `pyarrow` optional).
- State assumptions explicitly.
- Keep raw events free of inferred quantities — those belong in the analyzer.
