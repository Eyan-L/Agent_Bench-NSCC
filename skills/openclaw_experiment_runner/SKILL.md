---
name: openclaw_experiment_runner
description: Build the experiment execution harness for the OpenClaw + SWE-bench validation study. Use this skill when the user has a validation plan and instrumentation in place, and now needs driver scripts to actually run the conditions — AR-issue-only, AR-static-context, OpenClaw code agent, OpenClaw ablations, and serving-configuration sweeps — on SWE-bench Verified-mini (Phase 1) and Verified (Phase 2). Triggers on phrases like "run the experiments", "build the experiment driver", "orchestrate the SWE-bench evaluation", "sweep serving configs", "launch Phase 1 / Phase 2", "integrate with the SWE-bench harness", or requests for resume/retry and manifest generation. Use this whenever the user is moving from plan to execution on this study, even if they don't say "experiment runner" explicitly.
---

# Purpose

Produce the driver code and configuration files that execute the phased evaluation described in the validation plan. The runner consumes the plan's **Baselines and Ablations** section and produces per-run artifacts that `openclaw_trace_analyzer` can consume without further reshaping.

# Core Deliverables

- Config schema for `(condition × instance × seed × serving_config)` runs.
- Concrete condition implementations: AR-issue-only, AR-static-context, OpenClaw code agent, plus at least one agent ablation and one serving-configuration sweep.
- SWE-bench harness integration: checkout, environment provisioning, test execution, verdict capture.
- Placement: OpenClaw runs inside the SWE-bench task container with workspace set to the repo root (unless the user overrides).
- Idempotent driver with resume, skip-if-output-exists, and per-run manifests.
- Parallelism controls gated on GPU memory and CPU availability.

# Units and Identifiers

Every run is identified by a deterministic

```
run_id = hash(condition_id, instance_id, seed, config_id, model_name, plan_version)
```

The `run_id` is the only link between raw traces, benchmark verdicts, and analysis outputs — generate it once and carry it everywhere. If the hash inputs change for any reason (e.g. plan version bump), the old `run_id` is stale; do not reuse it.

# Required Conditions

Implement these conditions unambiguously. Each one exists as its own module with a clear interface `run(instance, config) -> RunResult`.

## AR-issue-only
Single-shot or fixed-k AR generations from the issue text alone. No repo interaction. One `LLM_INFER` step per turn, nested in a degenerate one-attempt trajectory.

## AR-static-context
Issue text plus an offline-constructed static context window (retrieval, file skeletons, test stubs). Static-context construction runs **before** serving and is logged as a distinct `TOOL_READ` step with `offline=true`, so it is never conflated with agentic repo exploration.

## OpenClaw code agent
Full multi-step trajectory with repo interaction, code edits, test execution, and retries. Workspace is the checked-out repo root inside the SWE-bench container. Tool profile restricted per the OpenClaw configuration guidance (see "OpenClaw Placement Defaults" below).

## Agent ablations (at least one required)
Pick one of:
- `no-retry` variant: attempt cap = 1.
- `no-test-loop` variant: patch proposed without local validation.
- `reduced-tool-profile` variant: read-only tools only.

Additional ablations are welcome; each one must be gated behind a flag and must keep all other dimensions fixed.

## Serving-configuration sweep (at least three points, for RQ3)
At least three configurations over the same conditions:
- `latency-oriented`: low batch, aggressive KV caching, small max concurrent requests.
- `throughput-oriented`: larger batch, higher concurrency, longer scheduling windows.
- `context-cache-oriented`: prefix cache enabled, long-context friendly, biased toward reusing system/context tokens across turns.

The sweep is the core RQ3 probe — it tests whether token-level wins (TTFT/TPOT) correlate with task-level wins (end-to-end latency, pass rate).

# Fairness Constraints to Enforce in Code

The runner must **hard-enforce** these; a violation is a loud failure, not a warning:

- Same benchmark instance set across conditions in a given sweep.
- Same model family across conditions; log any necessary tokenizer differences.
- Same serving engine endpoint used for AR and OpenClaw LLM calls within a sweep.
- Same repo snapshot (pinned commit) and same task environment (same container image digest).
- Same wall-clock time budget per task per condition.
- Same max token budget policy; if the agent uses cumulative budget across steps, log that cumulative value explicitly so downstream analysis can compare fairly.

Conditions that cannot satisfy a constraint must fail loud. Silent degradation makes the comparison meaningless.

# Implementation Guidance — Concrete Modules

- `runner/config.py` — `RunConfig`, `ModelConfig`, `ServingConfig`, `AgentConfig`, `ConditionConfig` dataclasses; YAML loader.
- `runner/conditions/ar_issue_only.py`
- `runner/conditions/ar_static_context.py`
- `runner/conditions/openclaw_agent.py`
- `runner/conditions/ablations/*.py`
- `runner/harness.py` — SWE-bench harness wrapper (container lifecycle, test execution, verdict parse). **Use the official SWE-bench evaluation harness; do not reimplement patch application or test invocation.**
- `runner/driver.py` — the main loop: for each `(condition, instance, seed, serving)` combination, check manifest, skip if done, otherwise run, write manifest.
- `runner/manifest.py` — per-run manifest JSON: `run_id`, `status`, paths to step trace, telemetry, verdict, exit code, wall start/end, notes.
- `runner/sweeps/phase0.yaml`, `runner/sweeps/phase1.yaml`, `runner/sweeps/phase2.yaml`.
- `scripts/phase0.sh`, `scripts/phase1.sh`, `scripts/phase2.sh`.

# SWE-bench Integration

- Use the official SWE-bench evaluation harness for the final verdict.
- Apply patches produced by the condition to the pinned commit; run the harness-provided test invocation; capture its verdict as the pass/fail signal.
- Timeouts at the harness level are recorded as `TIMEOUT`, **not** silently as `FAIL`.
- Distinguish four terminal outcomes:
  - `PASS` — harness says tests pass.
  - `FAIL` — harness says tests fail.
  - `TIMEOUT` — budget exceeded before a verdict.
  - `ERROR` — infra or driver failure (container didn't start, serving endpoint unreachable, etc.) — this is not signal.

Only `ERROR` is auto-retried. `FAIL` and `TIMEOUT` are real signal and must not be retried silently.

# OpenClaw Placement Defaults

Unless the user overrides:

- OpenClaw process runs **inside** the SWE-bench task container for the pilot.
- Workspace root is the checked-out repository root.
- No extra sandbox layer.
- Browser, messaging, canvas, and non-coding MCP bundles are **disabled**; tool profile is restricted to file read/write, shell, test execution, and search.
- Telemetry collector from `openclaw_instrumentation` runs as a sidecar in the same container; if that fails in the target environment, fall back to host-level sampling and flag the drift in the manifest.

# Resume / Retry / Flakiness

- Skip runs whose manifest already records a terminal status.
- Re-run `ERROR` status by default (infra flake); do NOT auto-re-run `FAIL` or `TIMEOUT`.
- Seeds are fixed per `run_id`; re-runs use the same seed unless the user explicitly requests a seed sweep (in which case seed becomes part of the hash inputs).

# Parallelism

- Cap concurrent task containers by `min(available_gpus / per_condition_gpu_demand, available_cpu_cores / 4)`.
- Serving-config sweeps should not run concurrent conditions against the same serving endpoint unless the endpoint's concurrency is itself part of the sweep.

# Required Sanity Checks Before Phase 1

- A single AR-issue-only run on one Verified-mini task produces: a manifest, a step trace with exactly one `LLM_INFER`, a telemetry file aligned to it, and a verdict.
- A single OpenClaw run on the same task produces: a manifest, a step trace with ≥1 attempt and mixed step types, a telemetry file covering the whole trajectory, and a verdict.
- Resuming the driver after killing it mid-sweep does not re-execute completed runs.
- Serving-config sweep on Phase 0 produces three manifests with distinct `config_id` values and verdicts that are actually comparable (same instance, same seed).

# Out of Scope

- Instrumentation internals (in `openclaw_instrumentation`).
- Metric computation (in `openclaw_trace_analyzer`).
- Plan writing (in `openclaw_swebench_code_agent_plan`).
- Report prose (in `openclaw_report_generator`).

# Writing Style

- Produce runnable Python and bash. Prefer `subprocess` + explicit container commands over opaque SDKs.
- Keep serving-config sweeps declarative (YAML), not scattered across code.
- Every failure mode maps to one of the four terminal outcomes; no silent drops, no `except: pass`.
- The driver must be restartable from any point without manual cleanup.
