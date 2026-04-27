# OpenClaw + SWE-bench Experiment Runner Prompt

Please use the `openclaw_experiment_runner` skill.

The validation plan and instrumentation are ready. I need the driver code that actually runs conditions on SWE-bench Verified-mini (Phase 1) and Verified (Phase 2).

## Scope
- Code agent only.
- OpenClaw inside the SWE-bench task container by default; workspace = repo root.
- Official SWE-bench harness for final verdicts.
- Artifacts per run: step trace, telemetry, manifest, verdict.

## Conditions to implement

1. **AR-issue-only** — single-shot or fixed-k AR from issue text.
2. **AR-static-context** — issue + offline static context, logged as a distinct `TOOL_READ` step with `offline=true`.
3. **OpenClaw code agent** — full trajectory with repo interaction and retries.
4. **At least one ablation** — pick from: `no-retry` (attempt cap = 1), `no-test-loop`, `reduced-tool-profile` (read-only).
5. **At least one serving-configuration sweep** with three points: `latency-oriented`, `throughput-oriented`, `context-cache-oriented`.

## Fairness (enforced in code, not just documented)

- Same instance set per sweep.
- Same model family across conditions.
- Same serving endpoint across AR and OpenClaw within a sweep.
- Same pinned commit + container image digest.
- Same wall-clock budget per task.
- Same token budget policy; if the agent uses a cumulative budget, log it explicitly.

Fail loud if a constraint cannot be satisfied. Do not silently degrade.

## Outcomes

Four terminal states per run: `PASS`, `FAIL`, `TIMEOUT`, `ERROR`.

- `ERROR` is infra and auto-retried.
- `FAIL` and `TIMEOUT` are real signal and are **not** auto-retried.

## Hard requirements

1. Deterministic `run_id = hash(condition_id, instance_id, seed, config_id, model_name, plan_version)`.
2. Idempotent driver: skip runs whose manifest has a terminal status; restart from any kill point without manual cleanup.
3. YAML sweep configs: `phase0.yaml`, `phase1.yaml`, `phase2.yaml`.
4. Parallelism capped by `min(gpus/demand, cpu_cores/4)`.
5. Per-run manifest JSON with paths to step trace, telemetry, verdict, exit code, wall start/end, notes.

## Validation before Phase 1

Describe how these checks will pass:

- One `AR-issue-only` run produces a manifest with exactly one `LLM_INFER` step and a verdict.
- One OpenClaw run on the same task produces a trajectory with mixed step types and a verdict.
- Re-running the driver after a kill does not repeat completed runs.
- Serving-config sweep on Phase 0 produces three manifests with distinct `config_id` values on the same instance+seed.

## Out of scope
- Instrumentation internals.
- Metric computation.
- Report writing.

## Output format
- Python + bash. YAML for sweep configs.
- Clear SWE-bench harness integration boundary; do not reimplement patch application or test invocation.

## Style
- Explicit `subprocess` + container commands over opaque SDKs.
- Declarative sweeps.
- No silent failure modes; no `except: pass`.
