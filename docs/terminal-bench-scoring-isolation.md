# Terminal-Bench Scoring Isolation

This document records how Agent Bench keeps Terminal-Bench verifier assets out
of the agent-visible solve stage, and how the harness makes them available only
when final scoring starts.

## Goal

Terminal-Bench tasks ship a vendored `tests/` directory. Those tests are the
verifier and should not be visible to the agent while it is solving the task.
The intended lifecycle is:

1. Resolve the task workspace from explicit `task.toml` config, image
   metadata, or the materialized task Dockerfile `WORKDIR`, falling back to
   `/app`.
2. Start the task container with the agent workspace mounted at that task
   workspace.
3. Let the agent modify the task workspace, install dependencies, and start
   any required services.
4. After the agent declares completion, inject the vendored task tests into
   `/tests`.
5. Run the verifier entrypoint, usually `bash /tests/test.sh`, with cwd set to
   the task workspace.
6. Read `/logs/verifier/reward.txt` from the host bind and convert it into the
   normal PASS/FAIL/TIMEOUT/ERROR result shape.

This preserves Terminal-Bench's single-task-container behavior, which matters
for tasks that require long-running services started by the agent, such as
`kv-store-grpc`.

## Solve-Time Container Layout

During `prepare_environment`, Terminal-Bench task containers expose only:

| Container path | Mode | Purpose |
|---|---:|---|
| task/image workdir, usually `/app` | rw | Agent workspace and task implementation files. |
| `/logs/verifier` | rw | Host-visible scoring output directory. |

`/tests` is intentionally not bind-mounted and not copied into the task
container during the solve stage.

The agent-facing prompts and OpenClaw bootstrap files also avoid telling the
agent about `/tests/test.sh` or `/logs/verifier/reward.txt`. Agents are
instructed to finish the task and declare `DONE`; final scoring is the harness's
responsibility.

Hermes and OpenHands still mount the same host workspace at `/app` inside the
agent runtime container. Their exec bridge maps that agent-visible `/app` to
the task container's resolved workspace, so task execution and scoring use the
same container path even for tasks such as `prove-plus-comm` whose Dockerfile
sets `WORKDIR /workspace`.

## Scoring-Time Test Injection

The loader still records:

- `tests_host_path`: the vendored task `tests/` directory on the host.
- `verifier_entrypoint`: `/tests/test.sh`.
- `reward_file`: `/logs/verifier/reward.txt`.

`run_tests` is the only stage that consumes `tests_host_path`.

Before running the verifier, all Terminal-Bench harnesses:

1. Validate that `instance_data["tests_host_path"]` exists and is a directory.
2. Remove any previous `/tests` directory from the live task container.
3. Copy the vendored tests into `/tests`.
4. Delete stale host-side `reward.txt`.
5. Run the verifier entrypoint from the resolved task workspace.
6. Read the newly written reward file.

Docker OpenClaw/ReAct runs use `docker cp` for the directory injection.
Hermes/OpenHands bridge harnesses use the shared Docker container abstraction.
Singularity runs stream the directory with `tar` into the live instance because
the existing Singularity `copy_into` helper is file-oriented.

## Why `/tests` Is Not Made Read-Only

Some Terminal-Bench verifiers use `/tests` as a verifier-local working area.
For example, `compile-compcert` writes probe binaries under `/tests`. Making the
directory read-only after injection would break those tasks.

The isolation boundary in this implementation is therefore time-based:

- hidden during the agent solve stage;
- present during final scoring.

## Known Boundary

This is not a fully adversarial scoring sandbox. The verifier runs in the same
task container that the agent used, so an intentionally malicious background
process left by the agent could observe `/tests` after scoring begins.

Stronger isolation would require a different design, such as replaying the
agent-produced filesystem state into a fresh scoring container while preserving
only the task state that the benchmark expects. That is harder for
Terminal-Bench than for patch-based benchmarks because some tasks rely on
running services and other live container state.

## Files To Check When Changing This

- `agent_bench/benchmarks/terminal_bench/app.py`
  - Docker and Singularity `run_tests` implementations.
- `agent_bench/benchmarks/terminal_bench/agent_harness_base.py`
  - Shared Hermes/OpenHands scoring flow.
- `agent_bench/benchmarks/terminal_bench/conditions/`
  - Agent-facing instructions should not ask the agent to run hidden tests.
- `agent_bench/benchmarks/terminal_bench/openclaw/agents/`
  - OpenClaw bootstrap/path rules should describe only solve-time paths.
- `profiles/harnesses/openclaw_terminal_bench.yaml`
  - Should not register a verifier skill for Terminal-Bench.
- `tests/test_agent_otel_terminal_bench.py`
  - Static guardrails for the scoring-only tests behavior.

## Verification

At minimum, run:

```bash
git diff --check
python3 -m py_compile \
  agent_bench/benchmarks/terminal_bench/app.py \
  agent_bench/benchmarks/terminal_bench/agent_harness_base.py \
  agent_bench/benchmarks/terminal_bench/conditions/openhands_agent.py \
  agent_bench/benchmarks/terminal_bench/conditions/hermes_agent.py \
  agent_bench/benchmarks/terminal_bench/conditions/openclaw_agent.py \
  agent_bench/benchmarks/terminal_bench/conditions/react.py \
  agent_bench/benchmarks/terminal_bench/loader.py
python3 -m pytest tests/test_agent_otel_terminal_bench.py -q
```

If `pytest` is unavailable in the local environment, the static guards in that
test file should still be kept in sync with the intended behavior.
