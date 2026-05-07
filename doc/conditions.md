# Adding a new condition

A *condition* is the procedure the agent follows to attempt a task. The current
conditions:

| Condition | Module | What it does |
|---|---|---|
| `ar_issue_only`     | `eval/conditions/code_agent/ar_issue_only.py`     | One-shot LLM call from the issue text. No tools. |
| `ar_static_context` | `eval/conditions/code_agent/ar_static_context.py` | One-shot + offline BM25 retrieval. |
| `openclaw_agent`    | `eval/conditions/code_agent/openclaw_agent.py`    | Multi-turn agent loop driven by OpenClaw. |
| `tb_react`          | `eval/conditions/code_agent/terminal_bench/react.py`          | Multi-turn ReAct loop with `<bash>` blocks (terminal-bench tasks). |
| `ablation_no_retry` | `eval/conditions/ablations/no_retry.py`           | `OpenClawAgentCondition` with retry_limit=1. |

## Pick the workload axis

Conditions live under `eval/conditions/<workload>/` where workload is one of:

- `code_agent` — code- and shell-automation tasks (SWE-bench, Terminal-Bench).
  Terminal-Bench is folded in here because TB tasks are code/automation work
  even though they're shell-driven rather than patch-driven.
- `doc_analysis` — placeholder
- `websearch` — placeholder
- `ablations` — variants of an existing condition

If your condition targets a workload that doesn't exist yet, create the folder
(`eval/conditions/<workload>/__init__.py`) — it's just a Python package.

Shared utilities sit at `eval/conditions/`:
- `_base.py` — `ConditionProtocol` + `ConditionResult`
- `_llm_client.py` — OpenAI-compat caller
- `_validate_bridge.py` — workspace IPC for self-validation

## Implement the protocol

```python
# eval/conditions/<workload>/<name>.py
from pathlib import Path
from typing import Dict, Optional

from .._base import ConditionResult        # shared utilities live one level up


class YourCondition:
    bench_type: str = "your_workload"     # purely informational; not dispatch

    def __init__(self, model_config, agent_config=None):
        self.model_config = model_config
        self.agent_config = agent_config

    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        # The driver passes these as kwargs for non-AR conditions only:
        workspace_root: Optional[str] = None,
        container_id: Optional[str] = None,
        env_info: Optional[Dict] = None,
    ) -> Dict:
        """Run the condition. Return the dict the driver expects:
            step_trace_path        : str  — JSONL of agent steps (eval/trace/logger.py)
            patch_path             : str  — file path; written empty if no patch
            total_prompt_tokens    : int
            total_completion_tokens: int
            telemetry_path         : str | None — _UtilSampler output (optional)
        """
        # 1. Read the task description from instance_data
        # 2. Run the agent loop / single LLM call / whatever your condition does
        # 3. Apply effects to workspace_root (the host-bound CWD inside container)
        # 4. Return the dict above
        ...
```

The signature flexibility — AR conditions take 3 positional args, agent
conditions take 6 — is enforced by the driver's dispatch chain rather than the
protocol. AR conditions don't need a workspace because they don't modify code.

## Patterns to lift from existing conditions

- **LLM call.** Use `eval.conditions._llm_client.call_llm(...)`. Streaming
  TTFT/TPOT capture is built in via `eval/trace/llm_wrapper.py`.
- **Step trace.** Each LLM call or tool exec should be one row in
  `step_trace.jsonl` so the analysis layer (`eval/analysis/`) can compute
  composition. Use `eval.trace.logger.StepLogger`.
- **Self-validation.** If your condition runs against a benchmark whose
  harness exposes `run_tests()` against the live workspace, instantiate
  `_validate_bridge.ValidateBridge` around your loop:

  ```python
  from .._validate_bridge import ValidateBridge

  bridge = ValidateBridge(
      harness=env_info["_harness"],     # driver puts harness ref here
      run_dir=run_dir,
      workspace_root=workspace_root,
      instance_id=instance_data["instance_id"],
      instance_data=instance_data,
      container_id=container_id,
      run_id=run_dir.name,
      timeout_seconds=...,
  )
  bridge.start()
  try:
      # ... agent loop ...
  finally:
      n = bridge.stop()
  ```

  The bridge drops `_oc_validate.sh` into the workspace; the agent invokes it
  via the existing `exec` tool and gets a JSON verdict back. Lets the loop
  exit early on PASS instead of running out the clock.
- **Telemetry.** Wrap the loop in `eval.conditions.code_agent.openclaw_agent._UtilSampler`
  for CPU/RSS/GPU sampling at run cadence. (TB uses the same.)

## Wire the dispatcher

Two edits to `eval/runner/driver.py`:

```python
# 1. Import (top of file)
from eval.conditions.your_workload.your_condition import YourCondition

# 2. Dispatch (in execute_run)
elif condition_type == "your_condition":
    condition = YourCondition(
        model_config=run_config.condition.model,
        agent_config=run_config.condition.agent,
    )
    result = condition.run(
        instance_data, run_dir, run_config.seed,
        workspace_root=env_info["workspace_root"],
        container_id=container_id,
        env_info=env_info,
    )
```

The `condition_type` string is the public dispatch key — it's referenced by
sweep YAMLs and shouldn't change once published.

## Test it

```bash
# Smoke import
PYTHONPATH=. python3 -c "
from eval.conditions.your_workload.your_condition import YourCondition
print('OK')
"

# Run a 1-instance sweep
cat > /tmp/your_condition_smoke.yaml <<EOF
sweep_id: your_condition_smoke
instances: [<some_instance_id>]
conditions:
  - { condition_id: your, condition_type: your_condition, model: {...}, agent: {...} }
serving:  { ... }
seeds: [0]
output_dir: data/runs/your_condition_smoke
EOF

PYTHONPATH=. python3 -m eval.runner.driver --sweep /tmp/your_condition_smoke.yaml
```

Inspect `data/runs/your_condition_smoke/<run_id>/`:
- `step_trace.jsonl` — agent steps with timings
- `verdict.json` — harness scoring result
- `manifest.json` — run state
- `spans.jsonl` (if you used the validate bridge / OC stack)
- `util.jsonl` (if you used `_UtilSampler`)

## Checklist

- [ ] File at `eval/conditions/<workload>/<name>.py` (workload package exists)
- [ ] `__init__` accepts `(model_config, agent_config)` for parity with peers
- [ ] `run()` returns dict with `step_trace_path`, `patch_path`,
      `total_prompt_tokens`, `total_completion_tokens`
- [ ] Patch artifact written even on failure (empty file is fine — the driver's
      patch-gate reads size, not content)
- [ ] Imports use relative-up syntax for shared utilities
      (`from .._base`, `from .._llm_client`, `from .._validate_bridge`)
- [ ] Dispatch added in `eval/runner/driver.py` with a unique `condition_type`
- [ ] Sweep YAML using the new condition runs end-to-end
