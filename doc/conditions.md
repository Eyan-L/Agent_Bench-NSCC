# Adding a new condition

A *condition* is the procedure the agent follows to attempt a task. The current
conditions:

| Condition | Module | What it does |
|---|---|---|
| `ar_issue_only`     | `orchestration/conditions/code_agent/ar_issue_only.py`     | One-shot LLM call from the issue text. No tools. |
| `ar_static_context` | `orchestration/conditions/code_agent/ar_static_context.py` | One-shot + offline BM25 retrieval. |
| `openclaw_agent`    | `orchestration/conditions/code_agent/openclaw_agent.py`    | Multi-turn agent loop driven by OpenClaw. |
| `tb_react`          | `orchestration/conditions/code_agent/terminal_bench/react.py`          | Multi-turn ReAct loop with `<bash>` blocks (terminal-bench tasks). |
| `ablation_no_retry` | `orchestration/conditions/ablations/no_retry.py`           | `OpenClawAgentCondition` with retry_limit=1. |

## Pick the workload axis

Conditions live under `orchestration/conditions/<workload>/` where workload is one of:

- `code_agent` — code- and shell-automation tasks (SWE-bench, Terminal-Bench).
  Terminal-Bench is folded in here because TB tasks are code/automation work
  even though they're shell-driven rather than patch-driven.
- `doc_analysis` — placeholder
- `websearch` — placeholder
- `ablations` — variants of an existing condition

If your condition targets a workload that doesn't exist yet, create the folder
(`orchestration/conditions/<workload>/__init__.py`) — it's just a Python package.

Shared utilities sit at `orchestration/conditions/`:
- `_base.py` — `ConditionProtocol` + `ConditionResult`
- `_llm_client.py` — OpenAI-compat caller
- `_validate_bridge.py` — workspace IPC for self-validation

## Implement the protocol

```python
# orchestration/conditions/<workload>/<name>.py
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
            step_trace_path        : str  — JSONL of agent steps (orchestration/trace/logger.py)
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

- **LLM call.** Use `orchestration.conditions._llm_client.call_llm(...)`. Streaming
  TTFT/TPOT capture is built in via `orchestration/trace/llm_wrapper.py`.
- **Step trace.** Each LLM call or tool exec should be one row in
  `step_trace.jsonl` so the analysis layer (`orchestration/analysis/`) can compute
  composition. Use `orchestration.trace.logger.StepLogger`.
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
- **Telemetry.** Wrap the loop in `orchestration.conditions.code_agent.openclaw_agent._UtilSampler`
  for CPU/RSS/GPU sampling at run cadence. (TB uses the same.)

## Wire the dispatcher

Two edits to `orchestration/runner/driver.py`:

```python
# 1. Import (top of file)
from orchestration.conditions.your_workload.your_condition import YourCondition

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
from orchestration.conditions.your_workload.your_condition import YourCondition
print('OK')
"

# Run a 1-instance sweep. Drop an application JSON, then point the driver at
# it plus an existing harness/backend JSON.
cat > config/application/swebench/your_smoke.json <<EOF
{
  "sweep_name": "your_condition_smoke",
  "plan_version": "v1.0",
  "application": "swebench",
  "instances": ["<some_instance_id>"],
  "repo_mapping": {"<some_instance_id>": {"repo_id": "...", "base_commit": "..."}},
  "container_image": "swebench/sweb.eval.x86_64.<...>:latest",
  "wall_clock_budget_seconds": 1800,
  "output_dir": "data/runs/your_condition_smoke",
  "seeds": [0],
  "parallel_limit_gpu": 1,
  "parallel_limit_cpu_cores": 8
}
EOF

PYTHONPATH=. python3 -m orchestration.runner.driver \
    --app config/application/swebench/your_smoke.json \
    --harness config/harness/openclaw_27b.json \
    --backend config/backend/vllm/tp1.json \
    --swebench-root /path/to/SWE-bench \
    --instance-data /path/to/verified.json
```

Inspect `data/runs/your_condition_smoke/<run_id>/`:
- `step_trace.jsonl` — agent steps with timings
- `verdict.json` — harness scoring result
- `manifest.json` — run state
- `spans.jsonl` (if you used the validate bridge / OC stack)
- `util.jsonl` (if you used `_UtilSampler`)

## Checklist

- [ ] File at `orchestration/conditions/<workload>/<name>.py` (workload package exists)
- [ ] `__init__` accepts `(model_config, agent_config)` for parity with peers
- [ ] `run()` returns dict with `step_trace_path`, `patch_path`,
      `total_prompt_tokens`, `total_completion_tokens`
- [ ] Patch artifact written even on failure (empty file is fine — the driver's
      patch-gate reads size, not content)
- [ ] Imports use relative-up syntax for shared utilities
      (`from .._base`, `from .._llm_client`, `from .._validate_bridge`)
- [ ] Dispatch added in `orchestration/runner/driver.py` with a unique `condition_type`
- [ ] Sweep YAML using the new condition runs end-to-end
