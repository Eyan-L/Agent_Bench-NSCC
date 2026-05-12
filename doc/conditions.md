# Adding a new condition

A *condition* is the procedure the agent follows to attempt a task. The current
conditions:

| Condition | Module | What it does |
|---|---|---|
| `ar_issue_only`     | `applications/swebench/conditions/ar_issue_only.py`     | One-shot LLM call from the issue text. No tools. SWE-bench specific. |
| `ar_static_context` | `applications/swebench/conditions/ar_static_context.py` | One-shot + offline BM25 retrieval. SWE-bench specific. |
| `openclaw_agent`    | base: `applications/conditions/openclaw_agent.py`<br>SWE: `applications/swebench/conditions/openclaw_agent.py`<br>TB:  `applications/terminal_bench/conditions/openclaw_agent.py` | Multi-turn agent loop driven by OpenClaw. Same `condition_type` for both benchmarks; the driver picks the subclass by `instance_data["benchmark"]`. |
| `tb_react`          | `applications/terminal_bench/conditions/react.py`             | Multi-turn ReAct loop with `<bash>` blocks (terminal-bench tasks). No OpenClaw involvement. |

## Layout

- `applications/conditions/` — base classes + utilities shared by all apps.
  Today:
  - `openclaw_agent.py` — `OpenClawAgentCondition` base (per-app subclasses
    plug in `agent_id`, `_build_message`, optional `_collect_artifacts`)
  - `_llm_client.py` — OpenAI-compat HTTP caller + unified-diff extractor
  - `util_sampler.py` — `UtilSampler` context manager (CPU/RSS/GPU sampling)
- `applications/<app>/conditions/` — conditions specific to that benchmark, or
  per-app subclasses of a shared base.

`orchestration/` no longer carries any condition code — only driver / dispatcher
/ runner / analysis / trace.

## Adding a benchmark-specific condition (`applications/<app>/conditions/<name>.py`)

```python
from pathlib import Path
from typing import Dict, Optional

from applications.conditions._llm_client import call_llm


class YourCondition:
    bench_type: str = "your_workload"     # informational; not used for dispatch

    def __init__(self, model_config, agent_config=None):
        self.model_config = model_config
        self.agent_config = agent_config

    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        workspace_root: Optional[str] = None,
        container_id: Optional[str] = None,
        env_info: Optional[Dict] = None,
    ) -> Dict:
        """Return the dict the driver expects:
            step_trace_path        : str
            patch_path             : str
            total_prompt_tokens    : int
            total_completion_tokens: int
            telemetry_path         : str | None
        """
        ...
```

AR conditions take 3 positional args; agent conditions take 6 — the driver's
dispatch chain enforces the signature, not the protocol.

## Adding a subclass of `OpenClawAgentCondition`

Use this when your benchmark needs the OpenClaw CLI loop (with bridge, gateway,
session archival, OTel span synthesis, etc.) but its own framing or artifact
workflow. Override three things:

```python
from typing import Dict, Optional
from pathlib import Path

from applications.conditions.openclaw_agent import OpenClawAgentCondition


class YourBenchOpenClawCondition(OpenClawAgentCondition):
    agent_id = "your_bench"           # must match openclaw.json agents.list[].id
                                      # and the harness's predicted sandbox name

    def _build_message(self, instance_data: Dict) -> str:
        return f"<framing>\n\nTask:\n{instance_data['instruction']}"

    def _collect_artifacts(
        self,
        container_id: Optional[str],
        env_info: Dict,
        run_dir: Path,
        workspace_root: str,
    ) -> Optional[str]:
        # Return patch text (it'll be written to candidate.patch), or None for
        # no artifact (in which case driver.skip_patch_gate must be true for
        # this benchmark).
        return None
```

See `applications/swebench/conditions/openclaw_agent.py` for an example that
collects a git diff.

## Patterns to lift

- **LLM call.** `applications.conditions._llm_client.call_llm(...)` — handles
  the OpenAI-compat request shape and returns `(text, prompt_tokens, completion_tokens, error)`.
- **Step trace.** One row per LLM call or tool exec in `steps.jsonl` so
  `orchestration/analysis/` can compute composition.
- **In-loop self-validation.** Surface a `verify` OpenClaw skill at
  `applications/<app>/openclaw/skills/verify/SKILL.md` that tells the agent
  how to run the in-container test command (e.g. `python -m pytest` for
  SWE-bench, `bash /app/_oc_validate.sh` for TB). The skill runs in the same
  sandbox container as the agent's `exec` tool — no host IPC needed.
- **Telemetry.** Wrap the loop with `UtilSampler` from
  `applications.conditions.util_sampler` for CPU/RSS/GPU sampling. Both
  `tb_react` and `OpenClawAgentCondition` already do this.

## Wire the dispatcher

`orchestration/runner/driver.py:execute_run` — add an `elif condition_type ==
"your_condition":` branch with a lazy import:

```python
elif condition_type == "your_condition":
    from applications.<app>.conditions.<your_condition> import YourCondition
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

Keep the import lazy so the driver doesn't pull every app on startup.

If your new condition is an `OpenClawAgentCondition` subclass, you don't need a
new `condition_type` — extend the existing `openclaw_agent` branch's
benchmark-keyed dispatch.

## Test it

```bash
# Smoke import
PYTHONPATH=. python3 -c "from applications.<app>.conditions.<name> import YourCondition; print('OK')"

# Single-instance smoke run (using existing harness + app configs)
make eval APP=<app>/<exp> HARNESS=<harness_with_your_condition> ...
```

Inspect `data/runs/<sweep>/<ts>/<run_id>/`:
- `steps.jsonl` — agent steps with timings
- `verdict.json` — harness scoring result
- `manifest.json` — run state
- `spans.jsonl` — OTel snapshot (OC conditions only)
- `util.jsonl` — UtilSampler output

## Checklist

- [ ] File at `applications/<app>/conditions/<name>.py` (or `applications/conditions/<name>.py` if cross-app)
- [ ] `__init__` accepts `(model_config, agent_config)` for parity with peers
- [ ] `run()` returns dict with `step_trace_path`, `patch_path`, `total_prompt_tokens`, `total_completion_tokens`
- [ ] Patch artifact written even on failure (empty file is fine — driver gates on size, not content; benchmarks without patches use `skip_patch_gate`)
- [ ] Shared utilities imported via absolute paths (`from applications.conditions._llm_client import ...`)
- [ ] Dispatch branch added in `orchestration/runner/driver.py` with a lazy import
- [ ] Sweep harness JSON using the new condition runs end-to-end
