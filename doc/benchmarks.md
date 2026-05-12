# Adding a new benchmark

A *benchmark* (or "application" in the three-layer abstraction) is a task
suite the agent is scored against. The two existing applications are SWE-bench
(`applications/swebench/`) and Terminal-Bench (`applications/terminal_bench/`).
Adding a new one means writing the app code, a loader, per-(app × harness) glue,
dropping the task data into `config/`, and wiring the dispatcher.

## What you'll add

```
applications/<name>/
├── __init__.py
├── app.py             # lifecycle + scoring (historically called harness.py)
├── loader.py          # turn instance_id → instance_data dict
└── openclaw/          # OpenClaw customization for THIS app (per-(app × harness))
    ├── agents/        # AGENTS.md etc. — mounted into /root/.openclaw/agents/<id>/
    └── skills/verify/SKILL.md  # `verify` OpenClaw skill — tells the agent how to run the in-loop check

applications/<name>/tasks/           # per-task assets vendored under the app
└── ... (instance JSONs / task definitions / per-instance assets)

config/sweeps/<name>_<sweep>.yaml   # at least one sweep that targets it
```

## 1. The harness

A harness owns the **per-run lifecycle**: stand up the environment, expose it
to the condition, score the result, tear down. The driver
(`orchestration/runner/driver.py`) calls these methods in order:

```python
class YourBenchmarkHarness:
    backend_id = "your_benchmark"          # used for log labels / instance names

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,                      # ignored by some benches; pass through
        base_commit: str,                  # ignored by some benches
        container_image: Optional[str] = None,
        workspace_root: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        """Return env_info dict. Required keys:
            container_id     : str | None — handle for `docker exec`-style calls,
                               or None for non-container backends.
            workspace_root   : str        — absolute path to the agent's CWD.
        Optional but commonly used:
            container_name, gateway_port, gateway_token, gateway_url, ...
        """

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,       # SWE-bench-style; None for non-patch benches
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        """Return verdict dict. Required keys:
            status           : "PASS" | "FAIL" | "ERROR" | "TIMEOUT" | "DEFERRED"
            test_pass_rate   : float in [0.0, 1.0]
            passed, failed   : int
            error_message    : str | None
        Conventions used by analysis:
            evaluation_dir   : str | None — where raw test logs live
            tests_status     : SWE-bench-shape map for per-test breakdown (optional)
        """

    def cleanup(self, container_id: Optional[str] = None, **kwargs) -> None:
        """Idempotent. Called from `finally` — must not raise."""
```

Implementation tips drawn from the existing harnesses:

- **Reuse primitives.** SWE-bench's `_score_via_singularity` builds the eval
  script via `swebench.harness.test_spec.make_test_spec` and parses the log
  with `get_eval_report`. If your benchmark publishes a Python harness package,
  prefer that over re-implementing the scoring math.
- **Two backend variants.** SWE-bench has `SWEBenchHarness` (docker:
  oc container + task container) and `SingularityHarness` (singularity:
  oc .sif + task .sif — currently being rewritten for the two-instance
  model). Pick what matches your deployment; both isolate the agent from
  the host.
- **Workspace bind matters.** The agent's CWD inside the container must point
  at a host directory you control, so the condition can read the post-run
  state and the harness can score it. Pick a stable bind path (e.g. `/testbed`
  for SWE-bench, `/app` for TB).
- **Scoring is deterministic given the workspace.** A good harness passes
  `patch=None` and re-collects from the workspace via `git diff` — so a future
  in-loop self-validation skill can call `run_tests()` on the live workspace
  without diff bookkeeping. (Today both apps' `verify` skills run the test
  command directly inside the sandbox container; no host IPC needed.)

## 2. The loader

```python
# applications/<name>/loader.py
def load_instances(dataset_path: str) -> Dict[str, Dict]:
    """Return {instance_id: instance_data}. instance_data must include:
        instance_id      : str
        benchmark        : str — "your_benchmark" — driver dispatches on this
        # plus whatever your harness needs in `instance_data`
    """
```

The driver calls the loader once per sweep, caches the result, then passes the
right `instance_data` to `prepare_environment(...)` for each run.

## 3. Drop the task data

Goes under the application itself — `applications/<name>/tasks/` (or wherever
the loader points). There's no required shape; it's just where the loader
reads from. Examples:

- SWE-bench: external `verified.json` (path via `--instance-data`) — single JSON of all instances. No tasks/ dir because the upstream dataset is one big JSON.
- Terminal-Bench: `applications/terminal_bench/tasks/<task>/{task.toml, environment/, tests/}` — one folder per task, vendored in-repo.

Pick whichever matches the upstream benchmark's distribution format.

## 4. Wire the dispatcher

Two places:

**`orchestration/runner/driver.py`** — the harness selection (currently in
`execute_run` near top): branch on `instance_data["benchmark"]`. Existing pattern:

```python
benchmark_kind = instance_data.get("benchmark", "swebench")
if benchmark_kind == "your_benchmark":
    from applications.your_benchmark.app import YourBenchmarkHarness
    harness = YourBenchmarkHarness(...)
else:
    # existing branches
```

Lazy import keeps a pure SWE-bench install from needing your benchmark's deps.

**`orchestration/runner/driver.py`** — the patch gate. Some benchmarks (TB) have no
patch artifact and the empty-patch FAIL gate must be skipped. Add your
benchmark to:

```python
skip_patch_gate = benchmark_kind in ("terminal_bench", "your_benchmark")
```

if applicable.

## 5. Add a sweep YAML

`config/sweeps/<name>_phase1.yaml`. Mirror an existing sweep
(`config/sweeps/tb_kv_store.yaml` is the smallest example). At minimum:

```yaml
sweep_id: your_benchmark_phase1
instances:
  - your_first_instance_id
conditions:
  - condition_id: openclaw_agent           # or any other registered condition
    condition_type: openclaw_agent
    model: { ... }
    agent: { ... }
serving:
  config_id: vllm_default
  ...
seeds: [0]
output_dir: data/runs/your_benchmark_phase1
```

## 6. Smoke test

```bash
PYTHONPATH=. python3 -c "
from applications.your_benchmark.app import YourBenchmarkHarness
from applications.your_benchmark.loader import load_instances
print('OK')
"
PYTHONPATH=. python3 -m orchestration.runner.driver \
    --sweep config/sweeps/your_benchmark_phase1.yaml \
    --max-runs 1
```

Then inspect `data/runs/your_benchmark_phase1/<run_id>/` for `verdict.json`,
`step_trace.jsonl`, and (if applicable) `spans.jsonl`.

## Checklist

- [ ] `applications/<name>/{__init__.py, app.py, loader.py}`
- [ ] `applications/<name>/openclaw/{agents,skills}` if running on OpenClaw (cross-app gateway helpers come from `applications/conditions/gateway_helpers.py`)
- [ ] Harness exposes `prepare_environment` / `run_tests` / `cleanup` with the
      shapes above
- [ ] `applications/<name>/tasks/` populated (or external dataset path documented)
- [ ] Driver dispatch branches on `instance_data["benchmark"]`
- [ ] Patch-gate exemption added (only if your benchmark has no patch artifact)
- [ ] At least one sweep YAML in `config/sweeps/`
- [ ] Smoke test produces a verdict file
