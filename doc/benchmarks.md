# Adding a new benchmark

A *benchmark* is a task suite the agent is scored against. The two existing
benchmarks are SWE-bench (`eval/benchmarks/swebench/`) and Terminal-Bench
(`eval/benchmarks/terminal_bench/`). Adding a new one means writing a harness,
a loader, dropping the task data into `config/`, and wiring the dispatcher.

## What you'll add

```
eval/benchmarks/<name>/
├── __init__.py
├── harness.py         # lifecycle + scoring
└── loader.py          # turn instance_id → instance_data dict

config/benchmarks/<name>/
└── ... (instance JSONs / task definitions / per-instance assets)

config/sweeps/<name>_<sweep>.yaml   # at least one sweep that targets it
```

## 1. The harness

A harness owns the **per-run lifecycle**: stand up the environment, expose it
to the condition, score the result, tear down. The driver
(`eval/runner/driver.py`) calls these methods in order:

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
- **Three backend variants are normal.** SWE-bench has `SWEBenchHarness`
  (docker), `HostHarness` (host-mode singularity scoring), and
  `SingularityHarness` (full singularity, including OC runtime). Pick what
  matches your deployment; HPC clusters typically need at least the
  singularity variant.
- **Workspace bind matters.** The agent's CWD inside the container must point
  at a host directory you control, so the condition can read the post-run
  state and the harness can score it. Pick a stable bind path (e.g. `/testbed`
  for SWE-bench, `/app` for TB).
- **Scoring is deterministic given the workspace.** A good harness passes
  `patch=None` and re-collects from the workspace via `git diff` — that lets
  the [validate bridge](../eval/conditions/_validate_bridge.py) call
  `run_tests()` mid-run on the live workspace state without diff bookkeeping.

## 2. The loader

```python
# eval/benchmarks/<name>/loader.py
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

Goes under `config/benchmarks/<name>/`. There's no required shape — it's just
where the loader reads from. Examples:

- SWE-bench: `config/benchmarks/swebench/verified.json` — single JSON of all instances.
- Terminal-Bench: `config/benchmarks/terminal_bench/<task>/{task.toml, environment/, tests/}` — one folder per task.

Pick whichever matches the upstream benchmark's distribution format.

## 4. Wire the dispatcher

Two places:

**`eval/runner/driver.py`** — the harness selection (currently in
`execute_run` near top): branch on `instance_data["benchmark"]`. Existing pattern:

```python
benchmark_kind = instance_data.get("benchmark", "swebench")
if benchmark_kind == "your_benchmark":
    from eval.benchmarks.your_benchmark.harness import YourBenchmarkHarness
    harness = YourBenchmarkHarness(...)
else:
    # existing branches
```

Lazy import keeps a pure SWE-bench install from needing your benchmark's deps.

**`eval/runner/driver.py`** — the patch gate. Some benchmarks (TB) have no
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
from eval.benchmarks.your_benchmark.harness import YourBenchmarkHarness
from eval.benchmarks.your_benchmark.loader import load_instances
print('OK')
"
PYTHONPATH=. python3 -m eval.runner.driver \
    --sweep config/sweeps/your_benchmark_phase1.yaml \
    --max-runs 1
```

Then inspect `data/runs/your_benchmark_phase1/<run_id>/` for `verdict.json`,
`step_trace.jsonl`, and (if applicable) `spans.jsonl`.

## Checklist

- [ ] `eval/benchmarks/<name>/{__init__.py, harness.py, loader.py}`
- [ ] Harness exposes `prepare_environment` / `run_tests` / `cleanup` with the
      shapes above
- [ ] `config/benchmarks/<name>/` populated
- [ ] Driver dispatch branches on `instance_data["benchmark"]`
- [ ] Patch-gate exemption added (only if your benchmark has no patch artifact)
- [ ] At least one sweep YAML in `config/sweeps/`
- [ ] Smoke test produces a verdict file
