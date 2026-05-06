"""Main experiment driver."""

import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import List

from eval.runner.config import load_sweep_config, RunConfig
from eval.runner.manifest import RunManifest, get_manifest_path, load_or_create_manifest
from eval.runner.harness import (
    SWEBenchHarness, HostHarness, SingularityHarness, validate_fairness_constraints,
)
from eval.runner.conditions.code_agent.ar_issue_only import ARIssueOnlyCondition
from eval.runner.conditions.code_agent.ar_static_context import ARStaticContextCondition
from eval.runner.conditions.code_agent.openclaw_agent import OpenClawAgentCondition
from eval.runner.conditions.ablations.no_retry import NoRetryAblation

# Terminal-Bench integration is optional: the import is lazy in run_sweep so a
# pure SWE-bench environment never has to install the TB-side deps (tomllib is
# stdlib on 3.11+, but the eval/benchmarks/ tree may be absent in older
# checkouts).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def execute_run(run_config: RunConfig, harness: SWEBenchHarness, 
                instance_data: dict, retry_errors: bool = True) -> RunManifest:
    """Execute a single run."""
    
    run_id = run_config.compute_run_id()
    run_dir = run_config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create manifest
    manifest = load_or_create_manifest(
        run_id=run_id,
        condition_id=run_config.condition.condition_id,
        instance_id=run_config.instance_id,
        seed=run_config.seed,
        serving_config_id=run_config.serving.config_id,
        output_dir=run_config.output_dir,
    )
    
    # Skip if already terminal (unless ERROR and retry enabled)
    if manifest.is_terminal():
        if not (retry_errors and manifest.should_retry()):
            logger.info(f"Skipping {run_id}: already {manifest.status}")
            return manifest
    
    # Update status to RUNNING
    manifest.status = "RUNNING"
    manifest.wall_start_ns = time.time_ns()
    manifest.save(get_manifest_path(run_id, run_config.output_dir))
    
    container_id = None
    env_info = None
    try:
        # Prepare environment. Passing instance_data lets benchmark-specific
        # harnesses (e.g. TerminalBenchHarness) read task.toml-derived fields
        # without re-loading the dataset; SWE-bench harnesses ignore the kwarg.
        env_info = harness.prepare_environment(
            instance_id=run_config.instance_id,
            repo_id=run_config.repo_id,
            base_commit=run_config.base_commit,
            container_image=run_config.container_image,
            workspace_root=run_config.workspace_root,
            run_id=run_id,
            instance_data=instance_data,
        )
        
        container_id = env_info["container_id"]
        
        # Execute condition
        condition_type = run_config.condition.condition_type
        
        if condition_type == "ar_issue_only":
            condition = ARIssueOnlyCondition(
                model_config=run_config.condition.model,
                max_turns=1,
            )
            result = condition.run(instance_data, run_dir, run_config.seed)
        
        elif condition_type == "ar_static_context":
            condition = ARStaticContextCondition(
                model_config=run_config.condition.model,
                max_turns=1,
            )
            result = condition.run(
                instance_data, run_dir, run_config.seed,
                workspace_root=env_info["workspace_root"],
                container_id=container_id,
                env_info=env_info
            )
        
        elif condition_type == "openclaw_agent":
            condition = OpenClawAgentCondition(
                model_config=run_config.condition.model,
                agent_config=run_config.condition.agent,
            )
            result = condition.run(
                instance_data, run_dir, run_config.seed,
                workspace_root=env_info["workspace_root"],
                container_id=container_id,
                env_info=env_info
            )
        
        elif condition_type == "ablation_no_retry":
            condition = NoRetryAblation(
                model_config=run_config.condition.model,
                agent_config=run_config.condition.agent,
            )
            result = condition.run(
                instance_data, run_dir, run_config.seed,
                workspace_root=env_info["workspace_root"],
                container_id=container_id,
                env_info=env_info
            )

        elif condition_type == "tb_react":
            # Terminal-Bench ReAct loop. Lives under eval/benchmarks/ so the
            # SWE-bench code path stays unchanged.
            from eval.benchmarks.terminal_bench.condition import (
                TerminalBenchReactCondition,
            )
            condition = TerminalBenchReactCondition(
                model_config=run_config.condition.model,
                agent_config=run_config.condition.agent,
            )
            result = condition.run(
                instance_data, run_dir, run_config.seed,
                workspace_root=env_info["workspace_root"],
                container_id=container_id,
                env_info=env_info
            )

        else:
            raise ValueError(f"Unknown condition type: {condition_type}")
        
        # Score via SWE-bench run_evaluation directly. Do NOT try to apply the
        # patch ourselves first — SWE-bench's harness handles both "patch does
        # not apply" and "tests fail" cleanly, and the verdicts it produces
        # carry richer signal (resolved, patch_successfully_applied,
        # tests_status.FAIL_TO_PASS/PASS_TO_PASS) than our binary verdict.
        # Terminal-Bench has no patch artifact — agent's effects live in the
        # task container — so we skip the patch-presence gate for it.
        patch_path = Path(result["patch_path"])
        benchmark_kind = instance_data.get("benchmark", "swebench")
        skip_patch_gate = benchmark_kind == "terminal_bench"

        if skip_patch_gate or (patch_path.exists() and patch_path.stat().st_size > 0):
            verdict = harness.run_tests(
                container_id=container_id,
                instance_id=run_config.instance_id,
                workspace_root=env_info["workspace_root"],
                timeout_seconds=run_config.wall_clock_budget_seconds,
                patch=patch_path.read_text() if patch_path.exists() else "",
                run_id=run_id,
                instance_data=instance_data,
            )
        else:
            verdict = {
                "status": "FAIL",
                "error_message": "No patch generated",
            }
        
        # Write verdict
        verdict_path = run_dir / "verdict.json"
        import json
        with open(verdict_path, "w") as f:
            json.dump(verdict, f, indent=2)
        
        # Update manifest
        manifest.status = verdict["status"]
        manifest.step_trace_path = result.get("step_trace_path")
        manifest.telemetry_path = result.get("telemetry_path")
        manifest.verdict_path = str(verdict_path)
        manifest.exit_code = result.get("exit_code", 0)
        manifest.total_prompt_tokens = result["total_prompt_tokens"]
        manifest.total_completion_tokens = result["total_completion_tokens"]
        
    except Exception as e:
        logger.error(f"Run {run_id} failed with error: {e}", exc_info=True)
        manifest.status = "ERROR"
        manifest.error_message = str(e)[:500]
    
    finally:
        manifest.wall_end_ns = time.time_ns()
        manifest.duration_seconds = (
            (manifest.wall_end_ns - manifest.wall_start_ns) / 1e9
        )
        manifest.save(get_manifest_path(run_id, run_config.output_dir))
        
        # Cleanup
        try:
            harness.cleanup(
                container_id,
                workspace_root=env_info.get("workspace_root") if env_info else None,
                gateway_port=env_info.get("gateway_port") if env_info else None,
                gateway_pid=env_info.get("gateway_pid") if env_info else None,
                node_pid=env_info.get("node_pid") if env_info else None,
                host_state_dir=env_info.get("host_state_dir") if env_info else None,
                container_name=env_info.get("container_name") if env_info else None,
            )
        except Exception:
            pass
    
    return manifest


def run_sweep(sweep_config_path: Path, swebench_root: Path,
              instance_data_path: Path, retry_errors: bool = True,
              max_workers: int = 1):
    """Run full sweep.

    `max_workers > 1` runs containers concurrently. Each container has its
    own Gateway on its own port (range 18800-18899 stride 4 → ~25 parallel),
    its own /testbed workspace, and tags OTel spans with oc.run_id so the
    shared collector output remains demuxable. Real bottleneck at scale is
    vLLM throughput (single endpoint).
    """

    # Load sweep config
    sweep_config = load_sweep_config(sweep_config_path)

    # Generate all run configs
    run_configs = sweep_config.generate_runs()

    logger.info(
        f"Loaded sweep '{sweep_config.sweep_name}' with {len(run_configs)} runs "
        f"(max_workers={max_workers})"
    )

    # Validate fairness constraints
    validate_fairness_constraints(run_configs)

    # Load instance data
    import json
    with open(instance_data_path) as f:
        instances = {inst["instance_id"]: inst for inst in json.load(f)}

    # Initialize harness. HARNESS_BACKEND selects the SWE-bench backend:
    #   docker      (default) — SWEBenchHarness; per-task SWE-bench image
    #   host                  — HostHarness;     run agent on the host
    #   singularity           — SingularityHarness; openclaw inside an OC sif
    # Terminal-Bench instances are auto-routed to TerminalBenchHarness
    # regardless of HARNESS_BACKEND (per-instance dispatch in _one below).
    needs_swe = any(
        inst.get("benchmark", "swebench") != "terminal_bench"
        for inst in instances.values()
    )
    needs_tb = any(
        inst.get("benchmark") == "terminal_bench" for inst in instances.values()
    )

    harness = None
    if needs_swe:
        backend = os.environ.get("HARNESS_BACKEND", "docker").lower()
        if backend == "host":
            logger.info("Using HostHarness (no docker; scoring deferred).")
            harness = HostHarness(swebench_root)
        elif backend == "singularity":
            logger.info("Using SingularityHarness (openclaw inside OC runtime sif).")
            harness = SingularityHarness(swebench_root)
        elif backend == "docker":
            harness = SWEBenchHarness(swebench_root)
        else:
            raise ValueError(
                f"Unknown HARNESS_BACKEND={backend!r}; expected docker|host|singularity"
            )
    else:
        logger.info("Sweep is pure terminal-bench; skipping SWE-bench harness build.")

    # Lazy-build the TB harness only if any instance in this sweep is TB.
    tb_harness = None
    if needs_tb:
        from eval.benchmarks.terminal_bench.harness import TerminalBenchHarness
        if harness is not None:
            logger.info(
                "Sweep contains terminal-bench instances; constructing "
                "TerminalBenchHarness alongside %s.",
                type(harness).__name__,
            )
        else:
            logger.info("Constructing TerminalBenchHarness.")
        tb_harness = TerminalBenchHarness(swebench_root)

    completed = 0
    failed = 0
    total = len(run_configs)

    def _one(rc):
        try:
            inst = instances[rc.instance_id]
            h = tb_harness if inst.get("benchmark") == "terminal_bench" else harness
            if h is None:
                raise RuntimeError(
                    f"instance {rc.instance_id} requires terminal_bench harness "
                    "but it was not constructed (loader bug?)"
                )
            return execute_run(rc, h, inst, retry_errors)
        except Exception as e:
            logger.error("run %s crashed: %s", rc.instance_id, e, exc_info=True)
            return None

    if max_workers <= 1:
        results_iter = (_one(rc) for rc in run_configs)
    else:
        pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="oc-eval-run")
        futures = [pool.submit(_one, rc) for rc in run_configs]
        results_iter = (f.result() for f in as_completed(futures))

    for manifest in results_iter:
        if manifest is None:
            failed += 1
        elif manifest.status in ("PASS", "FAIL", "TIMEOUT", "DEFERRED"):
            completed += 1
        elif manifest.status == "ERROR":
            failed += 1
        logger.info(
            f"Progress: {completed + failed}/{total} "
            f"({completed} completed, {failed} errors)"
        )

    if max_workers > 1:
        pool.shutdown(wait=True)

    logger.info(f"Sweep complete: {completed} runs completed, {failed} errors")


def main():
    parser = argparse.ArgumentParser(description="OpenClaw + SWE-bench experiment driver")
    parser.add_argument("--sweep", type=Path, required=True,
                       help="Path to sweep YAML config")
    parser.add_argument("--swebench-root", type=Path, required=True,
                       help="Path to SWE-bench repository")
    parser.add_argument("--instance-data", type=Path, required=True,
                       help="Path to SWE-bench instance data JSON")
    parser.add_argument("--no-retry-errors", action="store_true",
                       help="Do not retry ERROR runs")
    parser.add_argument("--max-workers", type=int,
                       default=int(os.environ.get("EVAL_MAX_WORKERS", "1")),
                       help="Concurrent containers (default 1; env EVAL_MAX_WORKERS)")

    args = parser.parse_args()

    run_sweep(
        sweep_config_path=args.sweep,
        swebench_root=args.swebench_root,
        instance_data_path=args.instance_data,
        retry_errors=not args.no_retry_errors,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
