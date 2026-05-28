"""Main experiment driver."""

import argparse
import logging
import multiprocessing as mp
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import List, Optional

from agent_bench.core.config import load_composed_config, RunConfig
from agent_bench.core.manifest import RunManifest, get_manifest_path, load_or_create_manifest
from agent_bench.core.profile_io import load_profile
from agent_bench.benchmarks.swebench.app import validate_fairness_constraints
from agent_bench.benchmarks.swebench.openclaw.app import SWEBenchHarness
from agent_bench.deployments.containers import DockerContainer, SingularityContainer
# Condition classes are lazy-imported in `execute_run` so the driver can boot
# without pulling every app's condition tree (and so an app that adds a new
# condition class doesn't force a driver edit). Harness classes that are only
# selected behind a config flag (e.g. SWEBenchOpenHandsHarness) are also
# imported lazily inside `run_sweep` below.

# Terminal-Bench integration is optional: the import is lazy in run_sweep so a
# pure SWE-bench environment never has to install the TB-side deps (tomllib is
# stdlib on 3.11+, but the agent_bench/benchmarks/terminal_bench/ tree may be absent in older
# checkouts).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _install_signal_handlers():
    """Convert SIGTERM/SIGHUP into KeyboardInterrupt so `finally` blocks run.

    SIGINT already raises KeyboardInterrupt by default — Python's default for
    SIGTERM is to exit without unwinding, which skips the `finally` in
    execute_run that calls `harness.cleanup`. We can't catch SIGKILL, but
    converting SIGTERM/SIGHUP into KeyboardInterrupt covers the common
    "user kill -15 / supervisor stop / parent terminal closed" cases —
    cleanup runs and both containers come down deterministically.
    """
    def _to_interrupt(signum, _frame):
        logger.warning("received signal %s — raising KeyboardInterrupt to trigger cleanup", signum)
        raise KeyboardInterrupt()
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _to_interrupt)
        except (ValueError, OSError):
            # SIGHUP is unavailable on Windows; secondary threads can't install.
            pass


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
        # AB_*_CONFIG paths (set in main()) are the authoritative config sources
        # used by harness adapters when rendering per-run runtime config.
        env_info = harness.prepare_environment(
            instance_id=run_config.instance_id,
            repo_id=run_config.repo_id,
            base_commit=run_config.base_commit,
            container_image=run_config.container_image,
            workspace_root=run_config.workspace_root,
            run_id=run_id,
            instance_data=instance_data,
        )

        # Hand the harness ref to the condition for any host-side helper that
        # needs to reach harness.run_tests / exec_in_container while the agent
        # loop is in flight. Underscore-prefixed key signals
        # "internal/transient — don't persist into manifests".
        env_info["_harness"] = harness

        container_id = env_info["container_id"]
        
        # Execute condition
        condition_type = run_config.condition.condition_type
        
        if condition_type == "ar_issue_only":
            # AR baselines are SWE-bench-specific (their prompts reference
            # "fixing a bug in a Python repository"); live under the app.
            from agent_bench.benchmarks.swebench.conditions.ar_issue_only import (
                ARIssueOnlyCondition,
            )
            condition = ARIssueOnlyCondition(
                model_config=run_config.condition.model,
                max_turns=1,
            )
            result = condition.run(instance_data, run_dir, run_config.seed)

        elif condition_type == "ar_static_context":
            from agent_bench.benchmarks.swebench.conditions.ar_static_context import (
                ARStaticContextCondition,
            )
            condition = ARStaticContextCondition(
                model_config=run_config.condition.model,
                max_turns=1,
                container=getattr(harness, "container", None),
            )
            result = condition.run(
                instance_data, run_dir, run_config.seed,
                workspace_root=env_info["workspace_root"],
                container_id=container_id,
                env_info=env_info
            )

        elif condition_type == "openclaw_agent":
            # Pick the per-app subclass by benchmark — same `condition_type`
            # in SWE-bench / TB / DR harness JSONs; instance_data is the
            # discriminator.
            benchmark = instance_data.get("benchmark", "swebench")
            if benchmark == "terminal_bench":
                from agent_bench.benchmarks.terminal_bench.conditions.openclaw_agent import (
                    TerminalBenchOpenClawCondition as _Cls,
                )
            elif benchmark == "deepresearch_bench":
                from agent_bench.benchmarks.deepresearch_bench.conditions.openclaw_agent import (
                    DeepResearchOpenClawCondition as _Cls,
                )
            elif benchmark == "longbench":
                from agent_bench.benchmarks.longbench.conditions.openclaw_agent import (
                    LongBenchOpenClawCondition as _Cls,
                )
            else:
                from agent_bench.benchmarks.swebench.conditions.openclaw_agent import (
                    SWEBenchOpenClawCondition as _Cls,
                )
            # `harness` here is whichever per-benchmark harness `_one` picked
            # (SWE-bench openclaw / openhands / TB / DR). Only `SWEBenchHarness`
            # currently exposes `.container`; the others fall through to None
            # and the condition keeps its legacy `docker exec` path.
            condition = _Cls(
                model_config=run_config.condition.model,
                agent_config=run_config.condition.agent,
                container=getattr(harness, "container", None),
            )
            result = condition.run(
                instance_data, run_dir, run_config.seed,
                workspace_root=env_info["workspace_root"],
                container_id=container_id,
                env_info=env_info
            )

        elif condition_type == "openhands_agent":
            # OpenHands is SWE-bench-only in this branch (no TB port yet); the
            # condition file lives next to the openclaw one under the app dir.
            from agent_bench.benchmarks.swebench.conditions.openhands_agent import (
                OpenHandsAgentCondition,
            )
            condition = OpenHandsAgentCondition(
                model_config=run_config.condition.model,
                agent_config=run_config.condition.agent,
                container=getattr(harness, "container", None),
            )
            result = condition.run(
                instance_data, run_dir, run_config.seed,
                workspace_root=env_info["workspace_root"],
                container_id=container_id,
                env_info=env_info
            )

        elif condition_type == "hermes_agent":
            from agent_bench.benchmarks.swebench.conditions.hermes_agent import (
                HermesAgentCondition,
            )
            condition = HermesAgentCondition(
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
            # Terminal-Bench ReAct loop — app-specific, no OpenClaw.
            from agent_bench.benchmarks.terminal_bench.conditions.react import (
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
        # TB and DR-Bench have no git-patch artifact — the agent's work lives
        # inside the task container (TB: services + files; DR: /workspace/report.md).
        skip_patch_gate = benchmark_kind in ("terminal_bench", "deepresearch_bench", "longbench")

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

        # Persist any agent-produced report (e.g. DR-Bench /workspace/report.md)
        # alongside the verdict, so it survives harness cleanup. report_path in
        # the verdict points at the harness's scratch dir which gets wiped.
        report_src = verdict.get("report_path") if isinstance(verdict, dict) else None
        if report_src:
            try:
                src_p = Path(report_src)
                if src_p.is_file() and src_p.stat().st_size > 0:
                    import shutil
                    shutil.copyfile(src_p, run_dir / "report.md")
            except OSError as e:
                logger.warning("persist report.md failed for %s: %s", run_id, e)
        
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
        
        # Cleanup. Forward env_info wholesale so harnesses can grab whichever
        # keys they need (oc_container_id / task_container_id / network_name
        # for two-container docker mode; gateway_pid / state_dir for legacy
        # singularity; etc.). Strip keys that conflict with the positional
        # argument or carry harness refs.
        try:
            cleanup_kwargs = dict(env_info or {})
            for k in ("container_id", "_harness"):
                cleanup_kwargs.pop(k, None)
            harness.cleanup(container_id, **cleanup_kwargs)
        except Exception:
            logger.exception("cleanup failed")
    
    return manifest


def run_sweep(app_path: Path, harness_path: Path, model_path: Path, serving_path: Path,
              swebench_root: Optional[Path], instance_data_path: Path,
              retry_errors: bool = True, max_workers: int = 1):
    """Run full sweep, composed from app / harness / backend JSON configs.

    `max_workers > 1` runs containers concurrently. Each container has its
    own Gateway on its own port (range 18800-18899 stride 4 → ~25 parallel),
    its own /testbed workspace, and tags OTel spans with oc.run_id so the
    shared collector output remains demuxable. Real bottleneck at scale is
    vLLM throughput (single endpoint).
    """

    # Load sweep config by composing the four axes.
    sweep_config = load_composed_config(app_path, harness_path, model_path, serving_path)

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
    #   docker      (default) — SWEBenchHarness;     openclaw container + task container
    #                            on a shared docker network. Host runs only vLLM.
    #   singularity            — SingularityHarness; openclaw .sif + task .sif (peer
    #                            instances). Same isolation model in singularity-land.
    # Terminal-Bench / DR-Bench instances are auto-routed to their respective
    # harnesses regardless of HARNESS_BACKEND (per-instance dispatch in _one below).
    _NON_SWE_BENCHMARKS = ("terminal_bench", "deepresearch_bench", "longbench")
    needs_swe = any(
        inst.get("benchmark", "swebench") not in _NON_SWE_BENCHMARKS
        for inst in instances.values()
    )
    needs_tb = any(
        inst.get("benchmark") == "terminal_bench" for inst in instances.values()
    )
    needs_dr = any(
        inst.get("benchmark") == "deepresearch_bench" for inst in instances.values()
    )
    needs_lb = any(
        inst.get("benchmark") == "longbench" for inst in instances.values()
    )

    # Peek at the harness config to learn which agent runtime this sweep
    # targets — openclaw (default), openhands, or hermes. Each picks a different
    # Layer-1 class even though they share the SWE-bench application.
    harness_choice = "openclaw"
    try:
        _hc = load_profile(harness_path)
        harness_choice = (_hc.get("harness") or "openclaw").lower()
    except Exception as e:  # noqa: BLE001 — non-fatal, fall back to default
        logger.warning("could not read 'harness' from %s: %s", harness_path, e)

    harness = None
    if needs_swe:
        if swebench_root is None:
            raise RuntimeError(
                "sweep contains SWE-bench instances but --swebench-root was "
                "not provided"
            )
        backend = os.environ.get("HARNESS_BACKEND", "docker").lower()
        if backend == "docker":
            container = DockerContainer()
        elif backend == "singularity":
            container = SingularityContainer()
        else:
            raise ValueError(
                f"Unknown HARNESS_BACKEND={backend!r}; expected docker|singularity. "
                "Host mode was removed — agent runtimes must always be containerized."
            )

        if harness_choice == "openhands":
            from agent_bench.benchmarks.swebench.openhands.app import SWEBenchOpenHandsHarness
            logger.info(
                "Using SWEBenchOpenHandsHarness with %s "
                "(openhands runtime + task container + exec-bridge).",
                type(container).__name__,
            )
            harness = SWEBenchOpenHandsHarness(swebench_root, container=container)
        elif harness_choice == "hermes":
            from agent_bench.benchmarks.swebench.hermes.app import SWEBenchHermesHarness
            logger.info(
                "Using SWEBenchHermesHarness with %s "
                "(hermes runtime + task container + exec-bridge).",
                type(container).__name__,
            )
            harness = SWEBenchHermesHarness(swebench_root, container=container)
        elif harness_choice == "openclaw":
            logger.info(
                "Using SWEBenchHarness with %s "
                "(openclaw runtime + task container + exec-bridge).",
                type(container).__name__,
            )
            harness = SWEBenchHarness(swebench_root, container=container)
        else:
            raise ValueError(
                f"Unknown harness={harness_choice!r}; expected openclaw|openhands|hermes."
            )
    else:
        logger.info("Sweep is pure terminal-bench; skipping SWE-bench harness build.")

    # Lazy-build the DR harness only if any instance in this sweep is DR-Bench.
    # Same two-container model as TB but with a fixed lightweight sandbox image
    # and host-side LLM-as-judge scoring (RACE + FACT via DeepSeek-V4).
    dr_harness = None
    if needs_dr:
        from agent_bench.benchmarks.deepresearch_bench.app import DeepResearchHarness
        dr_backend = os.environ.get("HARNESS_BACKEND", "docker").lower()
        if dr_backend != "docker":
            raise ValueError(
                f"deepresearch_bench only supports HARNESS_BACKEND=docker "
                f"(got {dr_backend!r}); singularity backend not implemented."
            )
        if harness is not None:
            logger.info(
                "Sweep contains deepresearch_bench instances; constructing "
                "DeepResearchHarness alongside %s.", type(harness).__name__,
            )
        else:
            logger.info("Constructing DeepResearchHarness.")
        dr_harness = DeepResearchHarness(swebench_root)

    # Lazy-build the LongBench harness only if any instance is LongBench.
    # Same two-container model as DR; per-subtask instance loops examples over
    # warm containers, retrieving from the shared ClawRAG service.
    lb_harness = None
    if needs_lb:
        from agent_bench.benchmarks.longbench.app import LongBenchHarness
        lb_backend = os.environ.get("HARNESS_BACKEND", "docker").lower()
        if lb_backend != "docker":
            raise ValueError(
                f"longbench only supports HARNESS_BACKEND=docker (got {lb_backend!r})."
            )
        logger.info("Constructing LongBenchHarness.")
        lb_harness = LongBenchHarness(swebench_root)

    # Lazy-build the TB harness only if any instance in this sweep is TB.
    # Same HARNESS_BACKEND env var as SWE-bench:
    #   docker      (default) — TerminalBenchDockerHarness
    #                            (OC container + task container + bridge)
    #   singularity            — TerminalBenchSingularityHarness (HPC fallback)
    tb_harness = None
    if needs_tb:
        from agent_bench.benchmarks.terminal_bench.app import (
            TerminalBenchDockerHarness,
            TerminalBenchSingularityHarness,
        )
        tb_backend = os.environ.get("HARNESS_BACKEND", "docker").lower()
        if tb_backend == "docker":
            tb_cls = TerminalBenchDockerHarness
        elif tb_backend == "singularity":
            tb_cls = TerminalBenchSingularityHarness
        else:
            raise ValueError(
                f"Unknown HARNESS_BACKEND={tb_backend!r} for terminal_bench; "
                "expected docker|singularity."
            )
        if harness is not None:
            logger.info(
                "Sweep contains terminal-bench instances; constructing "
                "%s alongside %s.",
                tb_cls.__name__, type(harness).__name__,
            )
        else:
            logger.info("Constructing %s.", tb_cls.__name__)
        tb_harness = tb_cls(swebench_root)

    completed = 0
    failed = 0
    total = len(run_configs)

    def _one(rc):
        try:
            inst = instances[rc.instance_id]
            bm = inst.get("benchmark")
            if bm == "terminal_bench":
                h = tb_harness
            elif bm == "deepresearch_bench":
                h = dr_harness
            elif bm == "longbench":
                h = lb_harness
            else:
                h = harness
            if h is None:
                raise RuntimeError(
                    f"instance {rc.instance_id} (benchmark={bm!r}) requires a "
                    "harness that was not constructed (loader bug?)"
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
    parser = argparse.ArgumentParser(description="Agent_Bench experiment driver (composed config)")
    parser.add_argument("--app", type=Path, required=True,
                       help="Application config (profiles/benchmarks/<app>/<exp>.json)")
    parser.add_argument("--harness", type=Path, required=True,
                       help="Harness config (profiles/harnesses/<name>.yaml)")
    parser.add_argument("--model", type=Path, required=True,
                       help="Model config (profiles/models/<name>.json)")
    parser.add_argument("--serving", type=Path, required=True,
                       help="Serving config (profiles/serving/<engine>/<name>.yaml)")
    # swebench-root is only required when the sweep contains SWE-bench
    # instances — pure terminal_bench sweeps don't use it.
    parser.add_argument("--swebench-root", type=Path, required=False,
                       help="Path to SWE-bench repository (required for SWE-bench instances)")
    parser.add_argument("--instance-data", type=Path, required=True,
                       help="Path to instance data JSON (SWE-bench or TB)")
    parser.add_argument("--no-retry-errors", action="store_true",
                       help="Do not retry ERROR runs")
    parser.add_argument("--max-workers", type=int,
                       default=int(os.environ.get("EVAL_MAX_WORKERS", "1")),
                       help="Concurrent containers (default 1; env EVAL_MAX_WORKERS)")

    args = parser.parse_args()

    # Stash the composed-config paths in env so harness code can render
    # per-run runtime config without re-parsing CLI args.
    os.environ["AB_APP_CONFIG"] = str(args.app.resolve())
    os.environ["AB_HARNESS_CONFIG"] = str(args.harness.resolve())
    os.environ["AB_MODEL_CONFIG"] = str(args.model.resolve())
    os.environ["AB_SERVING_CONFIG"] = str(args.serving.resolve())
    os.environ["AB_INSTANCE_DATA"] = str(args.instance_data.resolve())

    run_sweep(
        app_path=args.app,
        harness_path=args.harness,
        model_path=args.model,
        serving_path=args.serving,
        swebench_root=args.swebench_root,
        instance_data_path=args.instance_data,
        retry_errors=not args.no_retry_errors,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    _install_signal_handlers()
    main()
