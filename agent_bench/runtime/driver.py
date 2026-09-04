"""Main experiment driver."""

import argparse
from contextlib import contextmanager
import logging
import multiprocessing as mp
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Any, List, Optional

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


def _telemetry_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled", "skip"}:
            return False
    return None

def _spans_html_enabled_from_config() -> bool:
    """Read spans HTML generation from experiment/profile config.

    Supported shapes:
      [telemetry]
      spans_html = true

      telemetry:
        spans:
          html: true
    """
    config_env_keys = ("AB_EXPERIMENT_CONFIG", "AB_HARNESS_CONFIG", "AB_APP_CONFIG")
    for env_key in config_env_keys:
        path = os.environ.get(env_key)
        if not path:
            continue
        try:
            profile = load_profile(Path(path))
        except Exception as exc:  # noqa: BLE001 - missing optional config falls back to default
            logger.debug("could not read telemetry config from %s=%s: %s", env_key, path, exc)
            continue
        telemetry = profile.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        direct = _telemetry_bool(telemetry.get("spans_html"))
        if direct is not None:
            return direct
        spans = telemetry.get("spans")
        if isinstance(spans, dict):
            nested = _telemetry_bool(spans.get("html"))
            if nested is not None:
                return nested
    return True


def _maybe_render_spans_timeline(run_dir: Path, telemetry_path: Optional[str]) -> None:
    """Render spans.jsonl to HTML when telemetry is available.

    This is intentionally best-effort: timeline generation should never turn a
    benchmark PASS/FAIL into ERROR.
    """
    if not _spans_html_enabled_from_config():
        return

    candidates = []
    if telemetry_path:
        candidates.append(Path(telemetry_path))
    candidates.append(run_dir / "spans.jsonl")

    spans_path = next((p for p in candidates if p.is_file() and p.stat().st_size > 0), None)
    if spans_path is None:
        return

    try:
        from agent_bench.telemetry.otel.visualize_spans import _load_spans, render_html

        spans = _load_spans(spans_path)
        if not spans:
            return
        output_path = run_dir / "spans_timeline.html"
        output_path.write_text(render_html(spans, spans_path), encoding="utf-8")
        logger.info("  spans timeline -> %s", output_path)
    except Exception as exc:  # noqa: BLE001 - best-effort artifact
        logger.warning("spans timeline generation failed for %s: %s", run_dir.name, exc)


def _install_signal_handlers():
    """Convert termination signals into KeyboardInterrupt so `finally` blocks run.

    Python's default for SIGTERM is to exit without unwinding, which skips the
    `finally` in execute_run that calls `harness.cleanup`. We can't catch
    SIGKILL, but converting SIGINT/SIGTERM/SIGHUP into KeyboardInterrupt covers
    the common "Ctrl+C / kill -15 / supervisor stop / terminal closed" cases.
    """
    def _to_interrupt(signum, _frame):
        logger.warning("received signal %s — raising KeyboardInterrupt to trigger cleanup", signum)
        raise KeyboardInterrupt()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _to_interrupt)
        except (ValueError, OSError):
            # SIGHUP is unavailable on Windows; secondary threads can't install.
            pass


@contextmanager
def _protect_cleanup_from_interrupts():
    """Let cleanup finish after the first interrupt has started unwinding."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous: dict[int, Any] = {}

    def _log_deferred(signum, _frame):
        logger.warning("received signal %s during cleanup; finishing cleanup first", signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _log_deferred)
        except (ValueError, OSError):
            pass
    try:
        yield
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def _merge_agent_completion_into_verdict(verdict: dict, result: dict) -> dict:
    """Add execution metadata without changing the verifier result."""
    merged = dict(verdict)
    exit_code = result.get("exit_code", 0)
    completion_status = result.get("agent_completion_status")
    if not completion_status:
        completion_status = (
            "normal_completion" if exit_code in (0, None) else "runtime_error"
        )

    raw_verifier_status = str(
        merged.get("verifier_status") or merged.get("status") or "ERROR"
    ).upper()
    merged["verifier_status"] = (
        raw_verifier_status
        if raw_verifier_status in {"PASS", "FAIL", "TIMEOUT", "ERROR"}
        else "ERROR"
    )

    reward = merged.get("reward")
    if reward is None:
        raw_reward = merged.get("reward_raw")
        try:
            reward = float(raw_reward) if raw_reward not in (None, "") else None
        except (TypeError, ValueError):
            reward = None
    if (
        reward is None
        and raw_verifier_status in {"PASS", "FAIL"}
        and isinstance(merged.get("test_pass_rate"), (int, float))
    ):
        reward = float(merged["test_pass_rate"])
    merged["reward"] = float(reward) if reward is not None else None

    test_pass_rate = merged.get("test_pass_rate")
    has_valid_verifier_measurement = (
        reward is not None or raw_verifier_status in {"PASS", "FAIL"}
    )
    merged["test_pass_rate"] = (
        float(test_pass_rate)
        if has_valid_verifier_measurement
        and isinstance(test_pass_rate, (int, float))
        else None
    )

    merged["agent_completion_status"] = completion_status
    merged["agent_execution_outcome"] = completion_status
    merged["agent_exit_code"] = exit_code
    completion_error = result.get("agent_completion_error") or result.get(
        "error_message"
    )
    if completion_status != "normal_completion" and not completion_error:
        completion_error = f"agent trajectory ended as {completion_status}"
    merged["agent_completion_error"] = completion_error or None
    return merged


def _roadmap_lifecycle_verifier_error(result: dict) -> Optional[str]:
    """Return the safety error that must prevent Roadmap verification."""
    lifecycle = result.get("agent_lifecycle")
    if not isinstance(lifecycle, dict):
        return None
    if lifecycle.get("workspaceQuiesced") is not False:
        return None
    return str(
        lifecycle.get("lifecycleError")
        or lifecycle.get("orphanCancellationError")
        or "Roadmap workspace could not be quiesced before verifier"
    )


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
    prepare_phase_timing: dict[str, Any] = {}
    harness_spans_path = run_dir / "spans.jsonl"
    trace_container = getattr(harness, "container", None)
    if hasattr(trace_container, "set_trace_path"):
        trace_container.set_trace_path(harness_spans_path)
    try:
        # Prepare environment. Passing instance_data lets benchmark-specific
        # harnesses (e.g. TerminalBenchHarness) read task.toml-derived fields
        # without re-loading the dataset; SWE-bench harnesses ignore the kwarg.
        # AB_*_CONFIG paths (set in main()) are the authoritative config sources
        # used by harness adapters when rendering per-run runtime config.
        environment_prepare_start_ns = time.time_ns()
        try:
            env_info = harness.prepare_environment(
                instance_id=run_config.instance_id,
                repo_id=run_config.repo_id,
                base_commit=run_config.base_commit,
                container_image=run_config.container_image,
                workspace_root=run_config.workspace_root,
                run_id=run_id,
                instance_data=instance_data,
            )
        finally:
            environment_prepare_end_ns = time.time_ns()
            prepare_phase_timing.update({
                "environment_prepare_start_ns": environment_prepare_start_ns,
                "environment_prepare_end_ns": environment_prepare_end_ns,
                "environment_prepare_seconds": max(0.0, (environment_prepare_end_ns - environment_prepare_start_ns) / 1e9),
            })
        if isinstance(env_info, dict):
            hpt = env_info.get("prepare_phase_timing")
            if isinstance(hpt, dict): prepare_phase_timing.update(hpt)

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
                if instance_data.get("benchmark_variant") == "roadmapbench":
                    from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
                        RoadmapBenchOpenClawCondition as _Cls,
                    )
                else:
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
            elif benchmark == "request_bench":
                from agent_bench.benchmarks.request_bench.conditions.openclaw_agent import (
                    RequestBenchOpenClawCondition as _Cls,
                )
            elif benchmark == "swebench_pro":
                from agent_bench.benchmarks.swebench_pro.conditions.openclaw_agent import (
                    SWEBenchProOpenClawCondition as _Cls,
                )
            elif benchmark == "swe_atlas_qa":
                from agent_bench.benchmarks.swe_atlas_qa.conditions.openclaw_agent import (
                    SWEAtlasQAOpenClawCondition as _Cls,
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
            benchmark = instance_data.get("benchmark", "swebench")
            if benchmark == "terminal_bench":
                from agent_bench.benchmarks.terminal_bench.conditions.openhands_agent import (
                    TerminalBenchOpenHandsCondition as _Cls,
                )
            elif benchmark == "swebench_pro":
                from agent_bench.benchmarks.swebench_pro.conditions.openhands_agent import (
                    SWEBenchProOpenHandsCondition as _Cls,
                )
            else:
                from agent_bench.benchmarks.swebench.conditions.openhands_agent import (
                    OpenHandsAgentCondition as _Cls,
                )
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

        elif condition_type == "hermes_agent":
            benchmark = instance_data.get("benchmark", "swebench")
            if benchmark == "terminal_bench":
                from agent_bench.benchmarks.terminal_bench.conditions.hermes_agent import (
                    TerminalBenchHermesCondition as _Cls,
                )
            elif benchmark == "deepresearch_bench":
                from agent_bench.benchmarks.deepresearch_bench.conditions.hermes_agent import (
                    DeepResearchHermesCondition as _Cls,
                )
            elif benchmark == "request_bench":
                from agent_bench.benchmarks.request_bench.conditions.hermes_agent import (
                    RequestBenchHermesCondition as _Cls,
                )
            elif benchmark == "swebench_pro":
                from agent_bench.benchmarks.swebench_pro.conditions.hermes_agent import (
                    SWEBenchProHermesCondition as _Cls,
                )
            elif benchmark == "swe_atlas_qa":
                from agent_bench.benchmarks.swe_atlas_qa.conditions.hermes_agent import (
                    SWEAtlasQAHermesCondition as _Cls,
                )
            else:
                from agent_bench.benchmarks.swebench.conditions.hermes_agent import (
                    HermesAgentCondition as _Cls,
                )
            condition = _Cls(
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
        skip_patch_gate = benchmark_kind in ("terminal_bench", "deepresearch_bench", "longbench", "request_bench", "swe_atlas_qa")
        agent_exit_code = result.get("exit_code", 0)
        is_roadmapbench = (
            benchmark_kind == "terminal_bench"
            and instance_data.get("benchmark_variant") == "roadmapbench"
        )
        phase_timing = result.get("phase_timing")
        if not isinstance(phase_timing, dict):
            phase_timing = {}
            result["phase_timing"] = phase_timing
        phase_timing.update(prepare_phase_timing)
        agent_start_ns = phase_timing.get("agent_execution_start_ns")
        environment_prepare_end_ns = phase_timing.get("environment_prepare_end_ns")
        if isinstance(agent_start_ns, int) and isinstance(environment_prepare_end_ns, int):
            phase_timing["post_prepare_before_agent_seconds"] = max(0.0, (agent_start_ns - environment_prepare_end_ns) / 1e9)
        if isinstance(agent_start_ns, int) and manifest.wall_start_ns:
            phase_timing["pre_agent_seconds"] = max(
                0.0, (agent_start_ns - manifest.wall_start_ns) / 1e9
            )

        lifecycle_verifier_error = (
            _roadmap_lifecycle_verifier_error(result)
            if is_roadmapbench else None
        )

        if lifecycle_verifier_error:
            verdict = {
                "status": "ERROR",
                "error_message": lifecycle_verifier_error,
                "lifecycle_error": lifecycle_verifier_error,
                "verifier_skipped": True,
            }
        elif agent_exit_code not in (0, None) and not is_roadmapbench:
            verdict = {
                "status": "ERROR",
                "error_message": result.get("error_message")
                or f"agent exited with code {agent_exit_code}",
                "agent_exit_code": agent_exit_code,
            }
        elif skip_patch_gate or (patch_path.exists() and patch_path.stat().st_size > 0):
            verifier_start_ns = time.time_ns()
            try:
                verdict = harness.run_tests(
                    container_id=container_id,
                    instance_id=run_config.instance_id,
                    workspace_root=env_info["workspace_root"],
                    timeout_seconds=run_config.wall_clock_budget_seconds,
                    patch=patch_path.read_text() if patch_path.exists() else "",
                    run_id=run_id,
                    instance_data=instance_data,
                    run_dir=run_dir,
                )
            finally:
                verifier_end_ns = time.time_ns()
                phase_timing.update({
                    "verifier_start_ns": verifier_start_ns,
                    "verifier_end_ns": verifier_end_ns,
                    "verifier_seconds": (
                        verifier_end_ns - verifier_start_ns
                    ) / 1e9,
                })
        else:
            verdict = {
                "status": "FAIL",
                "error_message": "No patch generated",
            }

        if is_roadmapbench:
            verdict = _merge_agent_completion_into_verdict(verdict, result)
            if isinstance(result.get("agent_lifecycle"), dict):
                verdict["agent_lifecycle"] = result["agent_lifecycle"]
        
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
        answer_src = verdict.get("answer_path") if isinstance(verdict, dict) else None
        if answer_src:
            try:
                src_p = Path(answer_src)
                if src_p.is_file() and src_p.stat().st_size > 0:
                    import shutil
                    shutil.copyfile(src_p, run_dir / "answer.md")
            except OSError as e:
                logger.warning("persist answer.md failed for %s: %s", run_id, e)
        
        # Update manifest
        manifest.status = verdict["status"]
        manifest.step_trace_path = result.get("step_trace_path")
        manifest.telemetry_path = result.get("telemetry_path")
        manifest.verdict_path = str(verdict_path)
        manifest.exit_code = result.get("exit_code", 0)
        manifest.agent_runtime_mode = result.get("agent_runtime_mode")
        manifest.agent_execution_outcome = verdict.get("agent_execution_outcome")
        manifest.agent_completion_error = verdict.get("agent_completion_error")
        manifest.agent_lifecycle = verdict.get("agent_lifecycle")
        manifest.agent_execution_seconds = result.get("agent_execution_seconds")
        manifest.phase_timing = result.get("phase_timing")
        manifest.verifier_status = verdict.get("verifier_status")
        manifest.reward = verdict.get("reward")
        manifest.test_pass_rate = verdict.get("test_pass_rate")
        manifest.trajectory_path = result.get("trajectory_path")
        manifest.token_accounting_path = result.get("token_accounting_path")
        manifest.trajectory_tokens = result.get("trajectory_tokens")
        manifest.serving_tokens = result.get("serving_tokens")
        manifest.runtime_observability_path = result.get(
            "runtime_observability_path"
        )
        manifest.runtime_observability = result.get("runtime_observability")
        manifest.error_message = (
            verdict.get("error_message")
            if manifest.status in ("ERROR", "TIMEOUT")
            else None
        )
        manifest.total_prompt_tokens = result["total_prompt_tokens"]
        manifest.total_completion_tokens = result["total_completion_tokens"]
        
    except Exception as e:
        logger.error(f"Run {run_id} failed with error: {e}", exc_info=True)
        manifest.status = "ERROR"
        manifest.error_message = str(e)[:500]
    
    finally:
        # Preserve legacy duration_seconds (pre-cleanup) for historical comparability.
        manifest.wall_end_ns = time.time_ns()
        manifest.duration_seconds = (
            (manifest.wall_end_ns - manifest.wall_start_ns) / 1e9
        )
        cleanup_error = None
        cleanup_start_ns = time.time_ns()
        try:
            cleanup_kwargs = dict(env_info or {})
            for k in ("container_id", "_harness"):
                cleanup_kwargs.pop(k, None)
            with _protect_cleanup_from_interrupts():
                harness.cleanup(container_id, **cleanup_kwargs)
        except BaseException:
            logger.exception("cleanup failed")
            cleanup_error = "harness cleanup failed"
        finally:
            cleanup_end_ns = time.time_ns()
            final_phase_timing = dict(manifest.phase_timing) if isinstance(manifest.phase_timing, dict) else {}
            if isinstance(env_info, dict):
                hpt = env_info.get("prepare_phase_timing")
                if isinstance(hpt, dict): final_phase_timing.update(hpt)
            final_phase_timing.update(prepare_phase_timing)
            final_phase_timing.update({
                "cleanup_start_ns": cleanup_start_ns,
                "cleanup_end_ns": cleanup_end_ns,
                "cleanup_seconds": max(0.0, (cleanup_end_ns - cleanup_start_ns) / 1e9),
                "total_instance_end_ns": cleanup_end_ns,
            })
            if manifest.wall_start_ns:
                manifest.total_instance_seconds = max(0.0, (cleanup_end_ns - manifest.wall_start_ns) / 1e9)
                final_phase_timing["total_instance_seconds"] = manifest.total_instance_seconds
            manifest.phase_timing = final_phase_timing
            trajectory_path = run_dir / "trajectory.jsonl"
            if trajectory_path.exists():
                try:
                    from agent_bench.benchmarks.conditions.openclaw_measurement import (
                        append_trajectory_event,
                    )

                    append_trajectory_event(
                        run_dir,
                        "cleanup",
                        status="error" if cleanup_error else "completed",
                        error=cleanup_error,
                        lifecycle=(result.get("agent_lifecycle") if "result" in locals() else None),
                    )
                except Exception:
                    logger.exception("trajectory cleanup event write failed")
            _maybe_render_spans_timeline(run_dir, manifest.telemetry_path)
            manifest.save(get_manifest_path(run_id, run_config.output_dir))
            if hasattr(trace_container, "set_trace_path"):
                trace_container.set_trace_path(None)
    
    return manifest


def run_sweep(benchmark_path: Path, harness_path: Path, model_path: Path, serving_path: Path,
              benchmark_data_root: Optional[Path], instance_data_path: Path,
              retry_errors: bool = True, max_workers: int = 1):
    """Run full sweep, composed from benchmark / harness / backend JSON configs.

    `max_workers > 1` runs containers concurrently. Each container has its
    own Gateway on its own port (range 18800-18899 stride 4 → ~25 parallel),
    its own /testbed workspace, and tags OTel spans with oc.run_id so the
    shared collector output remains demuxable. Real bottleneck at scale is
    vLLM throughput (single endpoint).
    """

    # Load sweep config by composing the four axes.
    sweep_config = load_composed_config(benchmark_path, harness_path, model_path, serving_path)

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
        instance_rows = json.load(f)
    instance_rows = _attach_swebench_pro_runtime_resources(instance_rows)
    instances = {inst["instance_id"]: inst for inst in instance_rows}

    # Initialize harness. AB_DEPLOYMENT_CONTAINER_BACKEND selects the SWE-bench backend:
    #   docker      (default) — SWEBenchHarness;     openclaw container + task container
    #                            on a shared docker network. Host runs only vLLM.
    #   singularity            — SingularityHarness; openclaw .sif + task .sif (peer
    #                            instances). Same isolation model in singularity-land.
    # Terminal-Bench / DR-Bench instances are auto-routed to their respective
    # harnesses regardless of HARNESS_BACKEND (per-instance dispatch in _one below).
    _NON_SWE_BENCHMARKS = ("terminal_bench", "deepresearch_bench", "longbench", "request_bench", "swebench_pro", "swe_atlas_qa")
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
    needs_rb = any(
        inst.get("benchmark") == "request_bench" for inst in instances.values()
    )
    needs_sp = any(
        inst.get("benchmark") == "swebench_pro" for inst in instances.values()
    )
    needs_sa = any(
        inst.get("benchmark") == "swe_atlas_qa" for inst in instances.values()
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
        if benchmark_data_root is None:
            raise RuntimeError(
                "sweep contains SWE-bench instances but --benchmark-data-root was "
                "not provided"
            )
        backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker").lower()
        if backend == "docker":
            container = DockerContainer()
        elif backend == "singularity":
            container = SingularityContainer()
        else:
            raise ValueError(
                f"Unknown AB_DEPLOYMENT_CONTAINER_BACKEND={backend!r}; expected docker|singularity. "
                "Host mode was removed — agent runtimes must always be containerized."
            )

        if harness_choice == "openhands":
            from agent_bench.benchmarks.swebench.openhands.app import SWEBenchOpenHandsHarness
            logger.info(
                "Using SWEBenchOpenHandsHarness with %s "
                "(openhands runtime + task container + exec-bridge).",
                type(container).__name__,
            )
            harness = SWEBenchOpenHandsHarness(benchmark_data_root, container=container)
        elif harness_choice == "hermes":
            from agent_bench.benchmarks.swebench.hermes.app import SWEBenchHermesHarness
            logger.info(
                "Using SWEBenchHermesHarness with %s "
                "(hermes runtime + task container + exec-bridge).",
                type(container).__name__,
            )
            harness = SWEBenchHermesHarness(benchmark_data_root, container=container)
        elif harness_choice == "openclaw":
            logger.info(
                "Using SWEBenchHarness with %s "
                "(openclaw runtime + task container + exec-bridge).",
                type(container).__name__,
            )
            harness = SWEBenchHarness(benchmark_data_root, container=container)
        else:
            raise ValueError(
                f"Unknown harness={harness_choice!r}; expected openclaw|openhands|hermes."
            )
    else:
        logger.info("Sweep has no SWE-bench instances; skipping SWE-bench harness build.")

    sp_harness = None
    if needs_sp:
        sp_backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker").lower()
        if sp_backend != "docker":
            raise ValueError(
                "swebench_pro currently supports AB_DEPLOYMENT_CONTAINER_BACKEND=docker only; "
                "PBS/Singularity support is deferred."
            )
        sp_root = benchmark_data_root or Path(os.environ.get("AB_BENCHMARK_DATA_ROOT") or os.getcwd())
        if harness_choice == "openhands":
            from agent_bench.benchmarks.swebench_pro.openhands.app import (
                SWEBenchProOpenHandsHarness as sp_cls,
            )
            sp_harness = sp_cls(sp_root, container=DockerContainer())
        elif harness_choice == "hermes":
            from agent_bench.benchmarks.swebench_pro.hermes.app import (
                SWEBenchProHermesHarness as sp_cls,
            )
            sp_harness = sp_cls(sp_root, container=DockerContainer())
        elif harness_choice == "openclaw":
            from agent_bench.benchmarks.swebench_pro.openclaw.app import (
                SWEBenchProHarness as sp_cls,
            )
            sp_harness = sp_cls(sp_root, container=DockerContainer())
        else:
            raise ValueError(
                f"swebench_pro supports harness=openclaw|openhands|hermes (got {harness_choice!r})."
            )
        logger.info("Constructed %s for SWE-bench Pro.", type(sp_harness).__name__)

    # Lazy-build the Request-Bench harness only if any instance is Request-Bench.
    rb_harness = None
    if needs_rb:
        rb_backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker").lower()
        if rb_backend != "docker":
            raise ValueError(
                f"request_bench only supports AB_DEPLOYMENT_CONTAINER_BACKEND=docker (got {rb_backend!r})."
            )
        if harness_choice == "hermes":
            from agent_bench.benchmarks.request_bench.app import RequestBenchHermesHarness
            logger.info("Constructing RequestBenchHermesHarness.")
            rb_harness = RequestBenchHermesHarness(benchmark_data_root)
        elif harness_choice == "openclaw":
            from agent_bench.benchmarks.request_bench.app import RequestBenchHarness
            logger.info("Constructing RequestBenchHarness.")
            rb_harness = RequestBenchHarness(benchmark_data_root)
        else:
            raise ValueError(
                f"request_bench supports harness=openclaw|hermes (got {harness_choice!r})."
            )

    # Lazy-build the DR harness only if any instance in this sweep is DR-Bench.
    # Same two-container model as TB but with a fixed lightweight sandbox image
    # and host-side LLM-as-judge scoring (RACE + FACT via DeepSeek-V4).
    dr_harness = None
    if needs_dr:
        dr_backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker").lower()
        if dr_backend != "docker":
            raise ValueError(
                f"deepresearch_bench only supports AB_DEPLOYMENT_CONTAINER_BACKEND=docker "
                f"(got {dr_backend!r}); singularity backend not implemented."
            )
        if harness_choice == "hermes":
            from agent_bench.benchmarks.deepresearch_bench.app import (
                DeepResearchHermesHarness as dr_cls,
            )
        elif harness_choice == "openclaw":
            from agent_bench.benchmarks.deepresearch_bench.app import (
                DeepResearchHarness as dr_cls,
            )
        else:
            raise ValueError(
                f"deepresearch_bench supports harness=openclaw|hermes (got {harness_choice!r})."
            )
        if harness is not None:
            logger.info(
                "Sweep contains deepresearch_bench instances; constructing "
                "%s alongside %s.", dr_cls.__name__, type(harness).__name__,
            )
        else:
            logger.info("Constructing %s.", dr_cls.__name__)
        dr_harness = dr_cls(benchmark_data_root)

    # Lazy-build the LongBench harness only if any instance is LongBench.
    # Same two-container model as DR; per-subtask instance loops examples over
    # warm containers, retrieving from the shared ClawRAG service.
    lb_harness = None
    if needs_lb:
        from agent_bench.benchmarks.longbench.app import LongBenchHarness
        lb_backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker").lower()
        if lb_backend != "docker":
            raise ValueError(
                f"longbench only supports AB_DEPLOYMENT_CONTAINER_BACKEND=docker (got {lb_backend!r})."
            )
        logger.info("Constructing LongBenchHarness.")
        lb_harness = LongBenchHarness(benchmark_data_root)

    # Lazy-build the SWE-Atlas-QA harness only if any instance is SWE-Atlas-QA.
    sa_harness = None
    if needs_sa:
        sa_backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker").lower()
        if sa_backend == "docker":
            sa_container = DockerContainer()
        elif sa_backend == "singularity":
            sa_container = SingularityContainer()
        else:
            raise ValueError(
                f"Unknown AB_DEPLOYMENT_CONTAINER_BACKEND={sa_backend!r}; "
                "expected docker|singularity."
            )
        if harness_choice == "openclaw":
            from agent_bench.benchmarks.swe_atlas_qa.openclaw.app import (
                SWEAtlasQAHarness as sa_cls,
            )
        elif harness_choice == "hermes":
            from agent_bench.benchmarks.swe_atlas_qa.hermes.app import (
                SWEAtlasQAHermesHarness as sa_cls,
            )
        else:
            raise ValueError(
                f"swe_atlas_qa supports harness=openclaw|hermes (got {harness_choice!r})."
            )
        logger.info(
            "Constructing SWEAtlasQA harness (%s) with %s.",
            harness_choice,
            type(sa_container).__name__,
        )
        sa_harness = sa_cls(
            benchmark_data_root,
            container=sa_container,
        )

    # Lazy-build the TB harness only if any instance in this sweep is TB.
    # Harness choice now applies to Terminal-Bench too:
    #   openclaw  -> TerminalBenchOpenClawHarness (Docker or Singularity)
    #   openhands -> TerminalBenchOpenHandsHarness (docker)
    #   hermes    -> TerminalBenchHermesHarness (docker)
    tb_harness = None
    if needs_tb:
        tb_backend = os.environ.get("AB_DEPLOYMENT_CONTAINER_BACKEND", "docker").lower()
        if harness_choice == "openclaw":
            from agent_bench.benchmarks.terminal_bench.app import (
                TerminalBenchOpenClawHarness as tb_cls,
            )
            if tb_backend == "docker":
                tb_container = DockerContainer()
            elif tb_backend == "singularity":
                tb_container = SingularityContainer()
            else:
                raise ValueError(
                    f"Unknown AB_DEPLOYMENT_CONTAINER_BACKEND={tb_backend!r} for terminal_bench; "
                    "expected docker|singularity."
                )
            tb_harness = tb_cls(benchmark_data_root, container=tb_container)
        elif harness_choice == "openhands":
            if tb_backend != "docker":
                raise ValueError(
                    "terminal_bench + openhands currently requires "
                    "AB_DEPLOYMENT_CONTAINER_BACKEND=docker"
                )
            from agent_bench.benchmarks.terminal_bench.openhands.app import (
                TerminalBenchOpenHandsHarness as tb_cls,
            )
            tb_harness = tb_cls(benchmark_data_root, container=DockerContainer())
        elif harness_choice == "hermes":
            if tb_backend != "docker":
                raise ValueError(
                    "terminal_bench + hermes currently requires "
                    "AB_DEPLOYMENT_CONTAINER_BACKEND=docker"
                )
            from agent_bench.benchmarks.terminal_bench.hermes.app import (
                TerminalBenchHermesHarness as tb_cls,
            )
            tb_harness = tb_cls(benchmark_data_root, container=DockerContainer())
        else:
            from agent_bench.benchmarks.terminal_bench.app import (
                TerminalBenchDockerHarness,
                TerminalBenchSingularityHarness,
            )
            if tb_backend == "docker":
                tb_cls = TerminalBenchDockerHarness
            elif tb_backend == "singularity":
                tb_cls = TerminalBenchSingularityHarness
            else:
                raise ValueError(
                    f"Unknown AB_DEPLOYMENT_CONTAINER_BACKEND={tb_backend!r} for terminal_bench; "
                    "expected docker|singularity."
                )
            tb_harness = tb_cls(benchmark_data_root)
        if harness is not None:
            logger.info(
                "Sweep contains terminal-bench instances; constructing "
                "%s alongside %s.",
                type(tb_harness).__name__, type(harness).__name__,
            )
        else:
            logger.info("Constructing %s.", type(tb_harness).__name__)

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
            elif bm == "request_bench":
                h = rb_harness
            elif bm == "swebench_pro":
                h = sp_harness
            elif bm == "swe_atlas_qa":
                h = sa_harness
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
    return completed, failed


def _attach_swebench_pro_runtime_resources(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pro_rows = [
        dict(inst)
        for inst in instance_rows
        if isinstance(inst, dict) and inst.get("benchmark") == "swebench_pro"
    ]
    if not pro_rows:
        return instance_rows

    from agent_bench.benchmarks.swebench_pro.loader import attach_official_resources

    attached = {
        row["instance_id"]: row
        for row in attach_official_resources(pro_rows)
    }
    return [
        attached.get(inst.get("instance_id"), inst) if isinstance(inst, dict) else inst
        for inst in instance_rows
    ]


def main():
    parser = argparse.ArgumentParser(description="Agent_Bench experiment driver (composed config)")
    parser.add_argument("--benchmark-config", type=Path, required=True,
                       help="Benchmark config (profiles/benchmarks/<benchmark>/<sweep>.json)")
    parser.add_argument("--harness-config", type=Path, required=True,
                       help="Harness config (profiles/harnesses/<name>.yaml)")
    parser.add_argument("--model-config", type=Path, required=True,
                       help="Model config (profiles/models/<name>.json)")
    parser.add_argument("--serving-config", type=Path, required=True,
                       help="Serving config (profiles/serving/<engine>/<name>.yaml)")
    # benchmark-data-root is only required when the sweep contains SWE-bench
    # instances — pure terminal_bench sweeps don't use it.
    parser.add_argument("--benchmark-data-root", type=Path, required=False,
                       help="Path to the active benchmark data root (required for SWE-bench instances)")
    parser.add_argument("--benchmark-instance-data", type=Path, required=True,
                       help="Path to instance data JSON (SWE-bench or TB)")
    parser.add_argument("--no-retry-errors", action="store_true",
                       help="Do not retry ERROR runs")
    parser.add_argument("--max-workers", type=int,
                       default=int(os.environ.get("AB_RUN_MAX_WORKERS", "1")),
                       help="Concurrent containers (default 1; env AB_RUN_MAX_WORKERS)")

    args = parser.parse_args()

    # Stash the composed-config paths in env so harness code can render
    # per-run runtime config without re-parsing CLI args.
    os.environ["AB_BENCHMARK_CONFIG"] = str(args.benchmark_config.resolve())
    os.environ["AB_HARNESS_CONFIG"] = str(args.harness_config.resolve())
    os.environ["AB_MODEL_CONFIG"] = str(args.model_config.resolve())
    os.environ["AB_SERVING_CONFIG"] = str(args.serving_config.resolve())
    os.environ["AB_BENCHMARK_INSTANCE_DATA"] = str(args.benchmark_instance_data.resolve())

    completed, failed = run_sweep(
        benchmark_path=args.benchmark_config,
        harness_path=args.harness_config,
        model_path=args.model_config,
        serving_path=args.serving_config,
        benchmark_data_root=args.benchmark_data_root,
        instance_data_path=args.benchmark_instance_data,
        retry_errors=not args.no_retry_errors,
        max_workers=args.max_workers,
    )
    # A cell that completed ZERO runs while hitting errors produced nothing usable
    # (every run threw before scoring — e.g. an infra failure like a docker-network
    # collision; a real verifier FAIL counts as completed, so this never masks a
    # legit task failure). Exit non-zero so the fan-out orchestrator requeues it
    # instead of recording a silently-empty "done" cell.
    if completed == 0 and failed > 0:
        logger.error("no runs completed (all errored): %d error(s) — exiting non-zero "
                     "so the fan-out requeues this cell.", failed)
        return 1
    return 0


if __name__ == "__main__":
    _install_signal_handlers()
    sys.exit(main())
