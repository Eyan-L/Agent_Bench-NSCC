"""Fail-closed benchmark execution policy for the top-level run entrypoint."""
from __future__ import annotations

from pathlib import Path

from agent_bench.core.spec import ConfigError


def validate_benchmark_counts(completed: int, failed: int) -> None:
    """Reject sweeps containing infrastructure ERRORs.

    Verifier PASS/FAIL/TIMEOUT/DEFERRED outcomes are counted as completed by
    ``runtime.driver.run_sweep``.  ``failed`` is reserved for ERROR/exception
    paths, so any non-zero value means the experiment cell is incomplete and
    must not be reported to PBS as a successful job.
    """
    if failed > 0:
        raise ConfigError(
            "benchmark sweep contained infrastructure errors: "
            f"{completed} completed, {failed} error(s)"
        )


def _strict_run_benchmark(ctx) -> None:
    """Equivalent to core.execution._run_benchmark, but preserve sweep status."""
    from agent_bench.core import execution
    from agent_bench.runtime.driver import run_sweep

    resolved = ctx.resolved
    benchmark_path = resolved.profile_paths.get("benchmark")
    harness_path = resolved.profile_paths.get("harness")
    model_path = resolved.profile_paths["model"]
    serving_path = resolved.profile_paths["serving"]
    if benchmark_path is None or harness_path is None:
        raise ConfigError(
            "experiments must reference benchmark.profile and harness.profile to execute"
        )

    instance_data = ctx.instance_data_path or execution.materialize_benchmark_data(
        resolved,
        pbs_spec=ctx.plan.pbs_spec,
        env=ctx.env,
    )
    benchmark_data_root = execution._optional_path_from_raw_or_env(
        resolved,
        "benchmark_data_root",
        "AB_BENCHMARK_DATA_ROOT",
        ctx.env,
    )

    ctx.env["AB_BENCHMARK_CONFIG"] = str(Path(benchmark_path).resolve())
    ctx.env["AB_HARNESS_CONFIG"] = str(Path(harness_path).resolve())
    ctx.env["AB_MODEL_CONFIG"] = str(Path(model_path).resolve())
    ctx.env["AB_SERVING_CONFIG"] = str(Path(serving_path).resolve())
    ctx.env["AB_BENCHMARK_INSTANCE_DATA"] = str(Path(instance_data).resolve())
    ctx.env["AB_INSTANCE_DATA"] = str(Path(instance_data).resolve())
    ctx.env["AB_EXPERIMENT_CONFIG"] = str(resolved.spec.config_path.resolve())

    completed, failed = run_sweep(
        benchmark_path=benchmark_path,
        harness_path=harness_path,
        model_path=model_path,
        serving_path=serving_path,
        benchmark_data_root=benchmark_data_root,
        instance_data_path=instance_data,
        retry_errors=ctx.retry_errors,
        max_workers=ctx.max_workers,
    )
    # Keep the normal diagnostic artifact even when the cell will fail closed.
    execution._write_run_summary_log(ctx)
    validate_benchmark_counts(completed, failed)


def install_strict_benchmark_failure_policy() -> None:
    """Make the top-level execution stage fail when any run is ERROR."""
    from agent_bench.core import execution

    current = execution._run_benchmark
    if getattr(current, "_agent_bench_strict_benchmark_failures", False):
        return
    setattr(
        _strict_run_benchmark,
        "_agent_bench_strict_benchmark_failures",
        True,
    )
    execution._run_benchmark = _strict_run_benchmark
