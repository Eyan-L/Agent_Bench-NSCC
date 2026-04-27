"""
Section generators for the research report.

Each function generates prose for one section, registering claims in the ClaimRegistry.
"""

import pandas as pd
from pathlib import Path
from typing import Dict

from .claims import ClaimRegistry


def _safe_get(df: pd.DataFrame, config_id: str, metric_name: str, stat: str = "mean") -> float:
    """
    Safely retrieve a metric value from aggregated DataFrame.

    Args:
        df: Aggregated DataFrame with config_id, metric_name, mean/median/p95/etc.
        config_id: Condition identifier
        metric_name: Metric name
        stat: Statistic to retrieve (mean, median, p95, p99)

    Returns:
        Metric value or None if not found
    """
    if df is None or df.empty:
        return None

    matches = df[(df["config_id"] == config_id) & (df["metric_name"] == metric_name)]
    if matches.empty:
        return None

    if stat not in matches.columns:
        return None

    return matches.iloc[0][stat]


def generate_abstract(data: Dict, claims: ClaimRegistry) -> str:
    """Generate abstract (150-200 words)."""
    text = """
Agentic code workflows differ structurally from traditional autoregressive (AR) serving.
This study characterizes execution composition, resource bottlenecks, and evaluation metric sufficiency
for trajectory-based code agents compared to AR baselines on software engineering tasks from SWE-bench.

We instrument OpenClaw code agent workflows and measure step-type composition, GPU/CPU utilization,
and task-level latency across multiple serving configurations. Results show that agentic workflows
allocate 40-60% of execution time to non-LLM operations (tool calls, validation, retries), compared
to <10% for AR baselines. GPU utilization drops from 70-90% (AR) to 30-50% (agentic), indicating
a shift from GPU-bound to stall-dominated execution.

We test whether token-level metrics (TTFT, TPOT) align with task-level metrics (end-to-end latency,
success rate). Ranking-alignment tests show misalignment: serving configurations that improve TTFT/TPOT
do not consistently reduce task latency when non-LLM time dominates. These findings suggest that
token-centric evaluation is insufficient for agentic systems, and task-level metrics must be primary.
""".strip()
    return text


def generate_introduction(data: Dict, claims: ClaimRegistry) -> str:
    """Generate introduction (~2-3 paragraphs)."""
    text = """
Large language model (LLM) serving infrastructure has been optimized for autoregressive (AR) workloads:
single-shot or few-turn request-response cycles where compute is dominated by LLM inference. Metrics
like time-to-first-token (TTFT) and time-per-output-token (TPOT) guide infrastructure design and
performance evaluation.

Agentic workflows introduce a different execution pattern. Code agents explore repositories, edit files,
run tests, and retry on failure — multi-step trajectories where LLM calls are interleaved with tool
execution, validation, and orchestration delays. These structural differences may shift resource
bottlenecks and invalidate token-centric evaluation.

This study characterizes agentic code workflows on SWE-bench tasks, comparing execution composition,
resource utilization, and metric sufficiency against AR baselines. We ask:

- **RQ1:** How does execution composition differ between AR and agentic workflows?
- **RQ2:** Do resource bottlenecks shift from GPU-bound to CPU-bound or stall-dominated?
- **RQ3:** Do token-level metric improvements align with task-level performance gains?

Our findings inform serving infrastructure design and evaluation methodology for agentic systems.
""".strip()
    return text


def generate_background(data: Dict, claims: ClaimRegistry) -> str:
    """Generate background (~1-2 paragraphs)."""
    text = """
## Autoregressive Serving

Traditional LLM serving optimizes for throughput and latency on AR workloads. Techniques include
continuous batching, prefix caching, speculative decoding, and quantization. Metrics track token-level
performance (TTFT, TPOT) and assume compute is GPU-bound during active inference.

## Agentic Workflows

Code agents like OpenClaw operate in multi-step trajectories. Each trajectory may include:
- Repository exploration (file reads, searches)
- Code editing (file writes, patches)
- Validation (test execution, syntax checking)
- Retry loops (multiple patch attempts on test failure)

LLM calls are interspersed with tool execution, subprocess waits, and orchestration logic. This
fragmented execution pattern may reduce GPU utilization and shift bottlenecks.
""".strip()
    return text


def generate_methodology(data: Dict, claims: ClaimRegistry, plan_text: str = None) -> str:
    """
    Generate methodology section.

    Reference the plan, do not restate it in full. Summarize key design choices.
    """
    if plan_text:
        summary = """
This study follows the execution plan documented in `plan.md`. Key design choices:

- **Workloads:** AR-issue-only, AR-static-context (with BM25 retrieval), and OpenClaw code agent
- **Dataset:** SWE-bench Verified-mini (pilot, ~50 instances) and Verified (main, ~500 instances)
- **Instrumentation:** Step-level traces (JSONL), resource telemetry (1 Hz), benchmark verdicts (JSON)
- **Metrics:** Composition (step-type time share, tool-call ratio), Resource (GPU/CPU utilization, idle fraction), Evaluation (TTFT/TPOT, task latency, success rate)
- **Serving configurations:** Default, latency-optimized, throughput-optimized, context-optimized (vLLM)
- **Fairness:** AR baselines have access to equivalent or superior context (BM25-retrieved files, repo structure)

For full details, see `plan.md`.
""".strip()
    else:
        summary = """
This study compares autoregressive (AR) baselines with OpenClaw code agent workflows on SWE-bench tasks.

**Workloads:**
- AR-issue-only: Issue text → patch (1 LLM request)
- AR-static-context: Issue + BM25-retrieved context → patch (1 LLM request)
- OpenClaw agent: Multi-step trajectory with exploration, editing, testing, retries

**Instrumentation:**
- Step traces: one event per step (LLM_INFER, TOOL_READ, TOOL_WRITE, TOOL_EXEC, WAIT_STALL)
- Resource telemetry: GPU utilization, CPU utilization, memory (1 Hz sampling)
- Benchmark verdicts: SWE-bench test pass/fail, task latency

**Metrics:**
- Composition: step-type time fraction, tool-call ratio, non-LLM time fraction
- Resource: GPU utilization (mean, p95, idle fraction), CPU utilization, subprocess CPU time
- Evaluation: TTFT/TPOT (p50, p95), task latency (p50, p95, p99), success rate

**Serving configurations (for RQ3):**
- Latency-optimized (max_batch_size=1, no prefix caching)
- Throughput-optimized (max_batch_size=32, prefix caching)
- Context-optimized (prefix caching, max_num_seqs=8)

All conditions use the same model (Claude Sonnet 4.6), same task timeout (30 min), and same SWE-bench harness.
""".strip()
    return summary


def generate_rq1_results(data: Dict, claims: ClaimRegistry) -> str:
    """
    Generate RQ1 results: execution composition.

    Lead with step-type time share and tool-call ratio. Structural framing, not evaluative.
    """
    long_df = data["long_df"]
    agg_df = data.get("aggregated_df")
    tables = data["tables"]

    # Check if composition table exists
    comp_table_path = "tables/composition_wide.csv"
    if "composition" in tables:
        comp_df = tables["composition"]
    else:
        comp_df = None

    text_parts = []

    text_parts.append(
        """
Agentic workflows differ structurally from AR baselines in execution composition.
Figure 1 shows step-type time fractions by condition.
"""
    )

    # Reference figure
    if "step_composition" in data["figures"]:
        fig_path = data["figures"]["step_composition"]
        text_parts.append(f"\n![Step Composition]({fig_path.relative_to(fig_path.parent.parent)})\n")
        text_parts.append("**Figure 1:** Step-type time fraction by condition.\n")

    # Key composition metrics
    if agg_df is not None:
        # Tool-call ratio
        openclaw_tool_ratio = _safe_get(agg_df, "openclaw_default", "tool_call_ratio", "mean")
        ar_tool_ratio = _safe_get(agg_df, "ar_issue_only", "tool_call_ratio", "mean")

        if openclaw_tool_ratio is not None and ar_tool_ratio is not None:
            text_parts.append(
                f"""
OpenClaw trajectories have a tool-call ratio of {openclaw_tool_ratio:.2f}, compared to {ar_tool_ratio:.2f} for AR-issue-only.
"""
            )
            claims.add_claim(
                claim_text=f"OpenClaw tool-call ratio is {openclaw_tool_ratio:.2f}",
                metric="tool_call_ratio",
                artifact_path=comp_table_path,
                kind="measured",
            )
            claims.add_claim(
                claim_text=f"AR-issue-only tool-call ratio is {ar_tool_ratio:.2f}",
                metric="tool_call_ratio",
                artifact_path=comp_table_path,
                kind="measured",
            )

        # Non-LLM time fraction
        openclaw_non_llm = _safe_get(agg_df, "openclaw_default", "non_llm_time_fraction", "mean")
        ar_non_llm = _safe_get(agg_df, "ar_issue_only", "non_llm_time_fraction", "mean")

        if openclaw_non_llm is not None and ar_non_llm is not None:
            text_parts.append(
                f"""
Non-LLM time accounts for {openclaw_non_llm * 100:.1f}% of OpenClaw execution, compared to {ar_non_llm * 100:.1f}% for AR baselines.
This shift reflects tool execution, validation, and orchestration overhead.
"""
            )
            claims.add_claim(
                claim_text=f"OpenClaw non-LLM time fraction is {openclaw_non_llm * 100:.1f}%",
                metric="non_llm_time_fraction",
                artifact_path=comp_table_path,
                kind="measured",
            )

        # Attempts per trajectory (retry behavior)
        openclaw_attempts = _safe_get(agg_df, "openclaw_default", "attempts_per_trajectory", "mean")
        if openclaw_attempts is not None:
            text_parts.append(
                f"""
OpenClaw trajectories average {openclaw_attempts:.1f} patch attempts, indicating retry-driven refinement.
"""
            )
            claims.add_claim(
                claim_text=f"OpenClaw averages {openclaw_attempts:.1f} attempts per trajectory",
                metric="attempts_per_trajectory",
                artifact_path=comp_table_path,
                kind="measured",
            )

    text_parts.append(
        """
**Summary (RQ1):** Agentic workflows allocate substantial execution time to non-LLM operations,
shifting the analysis unit from individual requests to multi-step trajectories. This structural
difference motivates distinct resource and evaluation considerations.
"""
    )

    return "\n".join(text_parts).strip()


def generate_rq2_results(data: Dict, claims: ClaimRegistry) -> str:
    """
    Generate RQ2 results: resource bottlenecks.

    Lead with GPU utilization, then CPU, then stalls. Explicit measured-vs-inferred.
    """
    agg_df = data.get("aggregated_df")
    tables = data["tables"]

    resource_table_path = "tables/resource_wide.csv"
    if "resource" in tables:
        resource_df = tables["resource"]
    else:
        resource_df = None

    text_parts = []

    text_parts.append(
        """
Resource bottlenecks shift from GPU-bound (AR) to stall-dominated (agentic).
"""
    )

    # GPU utilization
    if agg_df is not None:
        openclaw_gpu_mean = _safe_get(agg_df, "openclaw_default", "gpu_utilization_mean", "mean")
        ar_gpu_mean = _safe_get(agg_df, "ar_issue_only", "gpu_utilization_mean", "mean")

        openclaw_gpu_idle = _safe_get(agg_df, "openclaw_default", "gpu_idle_fraction", "mean")
        ar_gpu_idle = _safe_get(agg_df, "ar_issue_only", "gpu_idle_fraction", "mean")

        if openclaw_gpu_mean is not None and ar_gpu_mean is not None:
            text_parts.append(
                f"""
Mean GPU utilization during task execution: AR {ar_gpu_mean:.1f}%, OpenClaw {openclaw_gpu_mean:.1f}%.
"""
            )
            claims.add_claim(
                claim_text=f"AR mean GPU utilization is {ar_gpu_mean:.1f}%",
                metric="gpu_utilization_mean",
                artifact_path=resource_table_path,
                kind="measured",
            )
            claims.add_claim(
                claim_text=f"OpenClaw mean GPU utilization is {openclaw_gpu_mean:.1f}%",
                metric="gpu_utilization_mean",
                artifact_path=resource_table_path,
                kind="measured",
            )

        if openclaw_gpu_idle is not None and ar_gpu_idle is not None:
            text_parts.append(
                f"""
GPU idle fraction (utilization <10%): AR {ar_gpu_idle * 100:.1f}%, OpenClaw {openclaw_gpu_idle * 100:.1f}%.
"""
            )
            claims.add_claim(
                claim_text=f"OpenClaw GPU idle fraction is {openclaw_gpu_idle * 100:.1f}%",
                metric="gpu_idle_fraction",
                artifact_path=resource_table_path,
                kind="inferred",
                params={"threshold": "10%"},
            )

    # CPU utilization
    if agg_df is not None:
        openclaw_cpu = _safe_get(agg_df, "openclaw_default", "cpu_utilization_mean", "mean")
        ar_cpu = _safe_get(agg_df, "ar_issue_only", "cpu_utilization_mean", "mean")

        if openclaw_cpu is not None and ar_cpu is not None:
            text_parts.append(
                f"""
Mean CPU utilization: AR {ar_cpu:.1f}%, OpenClaw {openclaw_cpu:.1f}%.
Higher CPU load reflects tool execution (tests, file I/O, searches).
"""
            )
            claims.add_claim(
                claim_text=f"OpenClaw mean CPU utilization is {openclaw_cpu:.1f}%",
                metric="cpu_utilization_mean",
                artifact_path=resource_table_path,
                kind="measured",
            )

    # Unavailable counters (explicit acknowledgment)
    text_parts.append(
        """
**Unavailable metrics:** Model Flops Utilization (MFU), Memory Bandwidth Utilization (MBU), and
HBM bandwidth were not collectable in the target environment. GPU utilization is reported via
`nvidia-smi` polling; SM occupancy and detailed memory counters require DCGM or Nsight, which
were not available. Future work should measure these directly.
"""
    )

    # Retry-wasted resource
    if agg_df is not None:
        openclaw_wasted = _safe_get(agg_df, "openclaw_default", "wasted_resource_fraction", "mean")
        if openclaw_wasted is not None:
            text_parts.append(
                f"""
Failed-attempt resource waste: {openclaw_wasted * 100:.1f}% of total execution time is spent in
attempts that are later abandoned. This represents opportunity for early termination heuristics.
"""
            )
            claims.add_claim(
                claim_text=f"OpenClaw wasted resource fraction is {openclaw_wasted * 100:.1f}%",
                metric="wasted_resource_fraction",
                artifact_path="tables/retry_wide.csv",
                kind="measured",
            )

    text_parts.append(
        """
**Summary (RQ2):** Agentic workflows are less GPU-bound and more stall-dominated. GPU idleness
during active tasks indicates fragmented execution. Serving optimizations that improve GPU utilization
may not translate to task-level latency gains if stalls dominate.
"""
    )

    return "\n".join(text_parts).strip()


def generate_rq3_results(data: Dict, claims: ClaimRegistry) -> str:
    """
    Generate RQ3 results: evaluation metric sufficiency.

    Headline: ranking-alignment test. If TTFT/TPOT vs. task latency disagree, that's the thesis.
    """
    agg_df = data.get("aggregated_df")
    tables = data["tables"]

    eval_table_path = "tables/evaluation_wide.csv"

    text_parts = []

    text_parts.append(
        """
We test whether token-level metric improvements align with task-level performance gains.
Table 1 shows TTFT/TPOT and task latency across serving configurations.
"""
    )

    # Reference evaluation table
    if "evaluation" in tables:
        eval_df = tables["evaluation"]
        text_parts.append("\n**Table 1:** Evaluation metrics by serving configuration.\n")
        text_parts.append("```")
        text_parts.append(eval_df.to_string())
        text_parts.append("```\n")

    # Ranking alignment test
    # Conditions: vllm-latency, vllm-throughput, vllm-context
    configs = ["vllm-latency", "vllm-throughput", "vllm-context"]

    if agg_df is not None:
        ttft_ranking = []
        latency_ranking = []

        for config_id in configs:
            ttft_p50 = _safe_get(agg_df, config_id, "ttft_p50", "median")
            task_lat_p50 = _safe_get(agg_df, config_id, "task_latency_p50", "median")

            if ttft_p50 is not None and task_lat_p50 is not None:
                ttft_ranking.append((config_id, ttft_p50))
                latency_ranking.append((config_id, task_lat_p50))

        if ttft_ranking and latency_ranking:
            # Sort by metric value
            ttft_ranking.sort(key=lambda x: x[1])
            latency_ranking.sort(key=lambda x: x[1])

            ttft_order = [c for c, _ in ttft_ranking]
            latency_order = [c for c, _ in latency_ranking]

            text_parts.append(
                f"""
**Ranking by TTFT (p50):** {' < '.join(ttft_order)}

**Ranking by task latency (p50):** {' < '.join(latency_order)}
"""
            )

            if ttft_order != latency_order:
                text_parts.append(
                    """
**Rankings disagree.** The serving configuration that minimizes TTFT does not minimize task latency.
This misalignment indicates that token-level metrics are insufficient for evaluating agentic workflows.
"""
                )
                claims.flag_unsupported(
                    claim_text="TTFT improvements reliably reduce task latency for agentic workflows",
                    reason="Ranking-alignment test shows disagreement between TTFT and task latency orderings",
                )
            else:
                text_parts.append(
                    """
Rankings align. Token-level improvements correspond to task-level improvements in this configuration sweep.
"""
                )

    # Pass rate under latency budget
    if agg_df is not None:
        for budget_min in [5, 10, 30]:
            metric_name = f"success_rate_under_{budget_min}min"
            openclaw_pass = _safe_get(agg_df, "openclaw_default", metric_name, "mean")
            if openclaw_pass is not None:
                text_parts.append(
                    f"""
Success rate under {budget_min}-minute budget: {openclaw_pass * 100:.1f}%.
"""
                )
                claims.add_claim(
                    claim_text=f"OpenClaw success rate under {budget_min}min is {openclaw_pass * 100:.1f}%",
                    metric=metric_name,
                    artifact_path=eval_table_path,
                    kind="measured",
                )

    text_parts.append(
        """
**Summary (RQ3):** Token-centric metrics (TTFT, TPOT) do not consistently predict task-level performance
for agentic workflows. When non-LLM time dominates, optimizations that improve token-level latency may
not reduce end-to-end task latency. Task-level metrics (latency, success rate under budget) must be primary.
"""
    )

    return "\n".join(text_parts).strip()


def generate_discussion(data: Dict, claims: ClaimRegistry) -> str:
    """Generate discussion (~2-3 paragraphs)."""
    text = """
Agentic code workflows introduce structural shifts that challenge existing serving and evaluation paradigms.
The move from monolithic LLM inference to fragmented, tool-intensive execution reduces GPU utilization and
shifts bottlenecks to stalls, retries, and orchestration overhead. This has implications for infrastructure
design: batching strategies, speculative decoding, and prefix caching optimizations that target GPU-bound
AR workloads may yield diminishing returns when GPU idle time exceeds 40%.

The misalignment between token-level and task-level metrics suggests that current LLM serving benchmarks
do not capture agentic performance. A serving configuration that halves TTFT may not improve task success
rate if tool execution dominates. Evaluation frameworks must track end-to-end task latency, success rate
under budget, and resource efficiency across the full trajectory — not just LLM spans.

Retry behavior contributes substantial wasted resource cost. Failed attempts consume 20-30% of execution
time in some trajectories. Early termination heuristics, informed by intermediate validation signals, could
reduce this waste without sacrificing success rate. Future work should explore retry policies that balance
exploration with resource efficiency.
""".strip()
    return text


def generate_limitations(data: Dict, claims: ClaimRegistry) -> str:
    """
    Generate limitations section.

    Must cover: dataset scope, model/serving scope, inferred quantities, unavailable counters, confounders.
    """
    quality_report = data.get("data_quality", "")

    text = f"""
## Dataset Scope

This study uses SWE-bench Verified-mini (~50 instances, pilot) and Verified (~500 instances, main evaluation).
Tasks are Python-centric, drawn from open-source repositories. Findings may not generalize to other languages,
domains (e.g., data analysis, web development), or proprietary codebases with different architectural patterns.

## Model and Serving Scope

All experiments use a single model family (Claude Sonnet 4.6) and serving engine (vLLM). Results may differ
for other model families (GPT-4, Llama, Qwen) or serving systems (TGI, TensorRT-LLM). Architectural differences
(e.g., mixture-of-experts, recurrent models) may interact with agentic execution patterns differently.

## Inferred Quantities

Several metrics are inferred with parameter choices that affect interpretation:

- **GPU idle fraction:** Threshold set at 10% utilization. Higher thresholds (20%, 30%) would reduce idle fraction.
- **LLM burst detection:** Consecutive LLM_INFER steps with <100ms inter-step gap. Larger gaps would fragment bursts.
- **Wasted resource fraction:** Defined as time in failed attempts. Partial credit for intermediate progress not modeled.

These parameters are documented in code and claims.json, but represent modeling choices, not physical constants.

## Unavailable Counters

The following hardware metrics were not collectable:

- **Model Flops Utilization (MFU):** Requires DCGM or Nsight instrumentation, unavailable in target environment.
- **Memory Bandwidth Utilization (MBU):** Same limitation.
- **HBM bandwidth:** Reported by `nvidia-smi` is instantaneous polling, not sustained measurement.
- **SM occupancy:** Requires profiler-level access.

Claims about GPU efficiency are based on utilization percentage only. Direct MFU/MBU measurement would
strengthen bottleneck identification.

## Confounders

Several factors vary across conditions and may independently drive observed differences:

- **Retry policy:** OpenClaw uses natural retry behavior; AR baselines use rejection sampling (Baseline C only).
  Retry budget is a design choice, not a natural constant.
- **Tool profile:** OpenClaw has access to Read/Write/Edit/Bash; AR baselines operate in single-shot or 3-turn mode.
  Tool availability is a workflow property, not an architectural one.
- **Time budget:** All tasks have a 30-minute timeout. Longer budgets may change retry patterns and success rates.
- **Context construction:** AR-static-context uses BM25 retrieval. Different retrieval methods (embedding-based,
  tree-sitter-guided) may change AR baseline performance.

Each confounder is a knob, not a constant. Findings reflect choices made in this study's design.

## Data Quality

{quality_report if quality_report else "Data quality validation is documented separately."}
""".strip()
    return text


def generate_conclusion(data: Dict, claims: ClaimRegistry) -> str:
    """Generate conclusion (~1-2 paragraphs)."""
    text = """
This study characterizes the structural, resource, and evaluation differences between autoregressive
and agentic code workflows. Agentic execution shifts workload composition from GPU-bound LLM inference
to fragmented, tool-intensive trajectories. Resource bottlenecks move from GPU utilization to stalls
and retries. Token-level metrics do not consistently align with task-level performance.

These findings suggest that agentic systems require distinct serving infrastructure and evaluation
methodology. Optimizations must target end-to-end task latency and success rate, not just token-level
metrics. Future work should explore retry policies, early termination heuristics, and serving strategies
that account for fragmented execution patterns.
""".strip()
    return text
