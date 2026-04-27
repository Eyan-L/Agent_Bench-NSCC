---
name: openclaw_swebench_code_agent_plan
description: Build an execution-ready validation plan for comparing trajectory-based OpenClaw code-agent workflows against traditional autoregressive serving on SWE-bench tasks.
---

# Purpose

Use this skill when the user wants a concrete validation plan for studying how **trajectory-based code-agent workflows** differ from **traditional AR serving** on software engineering tasks.

This skill is optimized for:
- OpenClaw as agent runtime
- SWE-bench Verified-mini and Verified
- characterization of execution composition
- characterization of resource bottlenecks
- comparison of token-level versus task-level evaluation metrics

Default scope:
- **code agent only**
- no web-search agent
- no document-analysis agent

# Core Goal

Convert the user's RQs into a **trackable, execution-ready study plan**.

Do not return a loose brainstorm.
Do not over-expand the scope.
Do not optimize only for pass rate.

Prioritize:
1. structural characterization,
2. clean comparison methodology,
3. observability and instrumentation,
4. reproducible runtime setup.

# Required Study Questions

The plan must support the following question families.

## RQ1: Execution composition
How does a trajectory-based workflow change the execution composition compared with traditional AR serving?

The plan must include:
- unified analysis unit
- trajectory and attempt definition
- step taxonomy
- step/time composition breakdown
- tool ratio / non-LLM ratio / waiting ratio
- input-output token ratio

## RQ2: Resource bottlenecks
Do these structural differences change the resource bottlenecks of the workload?

The plan must include:
- GPU resource view
- CPU and subprocess view
- memory view
- fragmentation / stall view
- retry overhead quantification

## RQ3: Evaluation metrics
Are token-centric metrics still sufficient for evaluating agentic workflow?

The plan must include:
- TTFT / TPOT on LLM spans
- end-to-end task latency
- p95 / p99 task latency
- at least one controlled serving-configuration comparison to test whether token-level improvements align with task-level improvements

# Required Definitions

The final plan must explicitly define the following units.

## Task
One SWE-bench problem instance.

## AR request / turn
A single prompt-response execution used by the AR baseline to generate a patch or patch proposal.

## Trajectory
One complete OpenClaw workflow for solving a task, from the initial issue input to final termination, including all intermediate LLM calls, tool calls, validation runs, retries, and failures.

## Attempt
One individual try within a trajectory. Prefer the operational definition:
- from the point the agent begins to form a concrete patch proposal,
- through validation / tests,
- until that proposal is accepted, rejected, or abandoned.

## Step
One atomic runtime event category that can be logged and timed.

# Required Step Taxonomy

At minimum, the plan must use these step categories:
- `LLM_INFER`
- `TOOL_READ`
- `TOOL_WRITE`
- `TOOL_EXEC`
- `WAIT_STALL`

Optional subtypes are allowed, such as:
- `TOOL_READ_SEARCH`
- `TOOL_EXEC_TEST`
- `TOOL_EXEC_LINT`
- `WAIT_TOOL_RETURN`
- `WAIT_ORCHESTRATION`

But the final metrics must still roll up to the required top-level categories.

# Required Comparison Design

The plan must compare at least these workloads.

## Baseline A: AR-issue-only
- issue text only
- single-shot or fixed small number of AR generations
- no interactive repo exploration

## Baseline B: AR-static-context
- issue text plus offline-constructed static context
- still AR-style, non-interactive execution
- no iterative repo interaction during the request

## Baseline C: OpenClaw code agent
- multi-step trajectory
- repo exploration allowed
- code edits allowed
- test / validation commands allowed
- retries allowed

# Fairness Constraints

The plan must state how to keep the comparison fair.

Prefer the following defaults unless the user overrides them:
- same benchmark instances
- same model family when possible
- same serving engine for AR and OpenClaw LLM calls
- same repo snapshot and task environment
- same time budget per task
- same max token budget policy where appropriate

# Required Workload / Dataset Guidance

Use the following default dataset plan.

## Pilot
- SWE-bench Verified-mini
- goal: instrumentation, trace completeness, pipeline debugging

## Main evaluation
- SWE-bench Verified
- goal: produce final structural and evaluation results

The plan should also propose task stratification such as:
- search-heavy tasks
- edit-test-loop tasks
- retry-heavy tasks

If useful, these may be defined post hoc from observed trajectories instead of pre-labeled categories.

# Required Runtime Guidance

The plan must include a runtime stack with the following components.

## Benchmark harness
- SWE-bench execution / evaluation harness
- repo checkout and test execution environment

## Agent runtime
- OpenClaw

## LLM serving engine
- one controllable serving endpoint for both AR and agentic conditions whenever possible

## Placement recommendation
Default recommendation:
- run OpenClaw **inside** the SWE-bench task container for the first version
- set workspace to the checked-out repository root
- avoid introducing an extra remote sandbox layer unless there is a strong reason

# Required OpenClaw Configuration Guidance

The plan must provide a concrete configuration sketch covering:
- model endpoint
- workspace root
- `tools.profile` selection
- allow / deny policy
- telemetry / diagnostics
- disabling irrelevant tools such as browser, messaging, canvas, and unrelated MCP bundles

# Required Instrumentation Guidance

The plan must specify three levels of observability.

## 1. Step trace
Each step should record timing, type, identifiers, and token usage if applicable.

## 2. Resource telemetry
Time-series telemetry for CPU, GPU, memory, and subprocess activity.

## 3. Benchmark verdicts
Final task status, pass/fail, timeouts, and artifact locations.

The plan must explain how these are aligned on a shared clock or timestamp base.

# Required Raw Trace Schema

The final plan must include at least one concrete raw schema with fields such as:
- `trajectory_id`
- `attempt_id`
- `step_id`
- `parent_step_id` if used
- `step_type`
- `tool_name`
- `command`
- `start_ts`
- `end_ts`
- `duration_ms`
- `prompt_tokens`
- `completion_tokens`
- `exit_code`
- `error_type`
- `repo_id`
- `instance_id`
- `config_id`
- `model_name`

# Required Metrics

The final plan must define metrics in four groups.

## Composition metrics
- step count breakdown
- time breakdown
- tool-call ratio
- non-LLM time fraction
- waiting/stall fraction
- attempts per trajectory
- input/output token ratio

## Resource metrics
- GPU utilization
- SM occupancy / MFU / MBU if available
- HBM bandwidth utilization
- GPU idle fraction while task remains active
- CPU utilization
- subprocess CPU time
- DRAM / RSS usage
- PCIe or network I/O if relevant

## Evaluation metrics
- TTFT
- TPOT
- end-to-end task latency
- p95 / p99 task latency
- pass rate or success rate
- success rate under latency budget

## Retry / fragmentation metrics
- failed-attempt cost fraction
- wasted resource fraction from retries
- number of LLM bursts per trajectory
- average LLM burst duration
- inter-step idle gaps

# Required Baselines and Ablations

The plan must contain:
- primary AR baselines
- OpenClaw baseline
- at least one agent ablation
- at least one runtime / serving configuration sweep for RQ3

Example serving configurations for RQ3:
- latency-oriented
- throughput-oriented
- context-cache-oriented

# Required Output Structure

Use exactly the following section order:

# Study Goal
# Scope
# Assumptions
# Research Questions Mapping
# Methodology
# Workload
# Dataset / Input Tasks
# Engine / Runtime Setup
# OpenClaw Configuration
# Instrumentation Plan
# Raw Trace Schema
# Baselines and Ablations
# Metrics
# Expected Outcomes
# Risks and Threats to Validity
# Execution Plan

# Writing Style

- Be concrete.
- Be implementation-oriented.
- Avoid generic brainstorming language.
- Distinguish clearly between directly measurable and inferred quantities.
- Make reasonable assumptions explicitly.
- Keep the plan scoped to code agent only.

# Execution Plan Requirements

Always end with phased execution.

## Phase 0
Sanity / tracing validation.

## Phase 1
Pilot on Verified-mini.

## Phase 2
Full evaluation on Verified.

For each phase, specify:
- objective
- what to run
- what to check
- expected deliverables
