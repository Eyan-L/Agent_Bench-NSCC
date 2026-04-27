# OpenClaw + SWE-bench Code-Agent Validation Prompt

You are a systems research planning assistant.

Your task is to produce an **execution-ready validation plan** for a study that compares:
1. **traditional autoregressive (AR) serving**, and
2. **trajectory-based code-agent workflow using OpenClaw**

on **SWE-bench style software engineering tasks**.

## Study scope
- Focus on **code agent only**.
- Do **not** expand to web search agents or document-analysis agents.
- The plan must target **OpenClaw** as the agent runtime.
- The benchmark/task source must be **SWE-bench Verified-mini for pilot** and **SWE-bench Verified for main evaluation** unless there is a strong reason to deviate.
- The study goal is **structural characterization and evaluation methodology**, not just raw pass-rate competition.

## Research questions

### RQ1. How does a trajectory-based workflow change the execution composition compared with traditional autoregressive (AR) serving?
Why it matters:
Traditional AR serving is naturally organized around requests or turns, whereas agentic workflow is organized around multi-step task execution. If the two workloads differ at this structural level, then request-level or token-level views alone will miss the properties that later shape resource bottlenecks and evaluation.

Subtasks:
- Establish a unified analysis unit for workload comparison by aggregating step-level traces into requests/turns for AR and trajectories for agentic workflow.
- Trajectory: one full workflow instance for a task, from the initial input to the final termination, including all intermediate LLM calls, tool calls, and retries.
- Attempt: one individual try within a trajectory.
- Compare the execution composition of AR and agentic workflow, especially the fraction of execution spent in LLM inference, tool steps, and waiting/stall steps.
- Measure the input/output token ratio of each workload to characterize its execution shape.

Reference metrics:
- step type breakdown
- tool-call ratio
- fraction of non-LLM steps
- waiting/stall fraction
- input/output token ratio
- arithmetic intensity for LLM spans if appropriate

### RQ2. Do these structural differences change the resource bottlenecks of the workload?
Why it matters:
Traditional AR serving is typically dominated by model execution on GPU resources. In agentic workflow, the workload may become less purely GPU-centric and instead exhibit different bottlenecks, such as higher CPU involvement, larger CPU memory demand, lower effective GPU utilization due to dependency stalls, or more fragmented execution across steps.

Subtasks:
- Compare the resource composition and dominant bottlenecks of traditional AR and agentic workflow across CPU, GPU, memory, PCIe, and network resources.
- Measure how dependency stalls and fragmented execution reduce hardware utilization.
- Quantify how retries and repeated attempts increase wasted resource consumption.

Reference metrics:
- GPU utilization, SM occupancy / MFU / MBU if available
- HBM bandwidth utilization
- CPU utilization / CPU time fraction, DRAM usage
- PCIe traffic
- network I/O / external wait ratio if relevant
- end-to-end latency

### RQ3. Are token-centric metrics still sufficient for evaluating agentic workflow?
Why it matters:
Traditional AR is usually evaluated with token-level metrics such as TTFT and TPOT. However, agentic workflow is task-oriented, so token-level metrics may no longer be reliable proxies for overall task performance.

Subtasks:
- Report TTFT and TPOT for the LLM steps of all workloads as the token-level baseline.
- Report end-to-end latency for the full task as the task-level baseline.
- Compare whether improvements under token-level metrics are consistent with improvements under task-level metrics.

Reference metrics:
- TTFT, TPOT of LLM spans
- end-to-end task latency
- p95 / p99 task latency
- pass rate / success rate under latency budget

## Hard requirements
Your answer must be concrete and implementation-oriented.

You must explicitly include all of the following:

### 1) Study framing
- study goal
- scope
- assumptions
- what claims this study can support
- what claims it cannot support

### 2) Units of analysis
Define clearly:
- task
- AR request / turn
- trajectory
- attempt
- step

### 3) Required step taxonomy
At minimum, define these step types:
- `LLM_INFER`
- `TOOL_READ`
- `TOOL_WRITE`
- `TOOL_EXEC`
- `WAIT_STALL`

You may add subtypes if useful, but the plan must at least support these categories.

### 4) Comparison methodology
The plan must specify:
- what is compared
- what is held constant
- what is varied
- fairness constraints
- how to build the AR baseline fairly

The AR side must include at least:
- `AR-issue-only`
- `AR-static-context`

The OpenClaw side must be a multi-step code-agent workflow with repo interaction and validation.

### 5) Workload and datasets
The plan must specify:
- pilot dataset
- main dataset
- filtering / sampling strategy
- how to stratify tasks or trajectories

### 6) Engine / runtime stack
The plan must specify:
- model choice principles
- serving engine choice
- OpenClaw runtime placement
- workspace / sandbox setup
- whether OpenClaw runs inside the SWE-bench container or outside it
- key OpenClaw tool restrictions

Default assumption to evaluate unless you give a stronger alternative:
- OpenClaw runs **inside** the SWE-bench task container
- workspace points to the checked-out repo
- tools are restricted to coding-related capabilities only

### 7) OpenClaw configuration guidance
Include a concrete configuration sketch for:
- workspace root
- tool profile
- deny / allow policy
- telemetry / tracing settings
- how to disable irrelevant tools such as browser / messaging / unrelated MCP tools

### 8) Instrumentation plan
The plan must define:
- step trace schema
- attempt/trajectory linkage
- token accounting
- resource telemetry
- GPU metrics
- CPU / subprocess metrics
- timestamp alignment method
- what gets stored as JSONL / CSV / Parquet

### 9) Trace schema
Include at least one proposed raw event schema with fields such as:
- trajectory_id
- attempt_id
- step_id
- step_type
- tool_name or command
- start_ts
- end_ts
- prompt_tokens
- completion_tokens
- exit_code
- error_type
- repo_id
- instance_id
- config_id

### 10) Baselines and ablations
The plan must include:
- primary AR baseline
- stronger AR baseline with static context
- OpenClaw code-agent baseline
- ablations on agent behavior or runtime configuration
- at least one set of serving configurations for testing RQ3

### 11) Execution phases
The plan must include:
- Phase 0: sanity / tracing validation
- Phase 1: pilot on Verified-mini
- Phase 2: full evaluation on Verified
- deliverables for each phase

## Output format
Use exactly this structure:

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

## Style constraints
- Be precise and concrete.
- Prefer explicit experimental design over brainstorming.
- Clearly separate directly measurable quantities from inferred quantities.
- State assumptions rather than asking follow-up questions unless absolutely necessary.
- Keep the plan scoped to **code agent only**.
