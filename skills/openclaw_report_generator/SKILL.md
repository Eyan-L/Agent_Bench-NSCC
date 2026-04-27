---
name: openclaw_report_generator
description: Turn computed metrics from the OpenClaw + SWE-bench study into a structured research report that directly answers RQ1 (execution composition), RQ2 (resource bottlenecks), and RQ3 (evaluation metrics). Use this skill when metric tables and figures exist and the user needs narrative, not more numbers — section drafts, figure captions, claim-vs-evidence mapping, limitations writing, or reviewer-style pushback checks. Triggers on phrases like "write up the results", "draft the RQ1 section", "what claims can we make", "write the limitations", "generate the report", "write the paper section", or any request to convert metric CSVs into prose. Use this whenever the user wants to move from tables to text on this study, even if they don't name the skill.
---

# Purpose

Generate a structured report, or specific sections thereof, from the metric tables produced by `openclaw_trace_analyzer`. The report must distinguish claims the data supports from claims it does not, and must cleanly answer RQ1, RQ2, and RQ3 without overclaiming.

The generator **ingests metrics**; it does not recompute them. If a claim requires a number that isn't in the analyzer's output, the right move is to ask for the analyzer to produce it — not to estimate it in prose.

# Core Deliverables

- Section drafts: Abstract, Introduction, Background, Methodology, Results (RQ1/RQ2/RQ3), Discussion, Limitations, Conclusion.
- Figure captions tied to the figures emitted by the analyzer.
- A claim/evidence mapping: every substantive claim in the prose is linked to a specific metric and a specific artifact path.
- A limitations section that explicitly calls out measured-vs-inferred, unavailable counters, and fairness caveats.
- A reviewer-pushback pass that stress-tests the weakest claims before the draft is returned.

# Required Structure (use this exact order)

1. Abstract
2. Introduction
3. Background (brief: AR serving vs agentic workflow)
4. Methodology — reference the plan, do not restate it in full
5. Results — RQ1 (composition)
6. Results — RQ2 (resource bottlenecks)
7. Results — RQ3 (evaluation)
8. Discussion
9. Limitations and Threats to Validity
10. Conclusion

# Per-RQ Reporting Rules

## RQ1 — Composition
Lead with the step-type time share and tool-call ratio. Contrast AR (near-100% `LLM_INFER`) with OpenClaw's mixed composition. Report non-LLM fraction, waiting/stall fraction, and input/output token ratio as the core structural signature.

**Do not claim "better" or "worse".** RQ1 is a structural question, not an evaluative one. The reader should leave this section knowing *how the workloads differ in shape*, not *which one is winning*.

## RQ2 — Resource bottlenecks
Lead with GPU utilization during trajectory wall time, then CPU utilization, then stalls and fragmentation. Be explicit about what's directly measured vs inferred. If MFU/MBU/HBM BW are unavailable, say so in-text — do not just omit the counters and let the reader assume the GPU is being used efficiently. Retry-wasted-resource fraction must be reported as a separate result, not buried in a paragraph about GPU idleness.

## RQ3 — Evaluation
**The core result is the ranking-alignment test**: does ordering conditions by TTFT/TPOT produce the same ordering as end-to-end task latency? Report both rankings as a table up front. If they disagree, that disagreement **is the headline finding** — treat it as the thesis of the section, not a footnote. Also report pass rate and pass-under-latency-budget for all three budgets (5, 10, 30 min) so the reader can judge the effect size.

# Claim / Evidence Discipline

For every quantitative claim, the report must be able to answer three questions:

1. Which metric supports this?
2. Which artifact (CSV path, figure path, table row) does the number come from?
3. Is the underlying metric **measured** or **inferred**? If inferred, with what parameter value?

Produce a machine-readable `claims.json` alongside the prose. Each entry:

```json
{
  "claim_text": "...",
  "metric": "ttft_p95",
  "artifact_path": "tables/evaluation_wide.csv",
  "kind": "measured"
}
```

Any claim in the prose that does not have a matching entry in `claims.json` is a bug.

# Unsupported-Claim Flags

The report generator must refuse to produce, or flag prominently in a `TODO_CLAIMS.md`, any of these:

- "OpenClaw is better" claims based on pass rate **without confidence intervals**.
- Claims of hardware-efficiency improvement **without direct counter evidence** (i.e. without DCGM/MFU/MBU data).
- Claims of **generalization beyond SWE-bench / Python code tasks**.
- Claims about **model-architectural causes** when only runtime composition was varied.
- Any quantitative comparison that isn't paired by `instance_id`.

When the analyzer's tables make one of these claims tempting, the generator's job is to refuse and push back — not to obey.

# Required Limitations Content

The limitations section must cover, at minimum:

- **Dataset scope**: SWE-bench Verified-mini + Verified only; Python-centric repos; specific instance subset if stratified.
- **Model and serving scope**: single model family, specific serving engine, specific configurations.
- **Inferred quantities**: list every inferred metric with its parameter value (GPU-idle threshold, LLM-burst T, etc.).
- **Unavailable counters**: which hardware metrics were not collectable and why (e.g. "DCGM not available in the target environment, so MFU is not reported").
- **Confounders**: retry policy, tool profile, time budget — each of these is a knob, not a natural constant, and each one could independently drive the observed structural differences.

# Pushback Pass

Before returning the draft, simulate a reviewer and produce a separate `pushback.md` with:

- Three strongest counter-arguments a skeptical reader would raise.
- For each, the weakest claim in the current draft that it targets.
- A proposed mitigation per counter: **rewrite**, **additional experiment**, or **accept-and-caveat**.

This is not optional. It is the mechanism that keeps the report honest when no external reviewer is in the loop yet.

# Implementation Guidance

- Ingest the long-format Parquet from the analyzer as the source of truth; do not re-compute metrics here.
- Prefer plain Markdown output unless the user asks for LaTeX. If LaTeX, use a minimal preamble and no custom commands.
- Figures are referenced by path; do not regenerate them.
- Numbers in prose are rounded according to the metric's effective precision (e.g. p95 latencies to the nearest 0.1 s; token counts to integers). The analyzer's raw precision is not always the right display precision.

# Out of Scope

- Computing metrics (in `openclaw_trace_analyzer`).
- Running experiments (in `openclaw_experiment_runner`).
- Instrumentation (in `openclaw_instrumentation`).
- Plan writing (in `openclaw_swebench_code_agent_plan`).

# Writing Style

- Third person, past tense for results, present tense for framing.
- Short paragraphs. One claim per paragraph where possible.
- **Tables before prose** — if the table says it, don't re-say it in words. Use prose for the interpretation the table can't carry on its own.
- No hedging adverbs ("somewhat", "fairly", "quite") for quantitative claims; use the number instead.
