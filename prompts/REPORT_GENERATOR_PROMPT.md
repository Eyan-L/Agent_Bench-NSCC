# OpenClaw + SWE-bench Report Generator Prompt

Please use the `openclaw_report_generator` skill.

Metric tables and figures exist. I need a structured report that directly answers **RQ1**, **RQ2**, and **RQ3**, with disciplined claim/evidence mapping and a reviewer-pushback pass.

## Inputs
- Long-format Parquet from the analyzer (source of truth).
- Wide CSV tables per metric family.
- Figure files (referenced by path; do not regenerate).

## Required section order
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

## Per-RQ rules

**RQ1**: Lead with step-type time share and tool-call ratio. Contrast AR (near-100% `LLM_INFER`) with OpenClaw's mixed composition. Structural framing — do not evaluate "better/worse".

**RQ2**: Lead with GPU utilization during trajectory wall time, then CPU, then stalls/fragmentation. Explicit measured-vs-inferred. If MFU/MBU/HBM BW are unavailable, say so in-text — do not silently omit. Retry-wasted resource is reported as its own result, not folded into GPU idleness.

**RQ3**: **Headline result is the ranking-alignment test** between TTFT/TPOT and end-to-end latency. If they disagree, that disagreement is the thesis, not a footnote. Also report pass rate and pass-under-latency-budget at 5, 10, 30 min.

## Claim / evidence discipline

Produce `claims.json` alongside the prose. Every quantitative claim maps to:

```json
{ "claim_text": "...", "metric": "...", "artifact_path": "...", "kind": "measured|inferred" }
```

Any claim in the prose without a matching entry is a bug.

## Unsupported-claim flags

Refuse to produce, or prominently flag in `TODO_CLAIMS.md`, any of these:

- "OpenClaw is better" claims based on pass rate without CIs.
- Hardware-efficiency claims without direct counter evidence.
- Generalization beyond SWE-bench / Python code tasks.
- Architectural causes when only runtime composition was varied.
- Quantitative comparisons that aren't paired by `instance_id`.

## Required limitations

Must cover:
- Dataset scope (Verified-mini + Verified; Python-centric; any stratification).
- Model/serving scope (single model family, specific engine, specific configurations).
- Inferred quantities with their parameter values.
- Unavailable counters (list them explicitly and say why).
- Confounders (retry policy, tool profile, time budget — each a knob, not a constant).

## Pushback pass (required)

Before returning the draft, produce `pushback.md` with:
- Three strongest reviewer counter-arguments.
- The weakest claim each targets.
- A mitigation per counter: **rewrite**, **additional experiment**, or **accept-caveat**.

## Out of scope
- Recomputing metrics — ingest only.
- Regenerating figures — reference by path.

## Output format
- Markdown by default; LaTeX only if asked (minimal preamble, no custom commands).
- One claim per paragraph where possible.
- Tables before prose.

## Style
- Third person, past tense for results, present tense for framing.
- Short paragraphs.
- No hedging adverbs for quantitative claims — use the number.
