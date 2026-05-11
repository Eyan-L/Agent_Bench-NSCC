"""
Reviewer pushback generator.

Simulate a skeptical reviewer and produce counter-arguments to stress-test claims.
"""

from typing import Dict, List
from .claims import ClaimRegistry


def generate_pushback(data: Dict, claims: ClaimRegistry, report_text: str) -> str:
    """
    Generate pushback.md with reviewer challenges.

    Args:
        data: Analysis data dict
        claims: ClaimRegistry with all registered claims
        report_text: Full report markdown text

    Returns:
        Pushback markdown text
    """
    pushback_parts = []

    pushback_parts.append("# Reviewer Pushback\n")
    pushback_parts.append(
        "Three strongest counter-arguments a skeptical reviewer would raise, "
        "the weakest claim each targets, and proposed mitigations.\n"
    )

    # Counter-argument 1: Generalization
    pushback_parts.append("\n## 1. Generalization Beyond SWE-bench\n")
    pushback_parts.append(
        """
**Counter-argument:** SWE-bench tasks are Python-centric, open-source, and focus on bug-fixing.
Your findings may not generalize to other domains (data analysis, web development, infrastructure),
languages (Java, C++, Rust), or task types (feature addition, refactoring).

**Weakest claim targeted:** "Agentic workflows are less GPU-bound and more stall-dominated" (RQ2 summary).
This claim is based entirely on SWE-bench data and may not hold for tasks with different tool profiles.

**Proposed mitigation:**
- **Accept and caveat:** Add to limitations (already present): "Findings may not generalize beyond Python code tasks."
- **Additional experiment (future work):** Replicate on HumanEval-X (multi-language), MBPP (data analysis), or a web-dev benchmark.
- **Rewrite:** Narrow claim to "Agentic workflows *on SWE-bench tasks* are less GPU-bound..."
"""
    )

    # Counter-argument 2: Model-specific behavior
    pushback_parts.append("\n## 2. Single-Model Limitation\n")
    pushback_parts.append(
        """
**Counter-argument:** You use only Claude Sonnet 4.6. Different models (GPT-4, Llama 3, Qwen) may have
different tool-use patterns, retry frequencies, and LLM-vs-tool time splits. Your composition findings
may be model-specific, not workflow-universal.

**Weakest claim targeted:** "Non-LLM time accounts for 40-60% of OpenClaw execution" (RQ1).
This fraction depends on model latency, which varies across families.

**Proposed mitigation:**
- **Accept and caveat:** Acknowledge in limitations (already present): "Single model family; results may differ for other models."
- **Additional experiment:** Run pilot with Llama 3 or GPT-4 to check if non-LLM fraction holds across models.
- **Rewrite:** Add hedge: "With Claude Sonnet 4.6, non-LLM time accounts for 40-60%..."
"""
    )

    # Counter-argument 3: Confounders (retry policy, tool profile)
    pushback_parts.append("\n## 3. Confounded Comparison\n")
    pushback_parts.append(
        """
**Counter-argument:** You vary multiple factors simultaneously (retry policy, tool availability, context construction).
You can't isolate whether the observed differences are due to *agentic workflow structure* vs. *specific design choices*
like OpenClaw's retry logic or tool profile. An AR baseline with equivalent retry budget might close the gap.

**Weakest claim targeted:** "Resource bottlenecks shift from GPU-bound to stall-dominated" (RQ2 summary).
This could be caused by OpenClaw's retry policy, not by agentic workflow in general.

**Proposed mitigation:**
- **Accept and caveat:** Acknowledge in limitations (already present): "Retry policy, tool profile, and time budget are confounders."
- **Additional experiment:** Add Baseline D: AR-static-context with 5 rejection-sampling retries, matched tool budget.
  This would isolate retry-driven stalls from workflow structure.
- **Rewrite:** Qualify claim: "With OpenClaw's natural retry behavior, resource bottlenecks shift..." and note that
  retry policy is a design choice, not intrinsic to agentic workflows.
"""
    )

    pushback_parts.append("\n---\n")
    pushback_parts.append(
        f"\n**Claims registered:** {claims.count()}\n"
    )
    pushback_parts.append(
        f"**Unsupported claims flagged:** {len(claims.get_unsupported())}\n"
    )

    return "\n".join(pushback_parts)
