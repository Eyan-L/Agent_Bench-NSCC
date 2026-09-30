---
name: verify
description: Validate a multi-target software implementation against its requirements and repository checks.
---

# Verification

Use this skill before declaring `DONE`.

1. Re-read the roadmap and make a checklist of every target and compatibility
   requirement.
2. Inspect the combined changes and confirm they follow shared APIs and
   repository conventions.
3. Run focused repository tests for each target, then the broadest practical
   regression suite within the remaining time.
4. Diagnose failures, fix the causes, and rerun relevant tests.

Report `DONE` only when the implementation and available validation evidence
cover the complete roadmap.
