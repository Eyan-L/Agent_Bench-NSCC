---
name: verify
description: Validate a multi-target RoadmapBench implementation against the roadmap and repository tests.
---

# RoadmapBench Verification

Use this skill before declaring `DONE`.

1. Re-read the roadmap and make a checklist of every target and compatibility
   requirement.
2. Inspect the integrated diff and confirm child work follows shared APIs and
   repository conventions.
3. Run focused repository tests for each target, then the broadest practical
   regression suite within the remaining time.
4. Treat failures as integration work: diagnose, fix, and rerun relevant tests.
5. Do not look for `/tests` or hidden verifier content. Final scoring tests are
   injected by the harness only after the agent run.

Report `DONE` only when the implementation and available validation evidence
cover the complete roadmap.
