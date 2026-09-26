# TOOLS.md

Repository tools operate on the shared `/app` workspace. Subagents share that
workspace with the parent, so assign non-overlapping file ownership and review
their changes before integration.

Hidden `/tests` are not available during implementation. Use repository tests,
targeted checks, and project-native validation commands while developing; the
harness injects authoritative verifier tests only during final scoring.
