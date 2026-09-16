# TOOLS.md

Repository tools operate on the shared `/app` workspace. This is the
single-agent control condition: perform all repository inspection, editing,
testing, debugging, and integration in the parent session. Do not delegate work
or attempt to create subagents.

Hidden `/tests` are not available during implementation. Use repository tests,
targeted checks, and project-native validation commands while developing; the
harness injects authoritative verifier tests only during final scoring.
