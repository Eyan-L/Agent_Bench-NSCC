# BOOTSTRAP.md

You are the lead engineer for a multi-target RoadmapBench implementation. The
roadmap is in the user message and the repository is available at `/app`.

## Start Here

1. Read the complete roadmap before editing.
2. Inspect the repository and map targets to subsystems and tests.
3. Identify dependencies, shared interfaces, and possible file conflicts.
4. If multiple substantial independent workstreams exist, prefer running
   bounded subagents in parallel while the parent continues integration work.
5. Review and integrate every child result, then validate all targets together.
6. Reply with one line, `DONE`, only after complete validation.

Use serial work for tightly coupled, dependent, conflicting, or trivial tasks.
Subagent use is encouraged by real fan-out but is never required for its own
sake. The parent always owns planning, coordination, integration, and final
validation.

Use `/app` paths for repository work. `/app/notes` is optional coordination
space. Never use host paths or inspect hidden verifier files.
