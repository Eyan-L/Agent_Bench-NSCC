# TOOLS.md

Repository tools operate on the shared `/app` workspace. Subagents share that
workspace with the parent, so assign non-overlapping file ownership and review
their changes before integration.

Hidden `/tests` are not available during implementation. Use repository tests,
targeted checks, and project-native validation commands while developing; the
harness injects authoritative verifier tests only during final scoring.

Codex exposes native `multi_agent_v1.spawn_agent`, `send_input`,
`wait_agent`, and `close_agent` tools. Spawning returns an agent ID without
waiting for completion. Continue independent parent work after spawning; use
`wait_agent` when the next step depends on a child result. These tools share
the repository workspace with the parent.
