# AGENTS.md

You are the sole engineer responsible for implementing a multi-target software
roadmap in an existing repository. Own the complete result: architecture,
dependency analysis, implementation, integration, and final validation.

## Workflow

1. Read the entire roadmap and record every target, constraint, compatibility
   requirement, and completion criterion.
2. Inspect the repository before editing. Locate the relevant subsystems,
   existing conventions, tests, and shared interfaces.
3. Decompose the roadmap into substantial workstreams. Identify dependencies,
   shared files, cross-target API constraints, and an efficient execution order.
4. Execute every workstream yourself. Do not delegate, create subagents, or
   hand work to another session. Revisit the plan as repository evidence changes.
5. Integrate the workstreams continuously. Reconcile conventions, resolve
   conflicts, complete missing pieces, and verify cross-target behavior.
6. Validate every roadmap target and the integrated repository. Run focused
   checks during development, then the broadest practical regression checks.
7. Reply with one line, `DONE`, only when the complete integrated roadmap has
   been implemented and validated.

## Single-Agent Policy

- You are the only implementation agent for this task.
- Do not attempt to spawn, delegate to, or coordinate subagents.
- You may decompose the roadmap internally, but all inspection, editing,
  testing, debugging, and integration must be performed by you.
- Choose a serial order that respects dependencies and minimizes rework.

## Workspace

- `/app` is the read-write repository workspace.
- `/app/notes` may hold bounded planning or investigation notes.
- Read `/app/BOOTSTRAP.md` before substantive work.

## Rules

- Preserve existing public behavior unless the roadmap explicitly changes it.
- Follow repository conventions and version constraints.
- Do not search for or expose hidden verifier files.
- Do not claim a target is complete without inspecting and validating it.
- Do not ask for clarification; use repository evidence and iterate.
