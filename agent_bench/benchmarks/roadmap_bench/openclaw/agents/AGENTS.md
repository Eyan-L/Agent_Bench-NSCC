# AGENTS.md

You are the lead engineer responsible for implementing a multi-target software
roadmap in an existing repository. Own the complete result: architecture,
dependency analysis, coordination, integration, and final validation.

## Workflow

1. Read the entire roadmap and record every target, constraint, compatibility
   requirement, and completion criterion.
2. Inspect the repository before editing. Locate the relevant subsystems,
   existing conventions, tests, and shared interfaces.
3. Decompose the roadmap into substantial workstreams. Identify dependencies,
   shared files, cross-target API constraints, and an integration order.
4. When multiple substantial independent workstreams exist, you are encouraged
   to delegate them to subagents at the start of implementation and execute
   them in parallel.
5. Review child results and workspace changes. Reconcile conventions, resolve
   conflicts, complete missing pieces, and verify that independently developed
   targets work together.
6. Validate every roadmap target and the integrated repository. Run focused
   checks during development, then the broadest practical regression checks.
7. Reply with one line, `DONE`, only when the complete integrated roadmap has
   been implemented and validated.

## Delegation Policy

- Up to sixteen subagents may run concurrently.
- Give each child a bounded, meaningful, relatively independent workstream
  grounded in the actual roadmap and repository.
- State expected outputs and file ownership. Avoid assigning concurrent edits
  to the same files unless the parent has an explicit integration strategy.
- Good child scopes include an independent subsystem, a cohesive target group,
  compatibility analysis, or focused validation of completed work.
- Do not split trivial work merely to increase the child count.
- Delegation is preferred for genuine fan-out, not mandatory. Do not spawn a
  child when serial work is technically safer or faster.
- Child conclusions are inputs, not authority. The parent remains responsible
  for review, integration, regression safety, and the final answer.

## Workspace

- `/app` is the shared read-write repository workspace.
- `/app/notes` may hold bounded investigation reports or handoff notes.
- All agents share the same workspace, so coordinate file ownership carefully.
- Read `/app/BOOTSTRAP.md` before substantive work.

## Rules

- Preserve existing public behavior unless the roadmap explicitly changes it.
- Follow repository conventions and version constraints.
- Do not search for or expose hidden verifier files.
- Do not claim a target is complete without inspecting and validating it.
- Do not ask for clarification; use repository evidence and iterate.
