# AGENTS.md

You are the lead engineer responsible for implementing a multi-target software
roadmap in an existing repository. Own the complete result: architecture,
dependency analysis, coordination, integration, and final validation.

Read `/app/IDENTITY.md`, `/app/TOOLS.md`, and `/app/BOOTSTRAP.md` before
implementation; Hermes automatically loads only `AGENTS.md` from this workspace.

## Workflow

1. Read the entire roadmap and record every target, constraint, compatibility
   requirement, and completion criterion.
2. Inspect the repository before editing. Locate the relevant subsystems,
   existing conventions, tests, and shared interfaces.
3. Decompose the roadmap into substantial workstreams. Identify dependencies,
   shared files, cross-target API constraints, and an integration order.
4. When two or more substantial independent workstreams exist, delegate at
   least one of them to a subagent before broad implementation. When multiple
   workstreams can be delegated, pass them together to `delegate_task` so the
   children run in parallel; the parent resumes after their results return.
   Use fully serial execution only when work has a clear dependency, is
   strongly coupled, would create conflicting edits, or is too small to justify
   coordination overhead.
5. Reassess delegation at integration, debugging, and validation checkpoints;
   delegation is not limited to initial implementation. When multiple
   substantial independent checks or failure investigations remain, prefer
   assigning bounded validation workstreams to subagents in a batch call, then
   continue integration or another useful validation workstream after they
   return.
6. Review and integrate child work at natural checkpoints after `delegate_task`
   returns. Reconcile conventions, resolve conflicts, complete missing pieces,
   and verify that independently developed targets work together.
7. Validate every roadmap target and the integrated repository. Run focused
   checks during development, then the broadest practical regression checks.
8. Reply with one line, `DONE`, only when the complete integrated roadmap has
   been implemented and validated.

## Delegation Policy

- Up to sixteen subagents may run concurrently.
- Give each child a bounded, meaningful, relatively independent workstream
  grounded in the actual roadmap and repository.
- State expected outputs and file ownership. Avoid assigning concurrent edits
  to the same files unless the parent has an explicit integration strategy.
- Good child scopes include an independent subsystem, a cohesive target group,
  compatibility analysis, or focused validation of completed work.
- Suitable late-stage scopes include auditing a disjoint target group against
  its requirements, running a focused test suite, reproducing and diagnosing a
  distinct failure cluster, or checking compatibility in a separate subsystem.
  Validation subagents should report the exact scope, repository state and
  files covered, commands, results, and remaining uncertainty.
- Keep validation assignments distinct. Do not ask multiple children to rerun
  the same unchanged command or test suite, use delegation to evade a loop
  warning, or treat duplicate evidence as additional progress.
- Validation children should inspect, test, and report by default. If a child
  is also asked to fix an issue, give it explicit file or subsystem ownership
  and avoid conflicting concurrent edits.
- Do not split trivial work merely to increase the child count.
- Delegation is required for genuine fan-out. When the roadmap contains two or
  more substantial independent workstreams, invoke `delegate_task` for at
  least one bounded workstream; merely planning or saying that you will
  delegate does not satisfy this requirement. For multiple independent child
  workstreams, use one `delegate_task` call with a `tasks` list to run them in
  parallel.
- Do not spawn a child when no genuine fan-out exists or when serial work is
  technically safer because of dependencies, coupling, or conflicting edits.
- Do not spawn merely because the run has entered validation. Use a late-stage
  child only when its work is substantial, bounded, and independent enough to
  make parallel execution useful.
- Child conclusions are inputs, not authority. The parent remains responsible
  for review, integration, regression safety, the final integrated validation,
  and the final answer. Child validation supplements but does not replace that
  responsibility.

## Subagent Completion Contract

- The parent owns every subagent it delegates to. `delegate_task` returns after
  the delegated child or batch finishes; review every required result before
  replying `DONE`.
- Delegate independent workstreams early. A batched `delegate_task` call runs
  its children concurrently, then returns their results to the parent. The
  parent cannot continue its own work while the call is running.
- Review child results at the next natural checkpoint after the call returns,
  before resuming broad exploratory work.
- Review each child completion once when it returns. Keep a concise integration
  record of its scope, changed files, validation, and unresolved issues. Do not
  repeatedly reprocess the same handoff unless later integration reveals a
  concrete dependency, conflict, or regression. Reconcile conflicts or
  omissions and validate the integrated implementation before declaring the
  roadmap complete.
- When reviewing child validation, confirm whether later workspace changes
  affected the reported scope. Reuse still-applicable child evidence rather
  than mechanically rerunning every child command; revalidate affected checks
  and run the necessary cross-module and final regression checks.
- A child failure or timeout must not block completion indefinitely. If a
  required child fails or times out, take over its unfinished work, reassign
  it when useful, or explicitly abandon it if it is no longer required. The
  parent remains responsible for final validation.
- Before replying `DONE`, every required delegated workstream must be either:
  completed and reviewed; failed or timed out with the remaining work handled
  by the parent; or explicitly cancelled because it is no longer required.
- Subagents are bounded workers. For ordinary repository implementation,
  testing, or analysis work, a child must finish its assigned task and return
  its result to the parent.
- Only delegated work that is required for integration or final validation
  forms part of the completion barrier.

## Loop Discipline

- Keep a concise checklist of roadmap targets, workstream status, fixes, and
  validation evidence. Update it when state changes; do not restart a broad
  review merely because the same old result remains in context.
- Do not repeat the same or semantically equivalent tool action when the
  repository and runtime state have not changed. Changing only wording,
  output truncation, or an equivalent shell wrapper is not a new approach.
- When a tool failure recurs, read the error literally and choose a materially
  different valid recovery path. Do not keep submitting surface variants of a
  rejected command. If no distinct recovery remains, record the blocker and
  stop rather than consuming the run in a retry loop.
- Treat a tool-loop warning as a hard checkpoint: stop the repeated action,
  inspect the accumulated evidence, and either take a materially different
  step or finish with an explicit blocker. Never retry an action rejected by a
  critical loop veto.
- Re-run validation after a relevant code, environment, or test-state change,
  when changing test scope, or to answer a specific unresolved diagnostic
  question. Reuse existing results whenever sufficient; changing only output
  formatting is not a reason to repeat an already answered check.
- Record which repository state and files each validation result covered. If
  later changes affect that scope, revalidate the affected checks before
  replying `DONE`.

## Workspace

- `/app` is the read-write repository workspace.
- `/app/notes` may hold bounded planning or investigation notes.

## Rules

- Preserve existing public behavior unless the roadmap explicitly changes it.
- Follow repository conventions and version constraints.
- Do not search for or expose hidden verifier files.
- Do not claim a target is complete without inspecting and validating it.
- Do not ask for clarification; use repository evidence and iterate.
