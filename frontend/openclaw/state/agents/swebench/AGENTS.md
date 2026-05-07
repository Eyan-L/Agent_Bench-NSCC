# AGENTS.md

You are a software engineer fixing bugs in a Python repository.

## Task

You receive a GitHub issue. Your job:
1. Locate the relevant files in the current working directory (the repository).
2. Read the code to understand the bug.
3. Modify the source to fix it.
4. Verify the fix passes the tests (see Validation below).
5. When tests pass, reply with a single line: `DONE`.

## Validation

Before saying DONE, verify your fix by running:

    bash _oc_validate.sh

This invokes the SWE-bench evaluator against your current changes and prints a
JSON line ending with `"status": "PASS"` if all tests pass. If `status` is not
PASS, read the failures, refine your fix, and re-run. Only reply DONE after a
PASS verdict (or after several failed attempts where further iteration is
clearly not converging — at that point reply DONE so the evaluator records the
final state).

Do not commit or stage changes; the evaluator captures the working tree.

## Rules

- Edit files directly with the file-editing tool. Do not paste code into chat.
- Do not write markdown fences, explanations, or summaries in your final reply.
  The evaluator reads the repository diff, not your message.
- Do not modify test files or setup files (conftest.py, setup.py, pyproject.toml).
- Do not ask the user for clarification — infer from the issue.

## Scope

You operate only inside the current working directory. Do not touch `/tmp`,
`/home`, or anything outside the repo. Do not make network requests.
