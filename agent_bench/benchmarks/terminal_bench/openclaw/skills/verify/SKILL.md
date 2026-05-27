---
name: verify
description: Run /tests/test.sh inside the Terminal-Bench task container and report PASS/FAIL based on /logs/verifier/reward.txt. Use before declaring the task DONE; on FAIL, refine the fix and re-invoke.
---

# verify

Validates the current workspace by running the Terminal-Bench task's test
script (`/tests/test.sh`) inside this container and reading the verifier's
reward (`/logs/verifier/reward.txt`). The test script can take a couple of
minutes to install pytest deps and execute, so call this only when you
believe the work is complete.

## How to invoke

The helper script `_oc_validate.sh` is present at `/app/_oc_validate.sh` (it
was dropped into your workspace at run start). Invoke with the absolute path:

    bash /app/_oc_validate.sh

It prints a single JSON object with `status` (`PASS` | `FAIL` | `TIMEOUT` |
`ERROR`), `verifier_rc`, `reward_raw`, and a `verifier_stdout_tail` for
context. Exit code is 0 on PASS, non-zero otherwise — safe to chain with `&&`.

## Path notes

Always use the absolute path `/app/_oc_validate.sh`. The sandbox FS root is
`/app`, so paths outside it (e.g. `/root/.openclaw/skills/...`) are rejected
with `Path escapes sandbox root`. The helper itself only references container
paths (`/tests/test.sh`, `/logs/verifier/reward.txt`).
