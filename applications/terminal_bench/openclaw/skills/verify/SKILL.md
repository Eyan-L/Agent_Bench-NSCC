---
name: verify
description: Run /tests/test.sh inside the Terminal-Bench singularity instance and report PASS/FAIL based on /logs/verifier/reward.txt. Use before declaring the task DONE; on FAIL, refine the fix and re-invoke.
---

# verify

Validates the current workspace by running the Terminal-Bench task's test
script (`/tests/test.sh`) inside the singularity instance and reading the
verifier's reward (`/logs/verifier/reward.txt`). The test script can take
several minutes to install pytest deps and execute, so call this only when
you believe the fix is complete.

## How to invoke

The helper script `_oc_validate.sh` is dropped into your working directory
at run start. From there:

    bash _oc_validate.sh

Prints a single JSON object with `status` (`PASS` | `FAIL` | `TIMEOUT` |
`ERROR`), `test_pass_rate`, and `verifier_stdout_tail`. Exit code is 0 on
PASS, non-zero otherwise — safe to chain with `&&`.
