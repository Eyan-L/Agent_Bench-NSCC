---
name: verify
description: Run the SWE-bench instance's tests against the current workspace edits. Use before declaring DONE; on a non-PASS verdict, refine the fix and re-invoke.
---

# verify

Validates the current workspace by running pytest in the per-instance
python env that ships with the SWE-bench task image. Your shell tool already
runs inside that env (the OpenClaw sandbox is the SWE-bench instance image),
so this is just a `pytest` invocation.

## How to invoke

From the repo root (`/testbed`):

    python -m pytest --tb=short -q

Exit code is 0 if all collected tests pass. If you know which tests are
relevant to the issue (e.g. `astropy/modeling/tests/test_separable.py`),
target them directly to keep iteration fast:

    python -m pytest astropy/modeling/tests/test_separable.py -x --tb=short -q

For the SWE-bench-canonical scoring (FAIL_TO_PASS + PASS_TO_PASS lists),
the harness on the host will re-score using `swebench.harness.run_evaluation`
after you say DONE — so this in-loop check is for *iteration*, not the
authoritative grade.

Don't call before edits land; pytest will just re-run the failing baseline.
Only call once you believe the patch is complete.
