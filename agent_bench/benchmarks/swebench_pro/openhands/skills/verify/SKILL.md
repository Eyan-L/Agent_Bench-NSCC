---
name: verify
description: Run a quick project-specific check for the current SWE-bench Pro workspace before stopping.
---

# verify

SWE-bench Pro tasks span several languages and repositories. From the repo root
(`/app` in the task container, `/workspace` in the OpenHands harness view),
inspect the project files and run the most relevant local check for the files
you changed.

The final score is computed by the harness after the agent stops using the
official SWE-bench Pro run script and parser for this instance.

