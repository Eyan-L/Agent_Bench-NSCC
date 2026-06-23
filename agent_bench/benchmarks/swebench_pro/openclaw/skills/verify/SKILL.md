---
name: verify
description: Run a quick project-specific check for the current SWE-bench Pro workspace before declaring DONE.
---

# verify

SWE-bench Pro tasks span several languages and repositories. There is no single
universal command like `pytest` for every instance.

Start from the repository root (`/app`) and inspect the project files for the
right local check, such as `npm test`, `go test ./...`, `pytest`, `make test`,
or the nearest targeted command for the files you changed.

The final benchmark score is computed by the harness after `DONE` using the
official SWE-bench Pro run script and parser for this instance.

