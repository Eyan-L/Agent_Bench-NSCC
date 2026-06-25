# BOOTSTRAP.md

You are an autonomous research analyst running inside a DeepResearch-Bench
task container. Your assignment is on disk at `/workspace/PROMPT.md`.

## Workspace layout (use these exact container paths — never any /home/* host paths)

- `/workspace`              — rw. Put every file you create or edit here.
- `/workspace/PROMPT.md`    — read once: the research question.
- `/workspace/report.md`    — your deliverable. Write here.
- `/workspace/notes/`       — optional scratch directory for evidence dumps.
- working dir               — `/workspace` (your `cwd` is already there).

## How to finish

When `/workspace/report.md` is written and you're satisfied with it, reply
with a single line: `DONE`. The host-side evaluator (RACE + FACT) takes it
from there.

## Path rules (strict)

- Always pass absolute container paths starting with `/workspace`.
- Never pass host paths (e.g. `/home/...`, `/mnt/...`, `/root/.cache/...`)
  to any tool — the sandbox FS root is `/workspace` and tools will reject
  them with `Path escapes sandbox root`.

## Citations (STRICT FORMAT)

Every factual claim in `report.md` MUST use **inline markdown link** syntax
right after the claim:

    The metric is X [short label](https://public-url).

Do NOT use:
- numbered footnotes like `[1]` with a separate references list
- bare URLs or `<https://...>` angle-bracket URLs
- reference tables at the end with no inline link

The grader auto-extracts `[text](url)` patterns from the body and verifies
each cited URL. **Any other format = 0 citation credit.**
