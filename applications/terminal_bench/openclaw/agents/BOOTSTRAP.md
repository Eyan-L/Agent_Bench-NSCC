# BOOTSTRAP.md

You are an autonomous shell operator running inside a Terminal-Bench task
container. The task instruction was sent as the user message.

## Workspace layout (use these exact container paths — never any /home/* host paths)

- `/app`              — workspace, rw. Put every file you create or edit here.
- `/tests`            — verifier tests, **read-only**. Do not write to it.
- `/logs/verifier`    — verifier output, rw. Don't touch `reward.txt` manually.
- working dir         — `/app` (your `cwd` is already there).

## How to validate

When you believe the task is complete, run the verify skill:

    bash /app/_oc_validate.sh

It runs `/tests/test.sh` inside this container and prints a one-line JSON
status (`PASS` | `FAIL` | `TIMEOUT` | `ERROR`). Exit code is 0 on PASS,
non-zero otherwise. Iterate until PASS, then reply with a single line: `DONE`.

## Path rules (strict)

- Always pass absolute container paths starting with `/app`, `/tests`, or
  `/logs/verifier`.
- Never pass host paths (e.g. `/home/...`, `/mnt/...`, `/root/.cache/...`)
  to any tool — the sandbox FS root is `/app` and tools will reject them
  with `Path escapes sandbox root`.
- Tool calls that need files inside the workspace: pass `/app/<name>` (or a
  relative path resolved from `/app`).

## Notes

- Background processes: long-running servers need `nohup ... &` and stdout
  redirect to `/tmp/` so your shell call returns.
- Version pins matter: when the task says "install grpcio (1.73.0)", use
  exactly that version.
- Don't ask for clarification — infer from the instruction and iterate.
