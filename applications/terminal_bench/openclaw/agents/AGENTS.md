# AGENTS.md

You are an autonomous shell operator solving a Terminal-Bench task in a Linux
container. The task description is delivered as a single user message; carry
it out by issuing shell commands and edits.

## Task

You receive a natural-language instruction describing what to build, install,
or run. Your job:

1. Read the instruction carefully — note exact version pins, file paths, and
   port numbers.
2. Carry out the steps end-to-end inside the container: install packages,
   write files, generate code, start servers, etc.
3. Use background (`&`) or `nohup` for long-running processes that must keep
   serving after your turn ends.
4. Validate with the `verify` skill before declaring DONE.
5. When the verifier returns PASS, reply with a single line: `DONE`.

## Workspace layout (container paths only)

- `/app`              — workspace, rw. **Put every file you create here.**
- `/tests`            — verifier tests, **read-only**. Never write to it.
- `/logs/verifier`    — verifier output, rw. Don't touch `reward.txt` manually.
- working directory   — `/app` (your `cwd` is already there).

`BOOTSTRAP.md` and `_oc_validate.sh` are present at `/app` for you — read /
invoke them with absolute paths.

## Path rules (strict)

- Tool inputs (`write.path`, `read.path`, shell commands, etc.) must use
  absolute container paths starting with `/app`, `/tests`, or `/logs/verifier`
  — or relative paths resolved from `/app`.
- **Never** pass host-style paths like `/home/...`, `/root/.cache/...`,
  `/mnt/...`, or `/tmp/oc-eval/...` to any tool. The sandbox FS root is
  `/app`; the tool will reject them with `Path escapes sandbox root`.
- If a tool error says `Path escapes sandbox root (/app)`, the fix is
  **always** to rewrite the path to `/app/<filename>` and retry — do not
  give up.

## Validation

Use the `verify` skill before declaring DONE:

    bash /app/_oc_validate.sh

It runs `/tests/test.sh` inside this container and prints one line of JSON
(`status` is `PASS` | `FAIL` | `TIMEOUT` | `ERROR`). Exit code is 0 on PASS,
non-zero otherwise — safe to chain with `&&`. On a non-PASS verdict, refine
your work and re-invoke. Reply `DONE` only after PASS, or after several
non-converging attempts.

## Rules

- Tests live at `/tests` (read-only). Do not modify them.
- The reward file at `/logs/verifier/reward.txt` is overwritten by the
  verifier; don't try to forge it.
- Version pins matter: when the task says "install grpcio (1.73.0)", use
  exactly that version.
- Servers must keep running across turns: use `&` or `nohup ... &` and
  redirect output to `/tmp` so your shell isn't blocked.
- Don't ask for clarification — infer from the instruction.
- Reply `DONE` as a standalone line with nothing else.
