# Exec bridge — routing OpenClaw tool calls through a custom socket

This document describes the **exec bridge** architecture that intercepts
OpenClaw's tool execution layer and routes it through a unix-domain socket
to a harness-side server. The server then dispatches into whatever backend
the harness owns (docker exec, singularity exec, …). OC remains pure
"generate the command"; the harness owns "actually run the command and
return a result."

This is the result of the 2026-05-11 session that took phase0 from
*infra-clean / verdict FAIL* to *infra-clean / verdict PASS (15/15)* on
the astropy SWE-bench instance.

## Why we built this

The shipped OpenClaw image runs tool calls through one of three internal
hosts:

- `tools.exec.host = "sandbox"` — OC spawns its own sandbox container via
  the bind-mounted docker socket and `docker exec`s into it.
- `tools.exec.host = "gateway"` — OC runs commands in its own container.
- `tools.exec.host = "node"` — OC talks to a remote node companion over
  WebSocket through the gateway.

For SWE-bench, "sandbox" is the only viable mode (the agent must see the
per-instance conda env, not OC's Debian). But that mode entangles two
concerns:

1. **Lifecycle**: OC's `ensureSandboxContainer` decides when to start /
   reuse / tear down the task container.
2. **Execution**: OC's exec runtime calls `docker exec` directly.

That entanglement broke the harness three different ways:

| Symptom | Root cause |
|---|---|
| stale sandbox from a prior run gets reused with broken bind mounts | OC's `(scope, sessionKey, configHash)` lookup matches across runs |
| can't reuse for singularity, ssh, anything else | OC's docker-CLI calls are hard-coded |
| OC's fs-bridge (read/write/edit) crashes on `Path escapes sandbox mount root` | OC's path-safety check expects host filesystem == sandbox.workspaceDir |

The fix: **make the harness own both lifecycle AND execution.** OC just
emits requests; the bridge fulfills them. This document records the
chosen design and the alternatives we rejected, so a future change
doesn't accidentally re-entangle them.

## What ships

| Component | Where | Purpose |
|---|---|---|
| `bridge.py` (inside OC container) | `orchestration/dispatcher/openclaw_remote/bridge.py` | Client. OC `spawn`s it; it speaks the socket protocol to the host server. |
| `server.py` (on host) | `orchestration/dispatcher/openclaw_remote/server.py` | Server. Listens on a unix socket, validates HMAC, dispatches into a backend. |
| `OpenClawRemoteDispatcher` | `orchestration/dispatcher/openclaw_remote/__init__.py` | `Dispatcher` impl that owns the per-run lifecycle of the two scripts above (token, socket, server subprocess, agent-container injection, cleanup). |
| `openclaw-external-exec-bridge.patch` | `harnesses/openclaw/patch/` | Patches OC's `bash-tools.exec-runtime` bundle so the `exec` tool's spawn argv points at `bridge.py` when `OPENCLAW_EXEC_BRIDGE` env is set. |
| `openclaw-fs-bridge.patch` | `harnesses/openclaw/patch/` | Patches OC's `sandbox` bundle so the **fs-bridge** (`read`/`write`/`edit`/`stat`/etc.) ALSO routes through `bridge.py`, and the host-path safety check is skipped. |
| `applications/swebench/app.py` | (modified) | Owns the lifecycle: pre-starts the task env container, spawns the server, mounts socket + bridge into OC, sets env vars, tears down. |
| `applications/swebench/conditions/openclaw_agent.py` | (modified) | After the agent finishes, the SWE-bench subclass's `_collect_artifacts` harvests the patch by running `git diff --cached` on the **host** (not in any container) with scaffold-path exclusion. Base loop lives in `applications/conditions/openclaw_agent.py`. |

## Run-time topology

```
                              host
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        │   driver.py (orchestration.runner)                      │
        │     │                                                   │
        │     ├─► prepare_environment (applications/swebench)     │
        │     │     1. prime ws_host from task image (docker cp)  │
        │     │     2. start task container ← swebench image      │
        │     │           -v ws_host:/testbed:rw                  │
        │     │           -e PATH=...envs/testbed/bin:...         │
        │     │     3. spawn server.py subprocess                 │
        │     │           --socket state/exec-bridge.sock         │
        │     │           --backend docker --target <task>        │
        │     │     4. start OC container ← agent-bench/openclaw  │
        │     │           -v ws_host:/testbed:rw                  │
        │     │           -v state_dir:/root/.openclaw            │ (the socket lives here)
        │     │           -v bridge.py:/usr/local/bin/oc-exec-bridge:ro
        │     │           -e OPENCLAW_EXEC_BRIDGE=/usr/local/bin/oc-exec-bridge
        │     │           -e OPENCLAW_EXEC_BRIDGE_SOCKET=/root/.openclaw/exec-bridge.sock
        │     │           -e OPENCLAW_EXEC_BRIDGE_TOKEN=<hex>     │
        │     │                                                   │
        │     ├─► condition.run                                   │
        │     │     docker exec oc-<run> openclaw agent ...       │
        │     │                                                   │
        │     │     ┌─── OC container ──────────────────────┐     │
        │     │     │                                       │     │
        │     │     │  patched OC runtime                   │     │
        │     │     │   tool: exec   ──────┐                │     │
        │     │     │   tool: read         │                │     │
        │     │     │   tool: write        │ spawn bridge.py│     │
        │     │     │   tool: edit         │ with argv/stdin│     │
        │     │     │                      ▼                │     │
        │     │     │              ┌───────────────┐        │     │
        │     │     │              │   bridge.py   │        │     │
        │     │     │              └──────┬────────┘        │     │
        │     │     │                     │ unix socket     │     │
        │     │     │                     │ + HMAC          │     │
        │     │     └─────────────────────┼─────────────────┘     │
        │     │                           │                       │
        │     │                           ▼                       │
        │     │                  ┌───────────────────┐            │
        │     │                  │     server.py     │            │
        │     │                  └─────────┬─────────┘            │
        │     │                            │ docker exec          │
        │     │                            ▼ -w /testbed          │
        │     │     ┌───────── task container ───────────┐        │
        │     │     │  /opt/miniconda3/envs/testbed/...  │        │
        │     │     │  /testbed/  ◄── ws_host bind       │        │
        │     │     └────────────────────────────────────┘        │
        │     │                                                   │
        │     ├─► collect_patch (host-side git diff on ws_host)   │
        │     ├─► run_tests (swebench.harness.run_evaluation)     │
        │     └─► cleanup (rm both containers, network,           │
        │                  SIGTERM bridge server, rm workspace)   │
        └─────────────────────────────────────────────────────────┘
```

## Protocol

Modeled on `infra/exec-host.ts`'s `ExecHostRequest`/`ExecHostResponse` in the
OpenClaw source — we deliberately keep it shape-compatible in case
upstream ever generalizes their macOS-only socket-exec path to Linux.

### Wire

JSON Lines over a unix-domain stream socket. One request per connection;
one response; then close.

### Request (client → server)

```json
{
  "type": "exec",
  "id": "<uuid>",
  "nonce": "<hex>",
  "ts": <ms-since-epoch>,
  "hmac": "<sha256-hex>",
  "requestJson": "<json-of-payload>"
}
```

HMAC: `HMAC-SHA256(token, "<nonce>:<ts>:<requestJson>")`. The token is
generated per-run by the harness (24 random bytes hex), bind-mounted as
a file into the OC container, and read into env `OPENCLAW_EXEC_BRIDGE_TOKEN`.

`requestJson` payload:

```json
{
  "command": ["argv0", "argv1", ...],
  "cwd": "/testbed",
  "env": {"KEY": "VAL", ...},
  "timeoutMs": null | <int>,
  "agentId": "swebench",
  "sessionKey": "agent:swebench:...",
  "stdinB64": null | "<base64-bytes>"
}
```

The client decides framing of `command`:
- `exec` tool sends `["/bin/sh", "-lc", "<shell-cmd>"]`.
- fs-bridge sends `["sh", "-c", "<plan-script>", "openclaw-sandbox-fs", ...args]`.

### Response (server → client)

```json
{
  "type": "exec-res",
  "ok": true,
  "payload": {
    "stdoutB64": "<base64>",
    "stderrB64": "<base64>",
    "stdout": "<utf-8 best-effort>",
    "stderr": "<utf-8 best-effort>",
    "exitCode": 0,
    "timedOut": false,
    "success": true
  }
}
```

Or on error:

```json
{
  "type": "exec-res",
  "ok": false,
  "error": {"code": "auth-failed", "message": "..."}
}
```

Stdin/stdout/stderr are base64 because file-read operations return binary
payloads that don't round-trip through JSON as UTF-8. Plain text fields
are kept for human / legacy consumers.

### Replay protection

Server keeps a sliding 10-minute set of seen nonces. Repeat nonces and
timestamps off by more than 5 minutes are rejected (`auth-failed`).

## OC-side patches

Two patches under `harnesses/openclaw/patch/`. Both are applied at OC
image build time (`harnesses/openclaw/docker/build.sh` → `apply_patch.py`).

### `openclaw-external-exec-bridge.patch`

Targets `dist/bash-tools.exec-runtime-*.js`. Inserts ~15 lines right at the
top of the sandbox branch of `spawnSpec`:

```javascript
if (opts.sandbox) {
    if (process.env.OPENCLAW_EXEC_BRIDGE) {
        return {
            mode: "child",
            argv: [
                process.env.OPENCLAW_EXEC_BRIDGE,
                "--workdir", opts.containerWorkdir ?? opts.sandbox.containerWorkdir,
                "--env-json", JSON.stringify(shellRuntimeEnv),
                "--command", execCommand,
                ...(timeoutMs ? ["--timeout-ms", String(timeoutMs)] : []),
            ],
            env: process.env,
            stdinMode: opts.usePty ? "pipe-open" : "pipe-closed"
        };
    }
    // ... original code path
}
```

This is the only place we change the agent's bash `exec` tool flow. When
the env var is unset, OC behaves exactly as upstream — no behavior change.

### `openclaw-fs-bridge.patch`

Targets `dist/sandbox-*.js`. Three hunks:

1. **Inject a helper** `__ocFsBridgeRun` near the top of the bundle that
   spawns bridge.py with `--argv-json` and pipes back stdout/stderr/code.
2. **Patch `runCommand`** (the chokepoint for every fs-bridge operation —
   read/write/stat/list/mkdir/unlink): when the bridge env is set, route
   through the helper instead of OC's built-in docker-exec call.
3. **Patch `runCheckedCommand`** to skip the `pathGuard.assertPathChecks`
   when bridge env is set. That check resolves /testbed → host path and
   rejects if the host path is outside `sandbox.workspaceDir` — a
   reasonable check for OC's macOS-desktop case, but wrong for our
   container-in-container setup where the resolved host path
   (`/home/hongrui/.cache/oc-eval/workspaces/…`) is correct but doesn't
   match what OC's mount tracker computed.
4. **Patch `readPinnedFile`** which is special-cased to use
   `fs.openSync(opened.fd)` against the resolved host path — won't work
   from inside the OC container, where only `/testbed` is mounted.
   Replace with a `cat "$1"` invocation through the bridge.

## Lifecycle (where each component starts/stops)

| Step | Action | Owner | When |
|---|---|---|---|
| 0 | (vLLM endpoint up — handled separately by Makefile `_ensure_serve`) | host process | before run |
| 1 | prime ws_host from task image (`docker cp` from scratch container) | `prepare_environment` | per-run start |
| 2 | start task env container (named `oc-sbx-<rid>-<slug>`) | `prepare_environment` | per-run start |
| 3 | spawn server.py subprocess | `prepare_environment` | per-run start |
| 4 | start OC container (named `oc-<rid>`) | `prepare_environment` | per-run start |
| 5 | run `openclaw agent ...` (via `docker exec` into OC) | condition.run | inner loop |
| 6 | host-side `git diff` on ws_host → candidate.patch | condition.run end | after DONE |
| 7 | `swebench.harness.run_evaluation` | harness.run_tests | after patch |
| 8 | SIGTERM server.py (`os._exit(0)` in handler, no socketserver.shutdown deadlock) | cleanup | finally |
| 9 | `docker rm -f` both OC + task container, `docker network rm`, rmtree ws_host + state_dir | cleanup | finally |

## Design decisions and rejected alternatives

### "Why not just patch the docker CLI inside OC?"

The first attempt was a PATH shim — drop a `docker` shell script into OC
container, let OC's `spawn("docker", ...)` resolve to it via PATH. Zero
OC patches, works for SWE-bench docker case.

**Rejected** because for singularity (future), OC would still issue
`docker run`, `docker inspect`, `docker network create` etc. as part of
its sandbox lifecycle. To make those work over singularity, the shim
would have to fake a substantial portion of the docker CLI. The bridge
approach instead **lets the harness own lifecycle entirely** — OC's
docker-side lifecycle calls (`ensureSandboxContainer`) still run, but
they find our pre-started container and reuse it; meanwhile the bridge
intercepts only the tool-call execution path that we want to redirect.

### "Why not reuse OC's existing `exec-host.ts` socket protocol?"

OpenClaw already ships a socket-based exec host
(`requestExecHostViaSocket` in `infra/exec-host.ts`). It's
**well-designed, structured, signed** — exactly what we want. We even
borrowed its wire format.

**Rejected as the trigger mechanism** because the socket path is gated
by `process.platform === "darwin" && OPENCLAW_NODE_EXEC_HOST === "app"`
in `node-host/invoke.ts:46-49` and only invoked from the
`node-host/invoke-system-run.ts` flow. To use it on Linux we'd need:

1. A patch removing the darwin gate (1 line, easy).
2. *Plus*: running OC in `tools.exec.host = "node"` mode, which means
   spinning up a **second OC process** as a node companion connected to
   the gateway via WebSocket. The socket-call only fires in the node
   companion's `system.run` path.

The second OC process is real architectural overhead for what's already
a simple "spawn a child process and capture stdout" pattern. So we use
OC's protocol design but invoke it directly from `bash-tools.exec-runtime`'s
spawnSpec (one bundle patch, no extra processes).

### "Why not a new `tools.exec.host = 'external'` type?"

That would have meant:

- Patching the `ExecHost` union type in `infra/exec-approvals.ts`.
- Patching `normalizeExecHost`.
- Patching `bash-tools.exec.ts` host dispatch (the host switch is
  inspected in ~6 places — line 308, 318, 321, 336, 338, 365, 384, 394,
  402, 427).
- Adding a new `bash-tools.exec-host-external.ts` mirroring the
  ~400-line existing host implementations.

~50-100 LOC of patches across 4-5 files, far more surface area than the
single-site env-var trigger.

### "Why env-var trigger and not openclaw.json config field?"

Both work. Env var is simpler — no config-schema change, no patching
config parser. The harness already sets one-shot env vars on the OC
container (`OPENCLAW_GATEWAY_TOKEN`, `OPENCLAW_RUN_ID`); this fits.

### "Why patch `runCommand` not the entry methods (`readFile`, `writeFile`, …)?"

There are 6 fs-bridge entry methods (readFile, writeFile, mkdirp, remove,
rename, stat). Patching all 6 would mean 6× the diff. They all funnel
through `runCommand`/`runCheckedCommand` for the actual docker-exec
call. Patching there covers everything in one spot.

`readPinnedFile` is a special case — `readFile` reads files via Node's
own `fs.openSync` (not docker exec at all). That's a separate patch site.

### "Why run `git diff` on the host instead of inside the OC container?"

Original design: `docker exec oc-<run> git -C /testbed diff --cached`. Two
distinct failures hit it:

1. **Dubious-ownership**: OC runs as `--user 0` (root), but `/testbed`
   files in the bind mount belong to host UID `hongrui` (1002). Git
   triggers its safe-directory protection and returns empty.
2. **Permission**: after `-c safe.directory=*` bypasses (1), `git add`
   tries to write to `.git/objects/`. Despite OC running as root,
   Docker's bind-mount permissions deny writes to the host-owned `.git`.

Workarounds (chown, run as host UID, etc.) all add complexity. Simpler:
the harness driver is itself running as host user `hongrui`, and the
workspace IS at a host path it owns. So run `git` on the host directly.
Removes a whole class of bind-mount-permission edge cases.

### "Why `os._exit(0)` from SIGTERM handler instead of `server.shutdown()`?"

Python's `socketserver.shutdown()` waits for `serve_forever()` to
return. Calling it from a same-thread signal handler **deadlocks** —
`shutdown()` blocks, and `serve_forever()` can't progress while the
signal handler is on the stack. This was caught when the harness left
6 leaked server processes across 6 test runs (cleanup tried to SIGTERM,
handler ran, server stayed alive).

The server has no in-memory state to flush (the socket file we unlink
manually, the registry is per-run and harness will recreate). `os._exit(0)`
is the right primitive.

### "Why pre-pend `/opt/miniconda3/envs/testbed/bin` to task container PATH?"

The SWE-bench image's default `PATH=/opt/miniconda3/bin:...` points at
the **base** conda env, which only has `python` itself, no astropy/numpy/
packaging/etc. The pinned-deps env is at `/opt/miniconda3/envs/testbed/`
and only gets activated by `~/.bashrc` — but only for interactive
shells. `docker exec bash -lc` is a login shell, not interactive, so
`.bashrc` is not sourced and the conda env stays un-activated.

Pre-pending the testbed env's `bin` to PATH at task-container-`docker run`
time makes `python` / `pip` always resolve to the pre-built env without
any `conda activate` ceremony.

There's a related subtle bug: server.py used to forward OC's `req.env.PATH`
to `docker exec -e PATH=...`, which overrode the task container's PATH
back to OC's runtime PATH (which doesn't have testbed). Fix: server.py
explicitly drops `PATH` from forwarded env (`docker_backend` in `server.py`).

### "Why trust `report.json` even when `run_evaluation` exits non-zero?"

SWE-bench's `swebench.harness.run_evaluation` sometimes returns rc=1
**after** successfully grading the instance — typically a container
teardown race where its `docker rm` collides with a concurrent docker
operation and emits `docker.errors.NotFound`. The grading was already
done; the report.json is on disk with `resolved: true`. We trust the
report.json when present, log the non-zero exit as a warning.

## How to extend

### Add a singularity backend

In `server.py` the `singularity_backend` stub is already in place. The
harness wiring change is in `applications/swebench/app.py` where
`prepare_environment` spawns the server:

```python
bridge_proc = subprocess.Popen([
    "python3", server_script,
    "--socket", str(bridge_socket_host),
    "--token-file", str(token_file),
    "--backend", "singularity",  # ← was "docker"
    "--target", singularity_instance_name,  # ← was task_container_name
])
```

And `prepare_environment` also needs to **start the Singularity instance**
in place of the docker task container (`singularity instance start
swebench.sif <instance-name>` with the right binds).

The bridge.py side, the OC patches, the agent prompts, the verify skill
— all unchanged. That's the point.

### Bypass the bridge (debug / ablation)

Unset `OPENCLAW_EXEC_BRIDGE` on the OC container. OC reverts to its
upstream behavior: `docker exec` into a sandbox container it manages
itself. The harness's pre-started task container will be reused via
OC's name-and-hash lookup (because we still pass it the right
`containerPrefix` and scope), but lifecycle is now OC's problem and
fs-bridge will fail with path-safety errors.

### Add per-call env injection (e.g. `LD_PRELOAD=...`)

Either:
- Set it on the **task container** at `docker run` (sticky for the run).
- Or pass it via `req.env` and the bridge will forward it as
  `docker exec -e LD_PRELOAD=...`. (Except `PATH`, which the server
  drops on purpose.)

## Verification status (2026-05-11)

phase0 = `astropy__astropy-12907`:

| Step | Result |
|---|---|
| OC + task + bridge come up | ✓ |
| `python` resolves to testbed env, `import astropy` works | ✓ |
| `read`/`write`/`edit` tools route through bridge | ✓ |
| `exec` tool routes through bridge | ✓ |
| Agent self-validates with `pytest` before DONE | ✓ |
| `git diff` extracts a clean patch (no .openclaw scaffold) | ✓ |
| `swebench.harness.run_evaluation` grading | resolved: true, 15/15 |
| cleanup leaves zero containers, zero networks, zero processes, zero workspace | ✓ |
