# Adding a new frontend

A *frontend* is the agent runtime — the image plus the orchestrator that
actually drives the LLM through tool calls. The only frontend currently
implemented is OpenClaw (`frontend/openclaw/`). This guide covers swapping in
or adding alongside another (e.g. SWE-Agent, claude-code, a custom CLI).

## What "frontend" includes

```
frontend/<name>/
├── apply_patch.py        # if the runtime needs binary patches at deploy time
├── configure.sh          # any setup / dep checks
├── gateway_up.sh         # bring the frontend up for one run
├── gateway_down.sh       # tear it down
├── patch/                # source patches (vendored)
│   └── *.patch
└── state/                # runtime config (mounted into the container)
    └── ...
```

Two distinct concerns:

1. **Image plumbing** — pulling/building the SIF, applying patches, ensuring
   the binary is on PATH inside the container. Lives in `frontend/<name>/`.
2. **Per-run lifecycle** — start the gateway/agent process, hand a token+port
   back to the harness, tear it down. Surfaced by `gateway_up.sh` /
   `gateway_down.sh`.

## 1. Image plumbing

For each frontend, decide:

- **Where does the binary live?** OpenClaw uses
  `$SCRATCH/sif_images/oc_runtime.sif` pulled from `docker://alpine/openclaw:main`.
  PBS handles the pull (see `scripts/execution_pbs/run_benchmark.pbs:STAGE_OC_CHECK`).
- **Does it need patching?** OpenClaw applies four patches to its bundled JS
  (see `frontend/openclaw/patch/MANIFEST.json`). The patch overlay is built
  by `scripts/execution_pbs/build_oc_overlay.sh` once per host.
- **What state does it read?** OpenClaw reads `state/openclaw.json` (gateway
  config + plugin enable/disable + agent definitions) and per-agent prompts
  (`state/agents/swebench/{AGENTS,SOUL,...}.md`).

If your frontend doesn't need patching, the layout simplifies to just
`gateway_up.sh` + `gateway_down.sh` + `state/`.

## 2. The gateway lifecycle scripts

The harness invokes these from `eval/benchmarks/<bench>/harness.py:prepare_environment`:

```bash
# gateway_up.sh — must be idempotent
#   $1: container_id  (or empty for non-container backends)
#   $2: port          (free port the harness picked)
#   $3: token         (random secret — write into $WORKSPACE/.your_frontend_env)
#   $4: run_id        (for OTel resource attrs)
#
# Behavior:
#   - Start the agent runtime in the background
#   - Block until it's ready (poll a readiness signal)
#   - Persist URL + token to /tmp/<frontend>_env inside the container so
#     subsequent `docker exec <frontend> agent ...` invocations can source it
#   - exit 0 on ready, non-zero with tail-of-log on failure
```

```bash
# gateway_down.sh — also idempotent; called from finally
#   $1: container_id
#
# Behavior:
#   - SIGTERM the gateway (give it time to flush — OTel exporters need this)
#   - SIGKILL after grace period
#   - Free the port
#   - Never raise — if there's nothing to stop, exit 0
```

Look at `frontend/openclaw/gateway_up.sh` for a working reference. Key points:

- **Token + port via env vars**, never on the command line — they show up in
  `ps` otherwise.
- **`OTEL_RESOURCE_ATTRIBUTES=oc.run_id=$RUN_ID`** so concurrent runs don't
  cross-contaminate the shared trace file.
- **Readiness via log grep** (`[gateway] ready`) is simpler and more reliable
  than HTTP polling for processes that take 1-2 minutes to load plugins.

## 3. Hook into the harness

The harness is the seam — it doesn't know about specific frontends, just calls
the lifecycle scripts. SWE-bench's `eval/benchmarks/swebench/harness.py`
hard-codes OpenClaw via `frontend/openclaw/gateway_up.sh`. To make it
pluggable:

```python
# in harness.prepare_environment
frontend = os.environ.get("OC_FRONTEND", "openclaw")
gateway_up = f"frontend/{frontend}/gateway_up.sh"
subprocess.run([gateway_up, container_id, str(port), token, run_id], check=True)
# ... return env_info with gateway_port/token/url ...
```

Then add a corresponding `gateway_down.sh` invocation in `cleanup`.

If you only ever use one frontend, skip the env-var indirection — the harness
can just call `frontend/<your_frontend>/gateway_up.sh` directly.

## 4. Tell conditions about it

Conditions invoke the agent through the gateway. The current pattern (see
`eval/conditions/code_agent/openclaw_agent.py:_invoke_agent`):

```python
# Inside the container, source the env file the gateway_up.sh script wrote
docker exec $cid bash -c "source /tmp/oc_gateway_env && \
    openclaw agent --json --message ..."
```

For your frontend:

1. Pick an in-container env file path (`/tmp/<frontend>_env`) and have
   `gateway_up.sh` write `$URL` and `$TOKEN` to it.
2. In your condition, source it before invoking the CLI.
3. Parse the CLI's JSON output into the per-step trace shape.

## 5. Span / telemetry parity

OpenClaw's diagnostics-otel plugin emits `openclaw.model.usage` and
`openclaw.tool.call` spans. The condition's `_snapshot_otel_spans` slices
those into `spans.jsonl`, with synth fallbacks (`_validate_bridge` does the
self-validation case; `_maybe_synthesize_*` fills missing per-call spans
when patches don't fire).

For a new frontend, mirror the schema so the analysis layer doesn't need to
branch:

| Span name | Required attrs |
|---|---|
| `openclaw.model.usage` | `openclaw.tokens.{input,output,cache_read,cache_write,total}`, `openclaw.model`, `openclaw.provider`, `openclaw.channel` |
| `openclaw.tool.call`   | `openclaw.tool.name`, `openclaw.tool.callId`, `openclaw.tool.isError`, `openclaw.runId`, `openclaw.sessionId` |

If your frontend doesn't natively emit OTel, write the spans directly into
`<run_dir>/spans.jsonl` from the condition (the TB ReAct condition does this
via `eval/conditions/code_agent/terminal_bench/react.py:_emit_span`).

## 6. Smoke test

```bash
# Manual: bring up the frontend, hit it once, tear down
RUN_DIR=$(mktemp -d)
bash frontend/your_frontend/gateway_up.sh "" 18900 "smoke-token" smoke-run
# ... your_frontend agent invocation ...
bash frontend/your_frontend/gateway_down.sh ""
```

Then run a 1-instance sweep with `OC_FRONTEND=your_frontend` set.

## Checklist

- [ ] `frontend/<name>/gateway_up.sh` and `gateway_down.sh`, both idempotent
- [ ] State files mounted (or copied) into the runtime container at the path
      the binary expects
- [ ] Env file (`/tmp/<frontend>_env`) holds URL + token for the condition to
      source
- [ ] `OTEL_RESOURCE_ATTRIBUTES=oc.run_id=$RUN_ID` set when launching the
      agent so spans demux per-run
- [ ] `gateway_down.sh` SIGTERMs first, gives ≥1s for OTel flush, then
      SIGKILLs
- [ ] Spans emitted in the canonical schema (or the condition writes them
      directly to `spans.jsonl`)
- [ ] Harness invokes the new frontend (either hard-coded or via `OC_FRONTEND`
      env var)
- [ ] One full run produces `verdict.json` + `step_trace.jsonl` + `spans.jsonl`
