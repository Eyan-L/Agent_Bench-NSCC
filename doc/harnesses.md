# Adding a new harness

A *harness* is the agent runtime — the runtime image plus the orchestrator that
actually drives the LLM through tool calls. The only harness currently
implemented is OpenClaw (`agent_bench/harnesses/openclaw/`). This guide covers swapping in
or adding alongside another (e.g. OpenHands, SWE-Agent, claude-code, a custom
CLI).

In the three-layer abstraction:

- **harness-wide shared parts** live under `agent_bench/harnesses/<name>/` (gateway
  lifecycle, patches, common config templates).
- **per-(app x harness) customization** lives under
  `agent_bench/benchmarks/<app>/<name>/` (agent prompts, skills, mount/gateway-start
  helpers, host-side verify wiring). This split is load-bearing — see
  [layout.md](layout.md).

## What "harness" includes

```
agent_bench/harnesses/<name>/
├── apply_patch.py        # if the runtime needs binary patches at deploy time
├── configure.sh          # any setup / dep checks
├── gateway_up.sh         # bring the harness up for one run
├── gateway_down.sh       # tear it down
├── patch/                # source patches (vendored)
│   └── *.patch
└── state/                # runtime config template (mounted into the container)
    └── ...
```

For each application that runs on this harness:

```
agent_bench/benchmarks/<app>/<name>/
├── agents/               # static prompt files (mounted into /root/.openclaw/agents/<id>/)
└── skills/               # SKILL.md folders (OpenClaw-specific concept; each one tells the agent how to invoke a capability)
```

Cross-app gateway helpers (port pool + singularity argv wrapper) live in
`agent_bench/benchmarks/conditions/gateway_helpers.py`.

Two distinct concerns:

1. **Image plumbing** — pulling/building the SIF, applying patches, ensuring
   the binary is on PATH inside the container. Lives in `agent_bench/harnesses/<name>/`.
2. **Per-run lifecycle** — start the gateway/agent process, hand a token+port
   back to the harness, tear it down. Surfaced by `gateway_up.sh` /
   `gateway_down.sh`.

## 1. Image plumbing

For each harness, decide:

- **Where does the binary live?** On NSCC, each harness starts from a
  base/upstream SIF and bakes an Agent_Bench runtime SIF locally:
  OpenClaw via `build_oc_sif.sh`, OpenHands via `build_oh_sif.sh`, and
  Hermes via `build_hm_sif.sh`. The PBS `prepare_images` stage handles the
  pull/build step for the selected harness.
- **Does it need patching?** OpenClaw applies runtime patches to its bundled JS
  (see `agent_bench/harnesses/openclaw/patch/MANIFEST.json`). Hermes applies Python patches
  from `agent_bench/harnesses/hermes/patch/MANIFEST.json`. OpenHands does not patch upstream
  code; its builder installs the pinned SDK/tools and bakes
  `/opt/openhands_driver.py`.
- **What state does it read?** OpenClaw reads `state/openclaw.json` (gateway
  config + plugin enable/disable + agent definitions). The per-app slice
  (agent prompts, skills) lives under `agent_bench/benchmarks/<app>/<name>/` and is
  merged with the shared template at run start.

If your harness doesn't need patching, the layout simplifies to just
`gateway_up.sh` + `gateway_down.sh` + `state/`.

## 2. The gateway lifecycle

In the current OpenClaw layout the harness's `prepare_environment` directly
runs `docker run -d ... --entrypoint /usr/local/bin/openclaw <image> gateway`
and waits for the gateway-ready log line via `docker logs`. There are no
separate `gateway_up.sh` / `gateway_down.sh` scripts in the in-flight path —
those exist under `agent_bench/harnesses/openclaw/` as reference / singularity-fallback
helpers but the docker mode doesn't shell into them.

Key invariants the harness must enforce:

- **Token + port via env vars on the docker run**, never on the command line
  — they'd show up in `ps` otherwise.
- **`OTEL_RESOURCE_ATTRIBUTES=oc.run_id=$RUN_ID`** so concurrent runs don't
  cross-contaminate the shared trace file.
- **Readiness via `docker logs | grep '[gateway] ... listening on'`** — more
  reliable than HTTP polling for processes that take 1-2 minutes to load
  plugins.
- **`pick_free_gateway_port` / `release_gateway_port`** from
  `agent_bench/benchmarks/conditions/gateway_helpers.py` to coordinate ports across
  concurrent runs.

## 3. Hook into the application

The app's `app.py:prepare_environment` should: pick a port (via
`pick_free_gateway_port`), build the docker run argv inline (state_dir +
bridge mounts + env vars), wait for gateway-ready, return env_info with
gateway_url + gateway_token. SWE-bench (`agent_bench/benchmarks/swebench/app.py`) and
TB (`agent_bench/benchmarks/terminal_bench/app.py`) both follow this pattern; the OC
container start is ~30 lines of `docker run -d ...` argv with the standard
mounts.

## 4. Tell conditions about it

Conditions invoke the agent through the gateway. The current pattern (see
`agent_bench/benchmarks/conditions/openclaw_agent.py:_invoke_agent`):

```python
# Inside the container, source the env file the gateway_up.sh script wrote
docker exec $cid bash -c "source /tmp/oc_gateway_env && \
    openclaw agent --json --message ..."
```

For your harness:

1. Pick an in-container env file path (`/tmp/<harness>_env`) and have
   `gateway_up.sh` write `$URL` and `$TOKEN` to it.
2. In your condition, source it before invoking the CLI.
3. Parse the CLI's JSON output into the per-step trace shape.

## 5. Span / telemetry parity

OpenClaw's diagnostics-otel plugin emits `openclaw.model.usage` and
`openclaw.tool.call` spans. The condition's `_snapshot_otel_spans` slices
those into `spans.jsonl`, with synth fallbacks (`_maybe_synthesize_*` fills
missing per-call spans when OC's selection/diagnostics-otel patches don't
fire on the build in use).

For a new harness, mirror the schema so the analysis layer doesn't need to
branch:

| Span name | Required attrs |
|---|---|
| `openclaw.model.usage` | `openclaw.tokens.{input,output,cache_read,cache_write,total}`, `openclaw.model`, `openclaw.provider`, `openclaw.channel` |
| `openclaw.tool.call`   | `openclaw.tool.name`, `openclaw.tool.callId`, `openclaw.tool.isError`, `openclaw.runId`, `openclaw.sessionId` |

If your harness doesn't natively emit OTel, write the spans directly into
`<run_dir>/spans.jsonl` from the condition (the TB ReAct condition does this
via `agent_bench/benchmarks/terminal_bench/conditions/react.py:_emit_span`).

## 6. Smoke test

```bash
# Manual: bring up the harness, hit it once, tear down
RUN_DIR=$(mktemp -d)
bash agent_bench/harnesses/your_harness/gateway_up.sh "" 18900 "smoke-token" smoke-run
# ... your_harness agent invocation ...
bash agent_bench/harnesses/your_harness/gateway_down.sh ""
```

Then run a 1-instance sweep that uses your harness condition.

## Checklist

- [ ] `agent_bench/harnesses/<name>/gateway_up.sh` and `gateway_down.sh`, both idempotent
- [ ] `agent_bench/harnesses/<name>/state/` template (with whatever shared config the runtime needs)
- [ ] For each application running on this harness: `agent_bench/benchmarks/<app>/<name>/{agents,skills}` (prompts + skills); harness-driving code lives in `agent_bench/benchmarks/<app>/app.py`.
- [ ] Env file (`/tmp/<harness>_env`) holds URL + token for the condition to source
- [ ] `OTEL_RESOURCE_ATTRIBUTES=oc.run_id=$RUN_ID` set when launching the agent so spans demux per-run
- [ ] `gateway_down.sh` SIGTERMs first, gives ≥1s for OTel flush, then SIGKILLs
- [ ] Spans emitted in the canonical schema (or the condition writes them directly to `spans.jsonl`)
- [ ] One full run produces `verdict.json` + `step_trace.jsonl` + `spans.jsonl`
