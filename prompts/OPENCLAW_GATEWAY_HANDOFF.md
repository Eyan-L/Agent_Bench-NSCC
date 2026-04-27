# OpenClaw SWE-bench Integration — Hand-off to Claude Code

## Current working state

End-to-end pipeline **works** for 3 conditions against SWE-bench Verified instance
`astropy__astropy-12907`:

- `ar_issue_only` — baseline, FAIL (LLM hallucinates diff, patch won't apply)
- `ar_static_context` — baseline + BM25, FAIL (same root cause)
- `openclaw_agent` — full trajectory, **PASS** (correctly fixes the bug)

Project root: `/home/wenyan/czlu/github/agentic_workflow`

## What's in place

### Docker / SWE-bench
- Per-run container: `docker run -d` on `swebench/sweb.eval.x86_64.<instance>:latest`
- Agent container runs with `--network host` so it can reach vLLM on localhost:8000
- SWE-bench evaluation uses `python -m swebench.harness.run_evaluation ...`, reads
  report from `<SWE-bench>/logs/run_evaluation/<run_id>/oc-eval/<instance_id>/report.json`

### vLLM
- Running on host, `http://localhost:8000/v1`, model `Qwen/Qwen3.6-35B-A3B`
- AR baselines use `chat_template_kwargs: {enable_thinking: false}` because
  Qwen3.6 puts reasoning tokens in `message.reasoning`, leaving `content=null`
  unless thinking is disabled
- OpenClaw agent uses thinking=off via agent config

### OpenClaw (installed at `/home/wenyan/.npm-global/lib/node_modules/openclaw`)
- Config: `~/.openclaw/openclaw.json` (also mounted from
  `<project>/openclaw_state/openclaw.json` into container)
- Agent profile at `<project>/openclaw_state/agents/swebench/*.md` — 6 minimal
  files replacing OpenClaw's Slack-bot persona (AGENTS.md etc)
- Running via `openclaw agent --local --agent swebench --session-id <id>
  --message <issue> --json`
- vllm provider registered in config with baseUrl=http://localhost:8000/v1,
  model id Qwen/Qwen3.6-35B-A3B

### Container mounts (harness.py `prepare_environment`)
- host Node binary → /usr/local/bin/node (ro)
- host npm pkg dir → /usr/local/lib/node_modules/npm (ro)
- host OpenClaw pkg → /opt/openclaw (ro)
- host openclaw.json → /root/.openclaw/openclaw.json (ro)
- 6 profile .md files → /root/.openclaw/agents/swebench/*.md (ro, per-file)
- ~/.cache/oc-eval/openclaw-plugin-deps → /root/.openclaw/plugin-runtime-deps (rw shared)
- Symlinks created post-start: openclaw → /opt/openclaw/openclaw.mjs,
  npm → /usr/local/lib/node_modules/npm/bin/npm-cli.js

### OpenClaw scaffold pollution workaround
OpenClaw writes AGENTS.md / SOUL.md / IDENTITY.md / USER.md / TOOLS.md /
HEARTBEAT.md + .openclaw/, memory/, sessions/ into workspace (/testbed) on
agent startup. `openclaw_agent.py` `_collect_patch` strips these before
`git diff HEAD` to keep candidate.patch clean.

## What's NOT working

Current `openclaw_agent` run in `--local` mode:

- **Tool calls fail once then fallback.** stderr has many
  `[tools] exec failed: gateway closed (1008): unauthorized: gateway token missing`
  lines. Tool calls eventually succeed via local Node fallback but take ~2x time.
- **Token usage = 0 in manifest.** `_extract_last_json` sometimes doesn't
  catch the final payload (has `meta.agentMeta.usage`). Other times it does.
  Inconsistent.
- **No per-tool-call trace data.** stderr in debug mode emits
  `[agent/embedded] ... durationMs=X` for LLM turns but not for individual
  tool calls. Only has aggregate `toolSummary.calls: 25, tools: [exec, read,
  write, edit]`, no individual durations.
- **OTel is configured** (diagnostics.otel.enabled=true, endpoint
  http://localhost:4318, collector oc-otel running) **but `--local` mode
  doesn't emit OTel** — schema says OTel is "by gateway components" only.
  Verified: 0 spans arrive at collector when running `--local`.

## What we want

**Switch from `--local` to Gateway mode.** Benefits:

1. Tool calls stop failing+fallback → ~2x speedup on openclaw_agent runs
2. OTel instrumentation activates → per-LLM-call spans in collector
3. Gateway's internal WebSocket is the intended architecture for tool
   orchestration anyway

## Open questions to investigate BEFORE writing harness code

```bash
OC=/home/wenyan/.npm-global/bin/openclaw

# 1. Gateway subcommand surface
$OC gateway --help 2>&1
# probably: start / stop / status / token? subcommands

# 2. Gateway schema (token config, port, bind)
$OC config schema 2>&1 | python3 -c "
import json, sys
s = json.load(sys.stdin)
print(json.dumps(s.get('properties', {}).get('gateway', {}), indent=2)
)"

# 3. Token field location (config set path)
$OC config schema 2>&1 | grep -iC 2 'gateway.*token\|token.*gateway'

# 4. Is gateway running now? Should be dormant since we were using --local
ps aux | grep -iE 'openclaw.*gateway' | grep -v grep
ss -tlnp 2>/dev/null | grep -E ':(19000|18789|19001)' \
    || netstat -tlnp 2>/dev/null | grep -E ':(19000|18789|19001)' \
    || echo "no gateway port"

# 5. Concurrency — each SWE-bench run has its own container. Can each
#    container run its own gateway? Default port is 19000 per schema
#    description. If we do --network host, port collisions across parallel
#    runs in phase 1/2.
```

## Design constraints

1. **Per-run fairness**: Each SWE-bench run must start from zero-state (no
   cached memory/sessions from previous runs). Fine to have per-container
   Gateway that dies with container.

2. **No host-side state pollution**: Gateway lifecycle is per-container.
   `--network host` means port conflicts — Gateway needs different port per
   container OR use `--network bridge` with port mapping (but then vLLM
   routing gets complex).

3. **Plugin cache sharing is OK** (read-only concerns only). Already done
   via rw mount of `~/.cache/oc-eval/openclaw-plugin-deps`.

4. **Timeouts**: Gateway startup adds latency per run. Currently Phase 0 is
   serial, so total time = sum. Phase 1 will be concurrent (max_workers > 1)
   — Gateway per container is required for isolation.

5. **OTel collector**: Already running at `oc-otel` container, endpoint
   `http://localhost:4318`, file exporter writing to `/tmp/otel/otel-traces.jsonl`.
   Each run should tag spans with `oc.run_id` attribute for de-multiplexing.

## Files to touch

- `eval/runner/harness.py` — add Gateway lifecycle to `prepare_environment`
  and `cleanup`. Container startup: start gateway, wait for ready, create+inject
  token, THEN let condition run. Teardown: stop gateway, rm container.
- `eval/runner/conditions/openclaw_agent.py` — drop `--local` flag.
  Currently invokes `openclaw agent --local --agent swebench ...`; should
  become `openclaw agent --agent swebench ...`.
- `openclaw_state/openclaw.json` — add gateway section (port, token, auth
  config). Exact schema TBD by investigation.
- Optionally: a new `scripts/openclaw_gateway_up.sh` that encapsulates the
  "start gateway, wait ready, create token" sequence, callable from harness
  via `docker exec`.

## Files that are correct (don't regress)

- `openclaw_state/agents/swebench/AGENTS.md` and siblings (minimal profile)
- OTel collector config at `/tmp/otel/config.yaml`
- AR baseline conditions (`ar_issue_only.py`, `ar_static_context.py`) —
  they use a shared `_llm_client.py` for HTTP to vLLM with
  `chat_template_kwargs.enable_thinking=false`

## Known bundled-source locations (already grep-verified)

- `dist/plugin-sdk/src/agents/pi-embedded-runner/run.d.ts` — runEmbeddedPiAgent
- `dist/plugin-sdk/src/agents/command/session-store.d.ts` — updateSessionStoreAfterAgentRun
- `dist/diagnostic-events-DU0Yv8MX.js` — emitDiagnosticEvent + isDiagnosticsEnabled
- `dist/plugin-sdk/src/infra/diagnostic-events.d.ts` — has `type: "model.usage"` schema declared
- `dist/plugin-sdk/diagnostics-otel.js` — the OTel exporter plugin
- `dist/extensions/diagnostics-otel/openclaw.plugin.json` — OTel plugin manifest
- `dist/agent-command-B6RjqeHV.js` — actual CLI agent command impl, contains
  `loadSessionStoreRuntime()` call site referenced in issue tracker discussion

## Stretch goal (do not attempt unless Gateway route works first)

Patch `--local` CLI path to emit `model.usage` diagnostic event, per
community issue — would let `--local` mode emit OTel too. But **do the
Gateway route first**; patching bundled source has maintenance cost.

## Required deliverables (must be committed to repo)

Everything you do has to survive a `git clone && make run-phase0` on a fresh
machine (assuming vLLM, Docker, and SWE-bench are already set up). That means
**no steps that live only in your working memory or one-off terminal commands**.

### Scripts to write and commit under `scripts/`

- `scripts/openclaw_gateway_up.sh` — Start the Gateway inside a given
  container: `docker exec <cid>` wrapper that runs `openclaw gateway start`,
  polls for readiness (health check or port listen), creates a token, and
  writes it where OpenClaw expects. Must be idempotent, must exit non-zero
  on failure with a clear message, must log to stderr what it's doing.
  Takes container id as arg.

- `scripts/openclaw_gateway_down.sh` — Tear down gracefully in case
  harness crashes mid-run. `docker exec <cid> openclaw gateway stop` or
  equivalent. Idempotent: ok if already stopped.

- `scripts/otel_collector_up.sh` — Start the OTel collector container
  (currently `oc-otel` is running by hand). Hard-codes the config path
  and exporter target under the project. Safe to re-run.

- `scripts/otel_collector_down.sh` — Stop + rm the OTel collector.

- `scripts/otel_reset_traces.sh` — Truncate the file exporter's output
  between runs so per-run spans don't mix.

### Config to commit

- `otel/collector-config.yaml` — The collector config that is currently at
  `/tmp/otel/config.yaml`. Move it inside the project so it survives
  reboots and is version-controlled. Update collector_up.sh to mount from
  this path.

- `openclaw_state/openclaw.json` — Updated to include the gateway config
  block (port, auth, whatever you set). Currently the config was edited
  piecemeal via `openclaw config set` commands. After this round, the
  committed openclaw.json should be the single source of truth; on a fresh
  machine you should be able to copy it to `~/.openclaw/openclaw.json` (or
  mount it into the container) and have a working config with no further
  setup.

### Makefile targets to add

In the top-level Makefile, add:

- `make otel-up` → calls `scripts/otel_collector_up.sh`
- `make otel-down` → calls `scripts/otel_collector_down.sh`
- `make otel-reset` → calls `scripts/otel_reset_traces.sh`
- Document in the Makefile's `help` target what these do.

### Documentation to update

- `README.md` — in the setup / quickstart section, add a note that
  running openclaw_agent requires `make otel-up` once at the start of
  the day. Call out that Gateway lifecycle is per-run (automatic, inside
  `make run-phase0`) but the OTel collector is shared (manual).

- `CONVENTIONS.md` (if it mentions OpenClaw) — update to reflect Gateway
  mode as the default, drop any references to `--local` mode as the
  runtime.

### Do NOT

- Leave behavior "works if you remember to run X command first" unless X is
  documented and wrapped in a Makefile target.
- Leave port numbers / token values hard-coded in harness.py if they
  belong in config. harness.py should read them from env or from
  `openclaw_state/openclaw.json`.
- Ship the OpenClaw token committed to git. If the token is persistent,
  put it in `.env` (already gitignored) and have harness pick it up from
  environment; or have the gateway_up script generate a fresh token each
  run and write it to a tmpfile that harness reads.

## Success criteria

After this round:

1. `make run-phase0` runs all 3 conditions, openclaw_agent still PASSes
2. `openclaw_agent` verdict.json shows real token numbers (not 0)
3. OTel collector's `/tmp/otel/otel-traces.jsonl` has spans from the
   openclaw_agent run (at least `agent.run` + per-LLM-call spans)
4. No `gateway closed` errors in openclaw_stderr.log
5. openclaw_agent run time ideally <60 seconds (currently ~110s due to
   tool fallback overhead)

## Reproducibility checklist

At the end, verify the whole thing survives a cold restart:

```bash
# Simulate fresh machine: stop everything, clear caches (keep plugin cache)
make clean-runs
docker rm -f $(docker ps -aq --filter name=oc-eval-) 2>/dev/null
make otel-down

# Cold start from commit
make otel-up
make run-phase0

# Success criteria above must still hold
```

If any step in this sequence needs you to manually run a command or edit a
file, it's not done. Wrap it in a script and commit.

## How to verify

```bash
cd /home/wenyan/czlu/github/agentic_workflow
rm -rf data/runs/phase0/
: > /tmp/otel/otel-traces.jsonl
make run-phase0

# manifest check
for d in $(ls -td data/runs/phase0/*/ | head -3); do
    python3 -c "
import json
m = json.load(open('$d/manifest.json'))
print(f\"{m.get('condition_id'):22s} status={m.get('status'):8s} \"
      f\"dur={m.get('duration_seconds'):6.0f}s \"
      f\"in={m.get('total_prompt_tokens'):7d} \"
      f\"out={m.get('total_completion_tokens'):6d}\")"
done

# OTel check
wc -l /tmp/otel/otel-traces.jsonl
cat /tmp/otel/otel-traces.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    if not line.strip(): continue
    d = json.loads(line)
    for rs in d.get('resourceSpans', []) or []:
        for ss in rs.get('scopeSpans', []) or []:
            for sp in ss.get('spans', []) or []:
                start = int(sp.get('startTimeUnixNano', 0))
                end = int(sp.get('endTimeUnixNano', 0))
                dur_ms = (end - start) / 1e6 if end and start else 0
                print(f\"{sp.get('name','?'):40s} {dur_ms:8.1f} ms\")
"

# No more gateway closed errors
RUN_DIR=$(grep -l '"condition_id": "openclaw_agent"' data/runs/phase0/*/manifest.json | xargs dirname)
grep -c 'gateway closed' "$RUN_DIR/openclaw_stderr.log" || echo "0 (good)"
```