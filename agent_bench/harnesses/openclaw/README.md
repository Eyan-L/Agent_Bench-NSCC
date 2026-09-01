# OpenClaw frontend patch + runtime config

Patches to OpenClaw JS package + helper scripts to spin up its gateway and
parse its OTel output.

**Pinned upstream:** OpenClaw `2026.4.22` (commit `00bd2cf`,
[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)). Full
pin in `patch/VERSION`.

## Layout

- `patch/` — `.patch` files + `MANIFEST.json` + `VERSION` (upstream pin)
- `apply_patch.py` — apply patches and named content transforms declared by
  `patch/MANIFEST.json` (modes: runtime/baked/skip). Subcommands: `apply`
  (default), `status`, `restore`. Unified diffs use `patch -p1`; transforms
  locate the pinned content-hashed runtime bundles by stable source anchors.
  Every modified JavaScript target is validated with `node --check`.
- `configure.sh` — generate per-run OpenClaw config
- `apply_profile.sh` — switch agent profile
- `gateway_up.sh` / `gateway_down.sh` — gateway lifecycle
- `defaults.env` — OpenClaw harness defaults using the `AB_*` env contract
- `state/` — runtime config templates and agent profile presets

## Post-compaction loop guard backport

The pinned `v2026.4.22` runtime predates OpenClaw's upstream
post-compaction loop guard. `apply_patch.py` backports the narrow behavior from
`openclaw/openclaw#77555` without upgrading the rest of OpenClaw:

1. A successful timeout/overflow auto-compaction arms a three-outcome window.
2. Completed tool outcomes reuse v2026.4.22's existing `toolName`, `argsHash`,
   and `resultHash` records.
3. If all three outcomes in that window are the same triple, the embedded run
   is aborted with `PostCompactionLoopPersistedError` and
   `detector=compaction_loop_persisted`.
4. If the three outcomes are not identical, the guard disarms and normal agent
   execution continues. It does not act as a general pre-compaction loop
   detector.

Like upstream, the guard is enabled unless
`tools.loopDetection.enabled` is explicitly set to `false`. The backport keeps
the upstream default window size of 3 rather than adding a new v2026.4.22
config-schema field.

The guard is baked into the OpenClaw image/overlay, so an existing NSCC overlay
must be rebuilt after changing this patch set. For the overlay workflow, rerun
`agent_bench/deployments/pbs/build_oc_overlay.sh` with `--force` before the next
benchmark smoke.

## Patch modes

`AB_HARNESS_OPENCLAW_PATCH_MODE`:
- `runtime` — apply on launcher start, idempotent via marker
- `baked` — assume image/overlay pre-patched, assert every patch/transform marker
- `skip` — run unpatched OpenClaw (ablation)

## OTel output

Patched OpenClaw writes JSONL to `AB_TELEMETRY_TRACE_FILE` (default
`/tmp/otel/otel-traces.jsonl`). Reader: `agent_bench/telemetry/otel_reader.py`.
