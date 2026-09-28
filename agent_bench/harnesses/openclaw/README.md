# OpenClaw frontend patch + runtime config

Patches to OpenClaw JS package + helper scripts to spin up its gateway and
parse its OTel output.

**Pinned upstream:** OpenClaw `2026.4.22` (commit `00bd2cf`,
[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)). Full
pin in `patch/VERSION`.

## Layout

- `patch/` — `.patch` files + `MANIFEST.json` + `VERSION` (upstream pin)
- `apply_patch.py` — apply patches per `patch/MANIFEST.json` (modes: runtime/baked/skip).
  Subcommands: `apply` (default), `status`, `restore`. The driver applies
  unified diffs and named content transforms, then validates with `node --check`.
- `runtime_compat_patch.py` — pinned v2026.4.22 session-patch timeout and
  offline pricing compatibility, used by the SIF and overlay builders.
- `configure.sh` — generate per-run OpenClaw config
- `apply_profile.sh` — switch agent profile
- `gateway_up.sh` / `gateway_down.sh` — gateway lifecycle
- `defaults.env` — OpenClaw harness defaults using the `AB_*` env contract
- `state/` — runtime config templates and agent profile presets

## Patch modes

`AB_HARNESS_OPENCLAW_PATCH_MODE`:
- `runtime` — apply on launcher start, idempotent via marker
- `baked` — assume image pre-patched, skip apply
- `skip` — run unpatched OpenClaw (ablation)

## OTel output

The v2026.4.22 manifest also includes the post-compaction loop guard. It aborts
three repeated completed tool outcomes with identical arguments and results
after successful compaction. v2026.9.2 uses its native guard and the separate
`v2026_9_2_patch.py` compatibility pass; the legacy backport is not reapplied.

Patched OpenClaw writes JSONL to `AB_TELEMETRY_TRACE_FILE` (default
`/tmp/otel/otel-traces.jsonl`). Reader: `agent_bench/telemetry/otel_reader.py`.
