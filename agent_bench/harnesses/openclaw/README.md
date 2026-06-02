# OpenClaw frontend patch + runtime config

Patches to OpenClaw JS package + helper scripts to spin up its gateway and
parse its OTel output.

**Pinned upstream:** OpenClaw `2026.4.22` (commit `00bd2cf`,
[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)). Full
pin in `patch/VERSION`.

## Layout

- `patch/` — `.patch` files + `MANIFEST.json` + `VERSION` (upstream pin)
- `apply_patch.py` — apply patches per `patch/MANIFEST.json` (modes: runtime/baked/skip).
  Subcommands: `apply` (default), `status`, `restore`. Driver shells out to
  `patch -p1` and validates with `node --check`.
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

Patched OpenClaw writes JSONL to `AB_TELEMETRY_TRACE_FILE` (default
`/tmp/otel/otel-traces.jsonl`). Reader: `agent_bench/telemetry/otel_reader.py`.
