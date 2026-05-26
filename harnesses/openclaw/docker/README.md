# OpenClaw harness image

Thin overlay on the official upstream image
`ghcr.io/openclaw/openclaw:<version>`. We only own the patch layer — the
base ships multi-arch (amd64 + arm64), non-root `node` user, `/healthz` +
`/readyz` endpoints, and `tini` as PID 1.

## Build

```bash
bash harnesses/openclaw/docker/build.sh
```

The version is read from `../patch/VERSION` (`upstream_version`); the
default tag is `agent-bench/openclaw:<version>`. Override the tag with
`OC_IMAGE_TAG=foo/openclaw:dev`.

What the build does:
1. `FROM ghcr.io/openclaw/openclaw:<version>` (multi-arch).
2. As root, `apt-get install python3 patch ca-certificates docker.io` so
   `apply_patch.py` can run and the gateway can spawn sandbox containers.
3. Plugin runtime-deps pre-warm (as `node`): `openclaw plugins list` plus a
   throwaway `openclaw agent --local` run against a dead endpoint. The agent
   run is what forces the *provider* plugins (openai, xai, ollama, …) to load
   and `npm install` their bundled deps (`@mariozechner/pi-ai`, `typebox`,
   `ws`) into `dist/extensions/<plugin>/node_modules/`. Baking these at build
   time (single process) prevents the gateway+agent ENOTEMPTY race that
   crashes ~1/3 of agent starts at run time. The throwaway config is removed
   afterwards. Must run before `apply_patch` so the per-plugin `pi-ai` copies
   exist for `target_glob_all` to patch.
4. `python3 apply_patch.py apply` (mode=runtime) — bakes the diffs in
   `../patch/*.patch` into upstream's `dist/*.js`.
5. `python3 apply_patch.py status` (mode=baked) — fails the build if any
   marker is missing.
6. `chown -R node:node /opt/openclaw-tools` and switch back to `USER node`.

## Smoke

```bash
bash harnesses/openclaw/docker/smoke.sh
```

Three checks:
- `openclaw --version` matches the pinned `upstream_version`
- `apply_patch.py status` (mode=baked) reports all four patches `applied`
- `openclaw gateway --help` runs

`smoke.sh` overrides `--entrypoint /usr/local/bin/openclaw` for the
version probe so the upstream `tini -s --` ENTRYPOINT doesn't intercept the
flag.

## Notes / known gotchas

- Bin shim: upstream's `/usr/local/bin/openclaw` is a real symlink to
  `/app/openclaw.mjs`. Do **not** `COPY --from=...` it across stages —
  Docker COPY follows the symlink, leaves a regular file at
  `/usr/local/bin/openclaw`, and `./dist/entry.js` then resolves to
  `/usr/local/bin/dist/entry.js` (which doesn't exist).
- Patch hashes: upstream's `dist/agent-command-*.js` etc. have different
  content-hashes than a local npm install. `apply_patch.py` resolves them
  via `target_glob` in `../patch/MANIFEST.json`, so the same patch files
  apply cleanly on either layout.

## Wiring into the harness

Today the SWE-bench application mounts host OpenClaw + Node into the
per-instance task container
(`applications/swebench/openclaw/gateway_compose.compose_oc_mounts`).
Running this image as a sidecar over a docker network — the actual
two-container split — is **deferred**. See
[../state/openclaw.json](../state/openclaw.json), already locked to
`tools.exec.host = "gateway"` so the agent's exec path is gateway-routable
when the split lands.
