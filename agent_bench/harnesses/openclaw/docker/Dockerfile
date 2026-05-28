# OpenClaw harness image — thin overlay on the official upstream image
# at ghcr.io/openclaw/openclaw:<version>. We only own the patch layer + a few
# runtime extras (docker CLI for native sandbox spawn, plugin pre-warm).
#
# Pin: agent_bench/harnesses/openclaw/patch/VERSION (upstream_version: 2026.4.22).
# Build context must be `agent_bench/harnesses/openclaw/` (apply_patch.py + patch/).
#
# Build:
#   bash agent_bench/harnesses/openclaw/docker/build.sh
#
# Smoke:
#   bash agent_bench/harnesses/openclaw/docker/smoke.sh
#
# Runtime model (see agent_bench/benchmarks/swebench/app.py:SWEBenchHarness):
#   The gateway in this container reads agents.defaults.sandbox.docker from
#   openclaw.json and `docker run`s a per-session sandbox container (the
#   SWE-bench instance image) where tool exec lands. So we need:
#     - docker CLI on PATH
#     - /var/run/docker.sock bind-mounted at runtime
#     - --user 0 at runtime (host docker.sock GID seldom matches `node`)

ARG OPENCLAW_VERSION=2026.4.22
FROM ghcr.io/openclaw/openclaw:${OPENCLAW_VERSION}
ARG OPENCLAW_VERSION

USER root
# Upstream image is bookworm-slim. Bringing in:
#   - python3 + patch  → applying the patches in patch/
#   - docker.io        → CLI so the gateway can spawn sandbox containers
#                        (talks to host daemon via the bind-mounted socket)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates patch python3 docker.io \
 && rm -rf /var/lib/apt/lists/*

# Bake the in-repo patches into upstream's bundled JS. apply_patch.py finds
# the content-hashed bundle by target_glob.
ENV OC_HOST_OPENCLAW=/app \
    OC_PATCH_MODE=runtime \
    OC_TOOLS_DIR=/opt/openclaw-tools \
    OPENCLAW_VERSION=${OPENCLAW_VERSION}

COPY apply_patch.py ${OC_TOOLS_DIR}/apply_patch.py
COPY defaults.env   ${OC_TOOLS_DIR}/defaults.env
COPY patch          ${OC_TOOLS_DIR}/patch

# Plugin runtime-deps pre-warm. OpenClaw lazily npm-installs each plugin's
# bundled deps (e.g. @mariozechner/pi-ai, typebox, ws) the first time that
# plugin is *loaded* in a node process, into dist/extensions/<plugin>/node_modules/.
# At run time the gateway + agent CLI start back-to-back and race-install into
# the same per-plugin dirs, trashing each other with ENOTEMPTY on the shared
# deps — this crashes ~1/3 of agent starts ("No patch generated" / empty
# conversation). Bake the installs at build time (single process, no second
# racer) so every subsequent load detects "already there" and skips npm.
#
# Two stages, because different OpenClaw entrypoints load different plugin sets:
#   (a) `plugins list` — reads manifests; warms the always-on / channel set.
#   (b) a throwaway `agent` run — the ONLY entrypoint that loads the *provider*
#       plugins (openai, xai, ollama, fireworks, github-copilot, lmstudio, kimi,
#       mistral, deepgram, elevenlabs, microsoft). Neither `plugins list` nor a
#       gateway start (which loads only ~5 always-on plugins) touches these, yet
#       they are exactly the ones that race at run time. The dead endpoint makes
#       the LLM call fail, but plugin load (and its npm install) completes first,
#       so the deps are baked regardless. The throwaway config is then removed so
#       the image ships only the warmed node_modules.
#
# Both stages MUST run before apply_patch — they install the per-plugin copies
# of @mariozechner/pi-ai that the patch step finds via target_glob_all.
RUN su node -c '/usr/local/bin/openclaw plugins list >/tmp/oc_warmup.log 2>&1' \
 && tail -20 /tmp/oc_warmup.log \
 && rm /tmp/oc_warmup.log

# The warmup config must mirror the harness's real plugin set (see
# profiles/harnesses/openclaw_27b.yaml -> openclaw.plugins). The harness ENABLES
# diagnostics-otel (≈11 @opentelemetry/* pkgs) + memory-core (chokidar/typebox),
# and acpx is always-on — these are exactly the plugins whose npm install fails
# with -122 against npmmirror at run time (the agent CLI's ensureCliPluginRegistry
# throws PluginLoadFailureError on acpx → no agent → "No patch generated"). So we
# enable them here too, and leave the provider plugins at their default-on state
# so their deps get baked as well (future-proof; harness disables them but other
# configs may not). Build uses the reliable default npm registry, not npmmirror.
RUN su node -c 'set -e; export HOME=/home/node; cd /home/node; \
      openclaw config set gateway.mode local >/dev/null; \
      openclaw config set diagnostics.enabled true >/dev/null; \
      openclaw config set diagnostics.otel.enabled true >/dev/null; \
      openclaw config set plugins.entries.diagnostics-otel.enabled true >/dev/null; \
      openclaw config set plugins.entries.memory-core.enabled true >/dev/null; \
      openclaw config set --strict-json models.providers.warmup "{\"baseUrl\":\"http://127.0.0.1:9/v1\",\"apiKey\":\"sk-warmup\",\"models\":[{\"id\":\"warmup\",\"name\":\"warmup\"}]}" >/dev/null; \
      openclaw config set --strict-json agents.list[0] "{\"id\":\"warmup\",\"default\":true,\"name\":\"warmup\",\"workspace\":\"/home/node/ws\",\"tools\":{\"profile\":\"coding\",\"exec\":{\"host\":\"node\",\"security\":\"full\"}}}" >/dev/null; \
      mkdir -p /home/node/ws; \
      timeout 300 openclaw agent --local --agent warmup --message warmup >/tmp/oc_agent_warmup.log 2>&1 || true; \
      echo "[warmup] runtime-dep install lines: $(grep -c "installed bundled runtime deps" /tmp/oc_agent_warmup.log || echo 0)"; \
      grep -E "installed bundled runtime deps" /tmp/oc_agent_warmup.log || true' \
 && echo "[warmup] asserting the harness-loaded residual plugins are now baked..." \
 && test -n "$(ls -A /app/dist/extensions/acpx/node_modules 2>/dev/null)" \
 && test -n "$(ls -A /app/dist/extensions/diagnostics-otel/node_modules 2>/dev/null)" \
 && test -n "$(ls -A /app/dist/extensions/memory-core/node_modules 2>/dev/null)" \
 && test -f /app/dist/extensions/openai/node_modules/@mariozechner/pi-ai/package.json \
 && rm -rf /home/node/.openclaw /home/node/ws /tmp/oc_agent_warmup.log

RUN python3 ${OC_TOOLS_DIR}/apply_patch.py apply \
 && OC_PATCH_MODE=baked python3 ${OC_TOOLS_DIR}/apply_patch.py status \
 && chown -R node:node ${OC_TOOLS_DIR}

# Switch back to upstream's runtime profile.
ENV OC_PATCH_MODE=baked
USER node
WORKDIR /app
