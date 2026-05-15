# OpenClaw harness image — thin overlay on the official upstream image
# at ghcr.io/openclaw/openclaw:<version>. We only own the patch layer + a few
# runtime extras (docker CLI for native sandbox spawn, plugin pre-warm).
#
# Pin: harnesses/openclaw/patch/VERSION (upstream_version: 2026.4.22).
# Build context must be `harnesses/openclaw/` (apply_patch.py + patch/).
#
# Build:
#   bash harnesses/openclaw/docker/build.sh
#
# Smoke:
#   bash harnesses/openclaw/docker/smoke.sh
#
# Runtime model (see applications/swebench/app.py:SWEBenchHarness):
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
# bundled deps (e.g. github-copilot needs @mariozechner/pi-ai) on first plugin
# init in any node process. With two openclaw processes (gateway + agent CLI)
# starting back-to-back, they race-install on the same dirs and trash each
# other with ENOTEMPTY. Bake the install at image build time so subsequent
# loads detect "already there" and skip the npm step.
#
# This MUST run before apply_patch — the warmup installs per-plugin copies
# of @mariozechner/pi-ai into dist/extensions/<plugin>/node_modules/, which
# the patch step then has to find via target_glob_all.
RUN su node -c '/usr/local/bin/openclaw plugins list >/tmp/oc_warmup.log 2>&1' \
 && tail -20 /tmp/oc_warmup.log \
 && rm /tmp/oc_warmup.log

RUN python3 ${OC_TOOLS_DIR}/apply_patch.py apply \
 && OC_PATCH_MODE=baked python3 ${OC_TOOLS_DIR}/apply_patch.py status \
 && chown -R node:node ${OC_TOOLS_DIR}

# Switch back to upstream's runtime profile.
ENV OC_PATCH_MODE=baked
USER node
WORKDIR /app
