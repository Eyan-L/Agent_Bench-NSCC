# OpenHands harness image — Python base + openhands-sdk + openhands-tools.
#
# Pin: harnesses/openhands/VERSION (upstream_version: 1.21.0).
# Build context must be `harnesses/openhands/` (driver.py + state/ + VERSION).
#
# Build:
#   bash harnesses/openhands/docker/build.sh
#
# Smoke:
#   bash harnesses/openhands/docker/smoke.sh
#
# Runtime model (see applications/swebench/app.py:SWEBenchOpenHandsHarness):
#
#   ┌─────────────────────────────────────┐
#   │ this image (oh harness container)    │
#   │ - openhands-sdk + openhands-tools    │
#   │ - /opt/openhands_driver.py           │ ← entry point
#   │ - oh-exec-bridge mounted at          │
#   │   /usr/local/bin/oh-exec-bridge      │ ← mounted by dispatcher
#   │ - OPENHANDS_EXEC_BRIDGE_* env set    │
#   └────────┬─────────────────────────────┘
#            │ tool exec → bridge CLI → unix socket → host server
#            ↓
#   ┌─────────────────────────────────────┐
#   │ swebench task container              │
#   │ - per-instance python env            │
#   │ - workspace bind-mounted at /testbed │
#   └─────────────────────────────────────┘
#
# We deliberately do NOT bake openhands' config.toml here — the harness
# renders one per run on the host (from config/harness + config/model)
# and bind-mounts it at /etc/agent-bench/state/config.toml.

ARG OPENHANDS_VERSION=1.21.0
FROM python:3.12-slim
ARG OPENHANDS_VERSION

# Why uv: openhands-sdk pulls lmnr which pins
# opentelemetry-semantic-conventions to an exact version that pip's resolver
# cannot satisfy. uv's resolver handles it cleanly.
#
# Why git + tini: git so the driver / agent can git diff for sanity (the real
# patch is collected on the host); tini so PID-1 cleans up if the harness
# stops the container with SIGTERM.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv \
 && uv pip install --system --no-cache \
        "openhands-sdk==${OPENHANDS_VERSION}" \
        "openhands-tools==${OPENHANDS_VERSION}"

# Build-time sanity: import both packages so a resolver regression in a
# patch release breaks the build, not every benchmark run.
RUN python -c "import openhands.sdk, openhands.tools; print('openhands sdk+tools OK')"

# Bake the driver into the image. Per-run config.toml + the bridge binary
# are bind-mounted at runtime, see SWEBenchOpenHandsHarness.
COPY driver.py /opt/openhands_driver.py
RUN chmod 0755 /opt/openhands_driver.py

# Canonical in-container mount targets created at build time so the runtime
# bind-mounts have somewhere to land.
RUN mkdir -p /workspace /etc/agent-bench/state /var/openhands/events

ENV OPENHANDS_SUPPRESS_BANNER=1 \
    OPENHANDS_VERSION=${OPENHANDS_VERSION}

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
