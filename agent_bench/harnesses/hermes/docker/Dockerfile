# Hermes harness image — thin overlay on the official upstream image
# at nousresearch/hermes-agent:<version>. We only own the patch layer and
# small runtime conveniences for Agent_Bench.
#
# Pin: agent_bench/harnesses/hermes/patch/VERSION (upstream_version: v2026.5.16).
# Build context must be `agent_bench/harnesses/hermes/` (apply_patch.py + patch/).
#
# Build:
#   bash agent_bench/harnesses/hermes/docker/build.sh
#
# Smoke:
#   bash agent_bench/harnesses/hermes/docker/smoke.sh
#
# Runtime model (see agent_bench/benchmarks/swebench/hermes/app.py:SWEBenchHermesHarness):
#
#   ┌─────────────────────────────────────┐
#   │ this image (hm harness container)    │
#   │ - upstream Hermes Agent CLI          │
#   │ - patched docker terminal backend    │ ← baked by patch/
#   │ - hm-exec-bridge mounted at          │
#   │   /usr/local/bin/hm-exec-bridge      │ ← mounted by dispatcher
#   │ - HERMES_EXEC_BRIDGE_* env set       │
#   └────────┬─────────────────────────────┘
#            │ tool/file/code exec → bridge CLI → unix socket → host server
#            ↓
#   ┌─────────────────────────────────────┐
#   │ swebench task container              │
#   │ - per-instance python env            │
#   │ - workspace bind-mounted at /testbed │
#   └─────────────────────────────────────┘
#
# Agent_Bench, not Hermes, owns the SWE-bench sandbox lifecycle. Hermes still
# believes its terminal backend is docker; the patch swaps docker exec for the
# external bridge when HERMES_EXEC_BRIDGE is present.

ARG HERMES_VERSION=v2026.5.16
FROM nousresearch/hermes-agent:${HERMES_VERSION}
ARG HERMES_VERSION

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git patch tini \
 && rm -rf /var/lib/apt/lists/*

ENV HERMES_VERSION=${HERMES_VERSION} \
    HERMES_TOOLS_DIR=/opt/hermes-tools \
    HERMES_PATCH_MODE=runtime

COPY apply_patch.py ${HERMES_TOOLS_DIR}/apply_patch.py
COPY patch ${HERMES_TOOLS_DIR}/patch

RUN python3 ${HERMES_TOOLS_DIR}/apply_patch.py apply \
 && HERMES_PATCH_MODE=baked python3 ${HERMES_TOOLS_DIR}/apply_patch.py status

RUN mkdir -p /workspace /opt/data

ENV HERMES_PATCH_MODE=baked \
    HERMES_HOME=/opt/data \
    PATH=/opt/hermes/.venv/bin:/opt/hermes:${PATH} \
    PYTHONPATH=/opt/hermes

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
