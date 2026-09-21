FROM node:22-bookworm-slim

ARG CLAUDE_CODE_VERSION=2.1.278
ARG LITELLM_VERSION=1.102.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --prefix /opt/claude-code \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && mkdir -p /opt/claude-code/bin \
    && cp /usr/local/bin/node /opt/claude-code/bin/node \
    && printf '%s\n' \
        '#!/bin/sh' \
        'exec "$(dirname "$0")/node" "$(dirname "$0")/../node_modules/@anthropic-ai/claude-code/cli.js" "$@"' \
        > /opt/claude-code/bin/claude \
    && chmod 0755 /opt/claude-code/bin/claude \
    && python3 -m venv /opt/litellm \
    && /opt/litellm/bin/pip install --no-cache-dir \
        "litellm[proxy]==${LITELLM_VERSION}" \
    && /opt/claude-code/bin/claude --version \
    && /opt/litellm/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("litellm"))'

ENTRYPOINT ["/opt/litellm/bin/litellm"]
