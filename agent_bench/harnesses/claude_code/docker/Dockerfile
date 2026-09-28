FROM node:22-bookworm-slim

ARG CLAUDE_CODE_VERSION=2.1.278
ARG LITELLM_VERSION=1.102.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --prefix /tmp/claude-code-install \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && mkdir -p /opt/claude-code/bin \
    && claude_entry="$(node -e \
        'const fs=require("fs"),path=require("path"); const dir="/tmp/claude-code-install/node_modules/@anthropic-ai/claude-code"; const pkg=JSON.parse(fs.readFileSync(path.join(dir,"package.json"),"utf8")); process.stdout.write(path.resolve(dir,pkg.bin.claude));')" \
    && test -x "$claude_entry" \
    && test "$(stat -c %s "$claude_entry")" -gt 1000000 \
    && cp "$claude_entry" /opt/claude-code/bin/claude \
    && chmod 0755 /opt/claude-code/bin/claude \
    && rm -rf /tmp/claude-code-install \
    && python3 -m venv /opt/litellm \
    && /opt/litellm/bin/pip install --no-cache-dir \
        "litellm[proxy]==${LITELLM_VERSION}" \
    && /opt/claude-code/bin/claude --version \
    && /opt/litellm/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("litellm"))'

ENTRYPOINT ["/opt/litellm/bin/litellm"]
