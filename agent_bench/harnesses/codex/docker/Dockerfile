FROM node:22-bookworm-slim AS download
ARG CODEX_VERSION=0.157.1
RUN npm install --prefix /download @openai/codex@${CODEX_VERSION} \
 && mkdir -p /payload/bin \
 && cp /download/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/* /payload/bin/

FROM python:3.11-slim-bookworm
ARG CODEX_VERSION=0.157.1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git tini \
 && rm -rf /var/lib/apt/lists/*
COPY --from=download /payload /opt/codex
RUN /opt/codex/bin/codex --version | grep -Fx "codex-cli ${CODEX_VERSION}" \
 && sha256sum /opt/codex/bin/* > /opt/codex/SHA256SUMS
ENV PATH=/opt/codex/bin:${PATH}
WORKDIR /tmp
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
