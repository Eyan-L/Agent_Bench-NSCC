# Roadmap-specific filesystem targets over the existing patched Hermes image.
ARG HERMES_BASE_IMAGE
FROM ${HERMES_BASE_IMAGE}

USER root
RUN mkdir -p /app /opt/data /usr/local/bin \
 && touch /usr/local/bin/hm-exec-bridge \
 && chmod 0755 /usr/local/bin/hm-exec-bridge
WORKDIR /app
