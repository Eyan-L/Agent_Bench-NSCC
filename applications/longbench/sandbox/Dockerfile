# LongBench task/sandbox image with ClawRAG bundled in.
#
# This is the per-task ENV the agent execs into (via the exec-bridge), NOT an
# agent-runtime image — so it is agent-runtime-agnostic (works for openclaw /
# hermes / openhands alike). ClawRAG runs INSIDE this container: the agent's
# retrieval (and the harness's ingest, via `docker exec`) reach it at
# http://localhost:8080. In-process HuggingFace embeddings + embedded ChromaDB,
# no Ollama, no cloud keys.
#
# Build:
#   docker build -t agent-bench/longbench-clawrag-sandbox:1.0 \
#     -f applications/longbench/sandbox/Dockerfile applications/longbench/sandbox
#
# Runtime: the harness starts this container with `sleep infinity` (exec-bridge
# target), then `docker exec -d <c> clawrag-serve` to bring ClawRAG up on :8080.

FROM python:3.12-slim

# bash (exec-bridge needs a shell), git (clone), curl, + native libs Docling /
# python-magic require.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash git curl ca-certificates \
        libmagic1 libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ClawRAG backend source (upstream main).
RUN git clone --depth 1 https://github.com/2dogsandanerd/ClawRag /opt/clawrag

# Backend deps. ClawRAG's requirements.txt pins a CPU-only torch (via an
# --extra-index-url) to "avoid CUDA downloads" — but we WANT GPU embeddings on
# the H100s. So install the CUDA torch/torchvision build FIRST (default PyPI =
# CUDA wheels), strip the CPU index line, then install the rest (torch already
# satisfied, so the CUDA build is kept). Run the task container with --gpus for
# sentence-transformers to use the GPU. + the HuggingFace embeddings package
# (upstream ships only the Ollama one).
RUN pip install --no-cache-dir torch torchvision \
 && sed -i '\#download.pytorch.org/whl/cpu#d' /opt/clawrag/backend/requirements.txt \
 && pip install --no-cache-dir -r /opt/clawrag/backend/requirements.txt \
 && pip install --no-cache-dir llama-index-embeddings-huggingface

# Run ChromaDB IN-PROCESS (no HTTP server). Patch chroma_manager.configure() so
# host="embedded" forces in-memory mode at EVERY call site (both startup AND the
# indexing path call configure()). This sidesteps the fragile `chroma run` server
# entirely (its typer CLI breaks on click>=8.2 and its HTTP request validation is
# skewed under py3.12) — for the per-example reset/ingest/query flow, in-process
# chroma is ideal. CHROMA_HOST is set to "embedded" in ENV below.
RUN sed -i 's|        self._in_memory = in_memory$|        self._in_memory = in_memory or host in ("embedded", "memory")|' /opt/clawrag/backend/src/core/chroma_manager.py \
 && sed -i 's|        if in_memory:|        if self._in_memory:|' /opt/clawrag/backend/src/core/chroma_manager.py \
 && python -c "import py_compile; py_compile.compile('/opt/clawrag/backend/src/core/chroma_manager.py', doraise=True)" \
 && grep -q 'host in ("embedded"' /opt/clawrag/backend/src/core/chroma_manager.py

# RAG-only config: local HF embeddings, embedded Chroma, no cloud keys. LLM_*
# is set only because ClawRAG's config loader requires LLM_PROVIDER at startup;
# the LLM is lazy and never invoked on the /search (retrieval) path.
ENV EMBEDDING_PROVIDER=huggingface \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    CHROMA_HOST=embedded \
    CHROMA_PORT=8000 \
    CLAWRAG_PORT=8080 \
    HF_HOME=/opt/clawrag/cache/hf \
    LLM_PROVIDER=ollama \
    LLM_MODEL=llama3:latest \
    OLLAMA_HOST=http://localhost:11434 \
    DEBUG=false \
    LOG_LEVEL=INFO

# Cache/data dirs only — the embedding model is NOT baked into the image. Mount
# an HF cache at $HF_HOME at runtime (e.g. the shared network-drive cache) so
# embedding models are shared across runs and swappable (MiniLM -> bge-m3 for ZH)
# by just changing EMBEDDING_MODEL, no rebuild. First use downloads into the mount.
RUN mkdir -p /opt/clawrag/cache/hf /opt/clawrag/data

# `clawrag-serve`: just the ClawRAG FastAPI backend (ChromaDB is in-process).
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  ': "${CLAWRAG_PORT:=8080}"' \
  'mkdir -p /opt/clawrag/data /opt/clawrag/cache/hf' \
  'echo "[clawrag-serve] starting ClawRAG (in-process ChromaDB) on :$CLAWRAG_PORT"' \
  'cd /opt/clawrag/backend' \
  'exec uvicorn src.main:app --host 0.0.0.0 --port "$CLAWRAG_PORT"' \
  > /usr/local/bin/clawrag-serve \
 && chmod +x /usr/local/bin/clawrag-serve

WORKDIR /opt/clawrag
# Default to the exec-bridge target idle command; the harness overrides as needed
# and brings ClawRAG up via `docker exec -d <c> clawrag-serve`.
CMD ["sleep", "infinity"]
