FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

RUN apt update && apt install -y \
    python3 python-is-python3 python3-pip python3-dev \
    git build-essential \
    libopenblas-dev \
    libssl-dev \
    wget curl \
    cmake \
    ninja-build \
    git-lfs \
    kmod \
    etcd \
    libibverbs1 \
    ibverbs-providers \
    rdma-core \
    pkg-config \
    tmux \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

ARG SGLANG_VERSION=0.5.5
RUN pip install sglang==${SGLANG_VERSION}

WORKDIR /opt
RUN wget -O nvshmem.tar.xz \
    https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.3.9_cuda12-archive.tar.xz \
    && tar -xJf nvshmem.tar.xz \
    && mv libnvshmem-linux-x86_64-3.3.9_cuda12-archive nvshmem \
    && rm nvshmem.tar.xz

ENV NVSHMEM_DIR=/opt/nvshmem
ENV LD_LIBRARY_PATH=/opt/nvshmem/lib:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    rdma-core \
    libibverbs-dev \
    librdmacm-dev \
    ibverbs-providers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/deepseek-ai/DeepEP.git
WORKDIR /app/DeepEP
RUN NVSHMEM_DIR=/opt/nvshmem python3 setup.py install

WORKDIR /app
RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
WORKDIR /app/DeepGEMM
RUN ./install.sh
