FROM ubuntu:25.10

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        htop \
        jq \
        libboost-dev \
        libbz2-dev \
        libconfig-dev \
        libcurl4-openssl-dev \
        libffi-dev \
        libfuse-dev \
        libhdf5-dev \
        libjson-c-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libnl-3-dev \
        libnuma-dev \
        libreadline-dev \
        libsensors-dev \
        libsqlite3-dev \
        libssl-dev \
        libtool \
        libuv1-dev \
        libyaml-dev \
        llvm \
        gfortran-12 \
        gcc-12 \
        g++-12 \
        pkg-config \
        python3 \
        strace \
        tar \
        tk-dev \
        wget \
        xz-utils \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-12 100

ENV CC=gcc
ENV CXX=g++
ENV FC=gfortran
ENV CUDAHOSTCXX=g++

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version && which rustc && cargo --version && which cargo

# Install Bencher for performance monitoring
# Update the following comment to trigger a rebuild to update the CLI:
# last update: 2026-5-11
# This is necessary because the cloud version and the CLI version have to match
# but obviously, version changes do not register in the Dockerfile hash.
RUN curl --proto '=https' --tlsv1.2 -sSfL https://bencher.dev/download/install-cli.sh | sh
RUN bencher --version && which bencher

# Install NVIDIA HPC SDK for nvfortran
ARG ARCH=aarch64
ARG HPC_SDK_VERSION=24.11
ARG HPC_SDK_NAME=nvhpc_2024_2411_Linux_${ARCH}_cuda_12.6
ENV HPC_SDK_URL=https://developer.download.nvidia.com/hpc-sdk/${HPC_SDK_VERSION}/${HPC_SDK_NAME}.tar.gz

ENV NVHPC_SILENT=true
ENV NVHPC_INSTALL_DIR=/opt/nvidia/hpc_sdk
ENV NVHPC_INSTALL_TYPE=single
RUN wget -q ${HPC_SDK_URL} -O /tmp/nvhpc.tar.gz && \
    mkdir -p /opt/nvidia && \
    tar -xzf /tmp/nvhpc.tar.gz -C /opt/nvidia && \
    rm /tmp/nvhpc.tar.gz && \
    cd /opt/nvidia/${HPC_SDK_NAME} && ./install && \
    rm -rf /opt/nvidia/${HPC_SDK_NAME}

ENV HPC_SDK_PATH=/opt/nvidia/hpc_sdk/Linux_${ARCH}/${HPC_SDK_VERSION}
ENV CUDA_PATH=${HPC_SDK_PATH}/cuda

ENV PATH=${HPC_SDK_PATH}/compilers/bin:${PATH} \
    MANPATH=${HPC_SDK_PATH}/compilers/man:${MANPATH} \
    LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${HPC_SDK_PATH}/math_libs/lib64:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=${CUDA_PATH}/lib64:${HPC_SDK_PATH}/math_libs/lib64:${LIBRARY_PATH}

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install OpenMPI configured with libfabric, libcxi, and gdrcopy support for use
# on Alps. This is based on examples in
# https://github.com/eth-cscs/cray-network-stack.
ARG gdrcopy_version=2.5.1
RUN set -eux; \
    git clone --depth 1 --branch "v${gdrcopy_version}" https://github.com/NVIDIA/gdrcopy.git; \
    cd gdrcopy; \
    make lib -j"$(nproc)" lib_install; \
    cd /; \
    rm -rf /gdrcopy; \
    ldconfig

ARG cassini_headers_version=release/shs-13.0.0
RUN set -eux; \
    git clone --depth 1 --branch "${cassini_headers_version}" https://github.com/HewlettPackard/shs-cassini-headers.git; \
    cd shs-cassini-headers; \
    cp -r include/* /usr/include/; \
    cp -r share/* /usr/share/; \
    rm -rf /shs-cassini-headers

ARG cxi_driver_version=release/shs-13.0.0
RUN set -eux; \
    git clone --depth 1 --branch "${cxi_driver_version}" https://github.com/HewlettPackard/shs-cxi-driver.git; \
    cd shs-cxi-driver; \
    cp -r include/* /usr/include/; \
    rm -rf /shs-cxi-driver

ARG libcxi_version=release/shs-13.0.0
RUN set -eux; \
    git clone --depth 1 --branch "${libcxi_version}" https://github.com/HewlettPackard/shs-libcxi.git; \
    cd shs-libcxi; \
    ./autogen.sh; \
    ./configure \
      --with-cuda=${CUDA_PATH}; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf /shs-libcxi; \
    ldconfig

ARG xpmem_version=0d0bad4e1d07b38d53ecc8f20786bb1328c446da
RUN set -eux; \
    git clone https://github.com/hpc/xpmem.git; \
    cd xpmem; \
    git checkout "${xpmem_version}"; \
    ./autogen.sh; \
    ./configure --disable-kernel-module; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf /xpmem; \
    ldconfig

# NOTE: xpmem is not found correctly without setting the prefix explicitly in
# --enable-xpmem
ARG libfabric_version=v2.4.0
RUN set -eux; \
    git clone --depth 1 --branch "${libfabric_version}" https://github.com/ofiwg/libfabric.git; \
    cd libfabric; \
    ./autogen.sh; \
    ./configure \
      --with-cuda=${CUDA_PATH} \
      --enable-cuda-dlopen \
      --enable-xpmem=/usr \
      --enable-tcp \
      --enable-cxi; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf /libfabric; \
    ldconfig

ARG openmpi_version=5.0.9
RUN set -eux; \
    curl -fsSL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${openmpi_version}.tar.gz" -o /tmp/ompi.tar.gz; \
    tar -C /tmp -xzf /tmp/ompi.tar.gz; \
    cd "/tmp/openmpi-${openmpi_version}"; \
    ./configure \
      --with-ofi \
      --with-cuda=${CUDA_PATH}; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf "/tmp/openmpi-${openmpi_version}" /tmp/ompi.tar.gz; \
    ldconfig

# Install uv: https://docs.astral.sh/uv/guides/integration/docker
COPY --from=ghcr.io/astral-sh/uv:0.11.15@sha256:e590846f4776907b254ac0f44b5b380347af5d90d668138ca7938d1b0c2f98d3 /uv /uvx /bin/
