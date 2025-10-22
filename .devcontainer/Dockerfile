FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SWIFT_SIGNING_KEY=A62AE125BBBFBB96A6E042EC925CC1CCED3D1561
ENV SWIFT_PLATFORM=ubuntu24.04
ENV SWIFT_BRANCH=swift-DEVELOPMENT-SNAPSHOT-2025-01-21-a
ENV SWIFT_VERSION=swift-DEVELOPMENT-SNAPSHOT-2025-01-21-a

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    binutils \
    git \
    gnupg2 \
    libc6-dev \
    libcurl4-openssl-dev \
    libedit2 \
    libgcc-13-dev \
    libpython3-dev \
    libsqlite3-0 \
    libstdc++-13-dev \
    libxml2-dev \
    libz3-dev \
    pkg-config \
    python3-lldb-13 \
    tzdata \
    unzip \
    zip \
    zlib1g-dev \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    libprotobuf-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-yaml \
    libgflags-dev \
    libc++-dev \
    libc++abi-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages needed for PyTorch build
RUN pip3 install --no-cache-dir \
    numpy \
    pyyaml \
    typing_extensions \
    sympy \
    cffi \
    cmake

# Download and install Swift main snapshot
RUN SWIFT_URL=https://download.swift.org/development/ubuntu2404/${SWIFT_BRANCH}/${SWIFT_VERSION}-${SWIFT_PLATFORM}.tar.gz \
    && wget -q $SWIFT_URL \
    && tar xzf ${SWIFT_VERSION}-${SWIFT_PLATFORM}.tar.gz -C / --strip-components=1 \
    && rm ${SWIFT_VERSION}-${SWIFT_PLATFORM}.tar.gz \
    && chmod -R o+r /usr/lib/swift

# Set up Swift environment
ENV PATH="/usr/bin:${PATH}"
ENV CC=clang
ENV CXX=clang++

# Verify Swift installation
RUN swift --version

# Create a non-root user for development
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER vscode

# Set up shell for the user
RUN echo 'export CC=clang' >> ~/.bashrc && \
    echo 'export CXX=clang++' >> ~/.bashrc && \
    echo 'export PATH=/usr/bin:$PATH' >> ~/.bashrc

# Default command
CMD ["/bin/bash"]
