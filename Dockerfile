# TaylorTorch Development Container
# This container has Swift and PyTorch pre-installed for CI builds
#
# Swift Version: Using specific development snapshot via Swiftly
# - Use 'main-snapshot' for the latest nightly, or 'main-snapshot-YYYY-MM-DD' for specific date
# - Current: main-snapshot-2025-11-03 on Ubuntu 24.04
# - Find snapshot names at: https://www.swift.org/download/
#
# PyTorch: Installed via pip for optimized MKL/MKLDNN binaries (~10x faster than source builds)
#
# Build with default snapshot:
#   docker build -t taylortorch-dev .
#
# Build with specific snapshot:
#   docker build --build-arg SWIFT_VERSION="swift-DEVELOPMENT-SNAPSHOT-2025-10-20-a" -t taylortorch-dev .
#
# Build with latest nightly:
#   docker build --build-arg SWIFT_VERSION="main-snapshot" -t taylortorch-dev .
FROM ubuntu:24.04

# Swift snapshot configuration
# Swiftly will handle downloading the correct snapshot for the platform
ARG SWIFT_VERSION="main-snapshot-2025-11-03"
ARG PYTORCH_VERSION="2.8.0"

# Set environment variables
ENV SWIFT_VERSION=${SWIFT_VERSION} \
    PYTORCH_VERSION=${PYTORCH_VERSION} \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Swift + PyTorch build deps)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-yaml \
    wget \
    ca-certificates \
    binutils \
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
    tzdata \
    unzip \
    zip \
    zlib1g-dev \
    libicu-dev \
    libssl-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    libprotobuf-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    libgflags-dev \
    ninja-build \
    llvm-18-dev \
    libc++-18-dev \
    libc++abi-18-dev \
    libomp-18-dev \
    && rm -rf /var/lib/apt/lists/*

# Remove LLVM 17 to prevent version conflicts
RUN apt-get update && \
    apt-get remove -y llvm-17-dev libc++-17-dev libc++abi-17-dev libomp-17-dev llvm-17 'llvm-17-*' 2>/dev/null || true && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages and PyTorch via pip
# Using pip-installed PyTorch provides optimized MKL/MKLDNN binaries (~10x faster than source builds)
RUN pip3 install --break-system-packages \
    numpy \
    pyyaml \
    typing_extensions \
    sympy \
    cffi \
    torch==${PYTORCH_VERSION}

# Detect PyTorch installation path and set up environment
RUN PYTORCH_INSTALL_DIR=$(python3 -c "import torch; print(torch.__path__[0])") && \
    echo "PyTorch installed at: ${PYTORCH_INSTALL_DIR}" && \
    echo "export PYTORCH_INSTALL_DIR=${PYTORCH_INSTALL_DIR}" >> /etc/profile.d/pytorch.sh && \
    echo "export LD_LIBRARY_PATH=${PYTORCH_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/pytorch.sh && \
    # Verify MKL/MKLDNN support
    python3 -c "import torch; config = torch.__config__.show(); print('MKL:', 'USE_MKL=ON' in config); print('MKLDNN:', 'USE_MKLDNN=ON' in config)"

# Install Swiftly (Swift toolchain manager) using the new manual method
RUN ARCH=$(uname -m) && \
    curl -f -L -O "https://download.swift.org/swiftly/linux/swiftly-${ARCH}.tar.gz" && \
    tar zxf "swiftly-${ARCH}.tar.gz" && \
    ./swiftly init --quiet-shell-followup && \
    rm "swiftly-${ARCH}.tar.gz"

# Add swiftly to the PATH
ENV PATH="/root/.local/share/swiftly/bin:$PATH"

# Install the specific Swift snapshot using Swiftly
RUN echo "Installing Swift snapshot: $SWIFT_VERSION" && \
    swiftly install "$SWIFT_VERSION" && \
    swiftly use "$SWIFT_VERSION" && \
    echo "Swift installation complete:" && \
    swift --version

# Set up environment variables for Swift
RUN SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)" && \
    SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_PATH}/usr" && \
    SWIFT_BIN_DIR="${SWIFT_TOOLCHAIN_DIR}/bin" && \
    echo "export PATH=/root/.local/share/swiftly/bin:${SWIFT_BIN_DIR}:\$PATH" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN_DIR=${SWIFT_TOOLCHAIN_DIR}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN_PATH=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFTPM_SWIFT_EXEC=${SWIFT_BIN_DIR}/swift" >> /etc/profile.d/swift.sh && \
    echo "export CC=clang" >> /etc/profile.d/swift.sh && \
    echo "export CXX=clang++" >> /etc/profile.d/swift.sh && \
    # Add SWIFT_TOOLCHAIN_DIR to pytorch.sh as well
    echo "export SWIFT_TOOLCHAIN_DIR=${SWIFT_TOOLCHAIN_DIR}" >> /etc/profile.d/pytorch.sh

# Verify installations
RUN . /etc/profile.d/swift.sh && \
    . /etc/profile.d/pytorch.sh && \
    swift --version && \
    echo "PyTorch install dir: $PYTORCH_INSTALL_DIR" && \
    ls -lh $PYTORCH_INSTALL_DIR/lib/libtorch.so && \
    echo "✅ Swift and PyTorch successfully installed"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash", "-l"]
