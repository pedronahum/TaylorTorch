FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SWIFT_SIGNING_KEY=A62AE125BBBFBB96A6E042EC925CC1CCED3D1561
ENV SWIFT_PLATFORM=ubuntu24.04
ENV SWIFT_BRANCH=swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a
ENV SWIFT_VERSION=swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a

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
    python3-lldb \
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
    libomp-17-dev \
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
    libc++-17-dev \
    libc++abi-17-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages needed for PyTorch build
RUN pip3 install --no-cache-dir --break-system-packages \
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

# Set up all environment variables early (before building PyTorch)
ENV PATH="/usr/bin:${PATH}"
ENV CC=clang
ENV CXX=clang++
ENV SWIFT_TOOLCHAIN_DIR=/usr
ENV SWIFT_LIB_DIR=/usr/lib/swift
ENV SWIFT_INCLUDE_DIR=/usr/include
ENV SWIFTPM_SWIFT_EXEC=/usr/bin/swift
ENV PYTORCH_INSTALL_DIR=/opt/pytorch
ENV PYTORCH_ROOT=/opt/pytorch
ENV PYTORCH_VERSION=v2.8.0

# Verify Swift installation
RUN swift --version

# Build PyTorch from source
WORKDIR /tmp/pytorch-build

# Clone and checkout PyTorch
RUN git clone --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git checkout ${PYTORCH_VERSION} && \
    git submodule sync && \
    git submodule update --init --recursive

# Configure and build PyTorch with CMake using Swift's clang/clang++
RUN cd pytorch && \
    mkdir -p build && \
    cd build && \
    CC=clang CXX=clang++ \
    CFLAGS="-I/usr/lib/llvm-17/include" \
    CXXFLAGS="-I/usr/lib/llvm-17/include" \
    cmake .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS="-I/usr/lib/llvm-17/include" \
    -DCMAKE_CXX_FLAGS="-I/usr/lib/llvm-17/include" \
    -DOpenMP_C_FLAGS="-fopenmp=libomp -I/usr/lib/llvm-17/include" \
    -DOpenMP_CXX_FLAGS="-fopenmp=libomp -I/usr/lib/llvm-17/include" \
    -DOpenMP_C_LIB_NAMES="libomp" \
    -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/usr/lib/llvm-17/lib/libomp.so" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/pytorch \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TEST=OFF \
    -DBUILD_CAFFE2=OFF \
    -DUSE_DISTRIBUTED=OFF \
    -DUSE_MPS=OFF \
    -DUSE_MKLDNN=OFF \
    -DUSE_XNNPACK=OFF \
    -DUSE_QNNPACK=OFF \
    -DUSE_FBGEMM=OFF \  
    -DUSE_CUDA=OFF \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -GNinja && \
    cmake --build . --target install -j$(nproc)

# Set up library paths after PyTorch is built
ENV LD_LIBRARY_PATH=${PYTORCH_ROOT}/lib:${LD_LIBRARY_PATH}
ENV CMAKE_PREFIX_PATH=${PYTORCH_ROOT}:${CMAKE_PREFIX_PATH}

# Clean up build directory to save space
RUN rm -rf /tmp/pytorch-build

# Create a non-root user for development
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER vscode

# Set up shell for the user with all necessary environment variables
RUN echo 'export CC=clang' >> ~/.bashrc && \
    echo 'export CXX=clang++' >> ~/.bashrc && \
    echo 'export PATH=/usr/bin:$PATH' >> ~/.bashrc && \
    echo 'export PYTORCH_INSTALL_DIR=/opt/pytorch' >> ~/.bashrc && \
    echo 'export SWIFT_TOOLCHAIN_DIR=/usr' >> ~/.bashrc && \
    echo 'export SWIFTPM_SWIFT_EXEC=/usr/bin/swift' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/opt/pytorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Default command
CMD ["/bin/bash"]
