# TaylorTorch Development Container
# This container has Swift and PyTorch pre-installed for CI builds
#
# Swift Version: Using specific development snapshot via Swiftly
# - Use 'main-snapshot' for the latest nightly, or 'main-snapshot-YYYY-MM-DD' for specific date
# - Current: main-snapshot-2025-11-03 on Ubuntu 24.04
# - Find snapshot names at: https://www.swift.org/download/
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

# Set environment variables
ENV SWIFT_VERSION=${SWIFT_VERSION} \
    PYTORCH_VERSION=v2.8.0 \
    PYTORCH_INSTALL_DIR=/opt/pytorch \
    DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=2

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

# Install Python packages for PyTorch
RUN pip3 install --break-system-packages \
    numpy \
    pyyaml \
    typing_extensions \
    sympy \
    cffi

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

# Set up environment variables for Swift and PyTorch
RUN SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)" && \
    SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_PATH}/usr" && \
    SWIFT_BIN_DIR="${SWIFT_TOOLCHAIN_DIR}/bin" && \
    echo "export PATH=/root/.local/share/swiftly/bin:${SWIFT_BIN_DIR}:\$PATH" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN_DIR=${SWIFT_TOOLCHAIN_DIR}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN_PATH=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFTPM_SWIFT_EXEC=${SWIFT_BIN_DIR}/swift" >> /etc/profile.d/swift.sh && \
    echo "export PYTORCH_INSTALL_DIR=/opt/pytorch" >> /etc/profile.d/swift.sh && \
    echo "export CC=clang" >> /etc/profile.d/swift.sh && \
    echo "export CXX=clang++" >> /etc/profile.d/swift.sh

# Verify OpenMP installation and detect Swift toolchain
RUN . /etc/profile.d/swift.sh && \
    SWIFT_BIN="$(which swift)" && \
    SWIFTLY_BASE="$(dirname $(dirname ${SWIFT_BIN}))" && \
    ACTIVE_TOOLCHAIN=$(cat "${SWIFTLY_BASE}/config.json" | sed -n 's/.*"inUse"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p') && \
    SWIFT_TOOLCHAIN_DIR="${SWIFTLY_BASE}/toolchains/${ACTIVE_TOOLCHAIN}/usr" && \
    CLANG_RESOURCE_DIR="$(${SWIFT_TOOLCHAIN_DIR}/bin/clang++ -print-resource-dir)" && \
    echo "Clang resource dir: ${CLANG_RESOURCE_DIR}" && \
    \
    # Find omp.h header file
    RESOURCE_OMP="${CLANG_RESOURCE_DIR}/include/omp.h" && \
    OMP_INCLUDE_DIR="" && \
    for candidate in \
        "${RESOURCE_OMP}" \
        "/usr/lib/llvm-18/lib/clang/18/include/omp.h" \
        "/usr/lib/llvm-18/include/omp.h" \
        "/usr/lib/gcc/x86_64-linux-gnu/13/include/omp.h" \
        "/usr/include/omp.h"; do \
        if [ -f "${candidate}" ]; then \
            OMP_INCLUDE_DIR="$(dirname "${candidate}")" && \
            echo "✓ Found omp.h at ${candidate}" && \
            break; \
        fi; \
    done && \
    if [ -z "${OMP_INCLUDE_DIR}" ]; then \
        echo "✗ omp.h not found" && exit 1; \
    fi && \
    \
    # Find libomp.so library file
    LLVM_BASE_DIR="$(realpath "${CLANG_RESOURCE_DIR}/../../..")" && \
    OMP_LIBRARY="" && \
    for candidate in \
        "${LLVM_BASE_DIR}/lib/libomp.so" \
        "/usr/lib/llvm-18/lib/libomp.so" \
        "/usr/lib/x86_64-linux-gnu/libomp.so" \
        "/usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so"; do \
        if [ -f "${candidate}" ]; then \
            OMP_LIBRARY="${candidate}" && \
            echo "✓ Found libomp at ${candidate}" && \
            break; \
        fi; \
    done && \
    if [ -z "${OMP_LIBRARY}" ]; then \
        echo "✗ libomp.so not found" && exit 1; \
    fi && \
    \
    # Save OpenMP paths
    OMP_INCLUDE_DIR="$(realpath "${OMP_INCLUDE_DIR}")" && \
    OMP_LIBRARY="$(realpath "${OMP_LIBRARY}")" && \
    echo "export OMP_INCLUDE_DIR=${OMP_INCLUDE_DIR}" >> /etc/profile.d/pytorch.sh && \
    echo "export OMP_LIBRARY=${OMP_LIBRARY}" >> /etc/profile.d/pytorch.sh && \
    echo "export SWIFT_TOOLCHAIN_DIR=${SWIFT_TOOLCHAIN_DIR}" >> /etc/profile.d/pytorch.sh

# Clean up to free disk space before PyTorch build
RUN df -h && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    df -h

# Clone PyTorch repository
RUN git clone --depth=1 --shallow-submodules --branch ${PYTORCH_VERSION} \
    https://github.com/pytorch/pytorch.git /tmp/pytorch || \
    (echo "First clone attempt failed, retrying..." && sleep 10 && \
    git clone --depth=1 --shallow-submodules --branch ${PYTORCH_VERSION} \
    https://github.com/pytorch/pytorch.git /tmp/pytorch)

# Update submodules
RUN cd /tmp/pytorch && \
    git submodule sync && \
    git submodule update --init --depth=1 --recursive

# Configure PyTorch build with libstdc++ (matches install script)
RUN . /etc/profile.d/pytorch.sh && \
    echo "Using OpenMP include: $OMP_INCLUDE_DIR" && \
    echo "Using OpenMP library: $OMP_LIBRARY" && \
    cd /tmp/pytorch && \
    mkdir -p build && \
    cd build && \
    \
    # Isolate OpenMP header if from GCC to avoid header conflicts
    if echo "$OMP_INCLUDE_DIR" | grep -q "/usr/lib/gcc"; then \
        ISOLATED_OMP_DIR="/tmp/taylortorch_omp" && \
        mkdir -p "$ISOLATED_OMP_DIR" && \
        cp "$OMP_INCLUDE_DIR/omp.h" "$ISOLATED_OMP_DIR/" && \
        OMP_INCLUDE_DIR="$ISOLATED_OMP_DIR" && \
        echo "Isolated OpenMP header to: $OMP_INCLUDE_DIR"; \
    fi && \
    \
    # Get compiler paths from Swift toolchain
    CLANG_RESOURCE_DIR="$(${SWIFT_TOOLCHAIN_DIR}/bin/clang++ -print-resource-dir)" && \
    RESOURCE_INCLUDE="$(realpath "${CLANG_RESOURCE_DIR}/include")" && \
    BUILD_CC="${SWIFT_TOOLCHAIN_DIR}/bin/clang" && \
    BUILD_CXX="${SWIFT_TOOLCHAIN_DIR}/bin/clang++" && \
    \
    # OpenMP flags
    OPENMP_C_FLAGS="-fopenmp=libomp" && \
    OPENMP_CXX_FLAGS="-fopenmp=libomp" && \
    if [ -n "$OMP_INCLUDE_DIR" ] && [ "$OMP_INCLUDE_DIR" != "$RESOURCE_INCLUDE" ]; then \
        OPENMP_C_FLAGS="$OPENMP_C_FLAGS -isystem ${OMP_INCLUDE_DIR}" && \
        OPENMP_CXX_FLAGS="$OPENMP_CXX_FLAGS -isystem ${OMP_INCLUDE_DIR}"; \
    fi && \
    \
    # Use libstdc++ (GCC's C++ standard library) - NOT libc++
    # This is critical for Swift/C++ interoperability on Linux
    CMAKE_CXX_FLAGS="-isystem ${RESOURCE_INCLUDE}" && \
    CMAKE_C_FLAGS="-isystem ${RESOURCE_INCLUDE}" && \
    \
    echo "=== Build Configuration ===" && \
    echo "Compiler: ${BUILD_CC}" && \
    echo "C++ Compiler: ${BUILD_CXX}" && \
    echo "Using libstdc++ (GCC's C++ standard library)" && \
    echo "===========================" && \
    \
    CC="${BUILD_CC}" CXX="${BUILD_CXX}" cmake .. \
        -DCMAKE_C_COMPILER="${BUILD_CC}" \
        -DCMAKE_CXX_COMPILER="${BUILD_CXX}" \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_EXTENSIONS=OFF \
        -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
        -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="" \
        -DCMAKE_SHARED_LINKER_FLAGS="" \
        -DCMAKE_PREFIX_PATH="/usr" \
        -DUSE_LIBCXX=OFF \
        -DOpenMP_C_FLAGS="${OPENMP_C_FLAGS}" \
        -DOpenMP_CXX_FLAGS="${OPENMP_CXX_FLAGS}" \
        -DOpenMP_C_LIB_NAMES="libomp" \
        -DOpenMP_CXX_LIB_NAMES="libomp" \
        -DOpenMP_libomp_LIBRARY="${OMP_LIBRARY}" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/pytorch \
        -DBUILD_PYTHON=OFF \
        -DBUILD_TEST=OFF \
        -DBUILD_CAFFE2=OFF \
        -DUSE_DISTRIBUTED=OFF \
        -DUSE_MPS=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_XNNPACK=OFF \
        -DUSE_QNNPACK=OFF \
        -DUSE_FBGEMM=OFF \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -GNinja

# Build PyTorch
RUN set -ex && \
    cd /tmp/pytorch/build && \
    echo "Starting PyTorch build with MAX_JOBS=${MAX_JOBS}..." && \
    cmake --build . --target install -j${MAX_JOBS} 2>&1 | tee /tmp/pytorch-build.log || \
    (echo "=== BUILD FAILED ===" && \
    echo "Last 100 lines of build log:" && \
    tail -n 100 /tmp/pytorch-build.log && \
    exit 1)

# Cleanup after PyTorch build
RUN rm -rf /tmp/pytorch /tmp/pytorch-build.log && \
    echo "PyTorch installation complete" && \
    echo "Verifying PyTorch library installation:" && \
    ls -lh /opt/pytorch/lib/ | head -20

# Set up PyTorch library paths
RUN echo "/opt/pytorch/lib" > /etc/ld.so.conf.d/pytorch.conf && \
    ldconfig && \
    echo "export LD_LIBRARY_PATH=/opt/pytorch/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/swift.sh

# Verify installations
RUN . /etc/profile.d/swift.sh && swift --version && \
    ls -lh /opt/pytorch/lib/libtorch.so && \
    echo "✅ Swift and PyTorch successfully installed"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash", "-l"]
