# TaylorTorch Development Container
# This container has Swift and PyTorch pre-installed for CI builds
#
# Swift Version: Using specific development snapshot via Swiftly
# - Use 'main-snapshot' for the latest nightly, or a specific date-stamped snapshot
# - Current: swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a on Ubuntu 24.04
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
ARG SWIFT_VERSION="swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a"

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
    clang-17 \
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
    libgflags-dev \
    libc++-17-dev \
    libc++abi-17-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for PyTorch
RUN pip3 install --break-system-packages \
    numpy \
    pyyaml \
    typing_extensions \
    sympy \
    cffi 

# Install Swiftly (Swift toolchain manager)
# Install Swiftly (Swift toolchain manager) using the new manual method
RUN ARCH=$(uname -m) && \
    curl -f -L -O "https://download.swift.org/swiftly/linux/swiftly-${ARCH}.tar.gz" && \
    tar zxf "swiftly-${ARCH}.tar.gz" && \
    ./swiftly init --quiet-shell-followup && \
    rm "swiftly-${ARCH}.tar.gz"

# Add swiftly to the PATH. **Note: The install path is different!**
ENV PATH="/root/.local/share/swiftly/bin:$PATH"

# Install the specific Swift snapshot using Swiftly
# Swiftly automatically downloads the correct snapshot for ubuntu24.04
RUN echo "Installing Swift snapshot: $SWIFT_VERSION" && \
    swiftly install "$SWIFT_VERSION" && \
    swiftly use "$SWIFT_VERSION" && \
    echo "Swift installation complete:" && \
    swift --version

# Set up environment variables for Swift and PyTorch
RUN SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)" && \
    SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_PATH}/usr" && \
    SWIFT_BIN_DIR="${SWIFT_TOOLCHAIN_DIR}/bin" && \
    # echo "export PATH=/root/.swiftly/bin:${SWIFT_BIN_DIR}:\$PATH" >> /etc/profile.d/swift.sh && \
    # New, correct line
    echo "export PATH=/root/.local/share/swiftly/bin:${SWIFT_BIN_DIR}:\$PATH" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN_DIR=${SWIFT_TOOLCHAIN_DIR}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN_PATH=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFT_TOOLCHAIN=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    echo "export SWIFTPM_SWIFT_EXEC=${SWIFT_BIN_DIR}/swift" >> /etc/profile.d/swift.sh && \
    echo "export PYTORCH_INSTALL_DIR=/opt/pytorch" >> /etc/profile.d/swift.sh && \
    echo "export CC=clang" >> /etc/profile.d/swift.sh && \
    echo "export CXX=clang++" >> /etc/profile.d/swift.sh

# Verify OpenMP installation (required for PyTorch parallel operations)
# PyTorch needs OpenMP for CPU parallelization, so we locate the headers and library
RUN CLANG_RESOURCE_DIR="$(clang++ -print-resource-dir)" && \
    echo "Clang resource dir: ${CLANG_RESOURCE_DIR}" && \
    \
    # Step 1: Find omp.h header file
    # Check common locations where OpenMP headers might be installed
    RESOURCE_OMP="${CLANG_RESOURCE_DIR}/include/omp.h" && \
    OMP_INCLUDE_DIR="" && \
    for candidate in \
    "${RESOURCE_OMP}" \
    "/usr/lib/llvm-17/lib/clang/17/include/omp.h" \
    "/usr/lib/llvm-17/include/omp.h" \
    "/usr/include/omp.h" \
    "/usr/lib/x86_64-linux-gnu/openmp/include/omp.h"; do \
    if [ -f "${candidate}" ]; then \
    OMP_INCLUDE_DIR="$(dirname "${candidate}")" && \
    echo "✓ Found omp.h at ${candidate}" && \
    break; \
    fi; \
    done && \
    if [ -z "${OMP_INCLUDE_DIR}" ]; then \
    echo "✗ omp.h not found - searching system..." && \
    find /usr -name "omp.h" 2>/dev/null | grep -v android || echo "omp.h not found!" && \
    exit 1; \
    fi && \
    \
    # Step 2: Find libomp.so library file
    # Check common locations where OpenMP runtime library might be installed
    LLVM_BASE_DIR="$(realpath "${CLANG_RESOURCE_DIR}/../../..")" && \
    echo "LLVM base dir: ${LLVM_BASE_DIR}" && \
    OMP_LIBRARY="" && \
    for candidate in \
    "${LLVM_BASE_DIR}/lib/libomp.so" \
    "/usr/lib/llvm-17/lib/libomp.so" \
    "/usr/lib/x86_64-linux-gnu/libomp.so" \
    "/usr/lib/x86_64-linux-gnu/libomp.so.5"; do \
    if [ -f "${candidate}" ]; then \
    OMP_LIBRARY="${candidate}" && \
    echo "✓ Found libomp at ${candidate}" && \
    break; \
    fi; \
    done && \
    if [ -z "${OMP_LIBRARY}" ]; then \
    echo "✗ libomp.so not found - searching system..." && \
    find /usr -name "libomp.so*" 2>/dev/null | grep -v android || echo "libomp.so not found!" && \
    exit 1; \
    fi && \
    \
    # Step 3: Save OpenMP paths to environment file for PyTorch build
    OMP_INCLUDE_DIR="$(realpath "${OMP_INCLUDE_DIR}")" && \
    OMP_LIBRARY="$(realpath "${OMP_LIBRARY}")" && \
    echo "export OMP_INCLUDE_DIR=${OMP_INCLUDE_DIR}" >> /etc/profile.d/pytorch.sh && \
    echo "export OMP_LIBRARY=${OMP_LIBRARY}" >> /etc/profile.d/pytorch.sh

# Clean up to free disk space before PyTorch build
RUN df -h && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    df -h

# Stage 1: Clone PyTorch repository at specific version with retry logic
# Clone the specific tag directly with --branch to avoid checkout issues
RUN git clone --depth=1 --shallow-submodules --branch ${PYTORCH_VERSION} \
    https://github.com/pytorch/pytorch.git /tmp/pytorch || \
    (echo "First clone attempt failed, retrying..." && sleep 10 && \
    git clone --depth=1 --shallow-submodules --branch ${PYTORCH_VERSION} \
    https://github.com/pytorch/pytorch.git /tmp/pytorch)

# Stage 2: Update submodules (checkout is already done by the clone)
RUN cd /tmp/pytorch && \
    git submodule sync && \
    git submodule update --init --depth=1 --recursive

# Stage 3: Configure PyTorch build with libc++ detection
# This is complex because PyTorch and Swift need to use the same C++ standard library
RUN . /etc/profile.d/pytorch.sh && \
    echo "Using OpenMP include: $OMP_INCLUDE_DIR" && \
    echo "Using OpenMP library: $OMP_LIBRARY" && \
    cd /tmp/pytorch && \
    mkdir -p build && \
    cd build && \
    \
    # Step 2: Locate compiler and toolchain directories
    # We need to know where clang and Swift are installed to configure the build
    CLANG_RESOURCE_DIR="$(clang++ -print-resource-dir)" && \
    LLVM_BASE_DIR="$(realpath "${CLANG_RESOURCE_DIR}/../../..")" && \
    echo "Clang resource dir: ${CLANG_RESOURCE_DIR}" && \
    echo "LLVM base dir: ${LLVM_BASE_DIR}" && \
    SWIFT_BIN="$(which swift)" && \
    SWIFT_TOOLCHAIN_DIR="$(dirname $(dirname ${SWIFT_BIN}))" && \
    echo "Swift toolchain dir: ${SWIFT_TOOLCHAIN_DIR}" && \
    \
    # Step 3: Find libc++ (C++ standard library)
    # TaylorTorch uses Swift C++ interop, so PyTorch must use the same C++ stdlib as Swift
    # Try Swift's bundled libc++ first, then fall back to system LLVM 17
    LIBCXX_INCLUDE_DIR="" && \
    LIBCXX_SOURCE="unknown" && \
    for candidate in \
    "${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include/c++/v1" \
    "${SWIFT_TOOLCHAIN_DIR}/include/c++/v1" \
    "${SWIFT_TOOLCHAIN_DIR}/lib/swift_static/clang/include/c++/v1" \
    "${CLANG_RESOURCE_DIR}/../../../include/c++/v1"; do \
    if [ -d "${candidate}" ] && [ -f "${candidate}/cstddef" ]; then \
    LIBCXX_INCLUDE_DIR="$(realpath "${candidate}")" && \
    LIBCXX_SOURCE="Swift toolchain" && \
    echo "✓ Found libc++ in Swift toolchain at ${LIBCXX_INCLUDE_DIR}" && \
    break; \
    fi; \
    done && \
    if [ -z "${LIBCXX_INCLUDE_DIR}" ]; then \
    echo "ℹ Swift toolchain doesn't include libc++, using system LLVM 17" && \
    for candidate in \
    "/usr/lib/llvm-17/include/c++/v1" \
    "/usr/include/c++/v1"; do \
    if [ -d "${candidate}" ] && [ -f "${candidate}/cstddef" ]; then \
    LIBCXX_INCLUDE_DIR="$(realpath "${candidate}")" && \
    LIBCXX_SOURCE="system LLVM 17" && \
    echo "✓ Found libc++ in system at ${LIBCXX_INCLUDE_DIR}" && \
    break; \
    fi; \
    done; \
    fi && \
    if [ -z "${LIBCXX_INCLUDE_DIR}" ]; then \
    echo "ERROR: Unable to locate libc++ headers." && \
    echo "Please ensure libc++-17-dev is installed." && \
    exit 1; \
    fi && \
    LIBCXX_LIBRARY_DIR="" && \
    if [ "$LIBCXX_SOURCE" = "Swift toolchain" ]; then \
    for candidate in \
    "${SWIFT_TOOLCHAIN_DIR}/lib/swift/linux" \
    "${SWIFT_TOOLCHAIN_DIR}/lib"; do \
    if ls "${candidate}/libc++.so"* >/dev/null 2>&1; then \
    LIBCXX_LIBRARY_DIR="$(realpath "${candidate}")" && \
    echo "✓ Found libc++ library in Swift toolchain at ${LIBCXX_LIBRARY_DIR}" && \
    break; \
    fi; \
    done; \
    fi && \
    if [ -z "${LIBCXX_LIBRARY_DIR}" ]; then \
    for candidate in \
    "/usr/lib/llvm-17/lib" \
    "/usr/lib/x86_64-linux-gnu" \
    "/usr/lib"; do \
    if ls "${candidate}/libc++.so"* >/dev/null 2>&1; then \
    LIBCXX_LIBRARY_DIR="$(realpath "${candidate}")" && \
    echo "✓ Found libc++ library at ${LIBCXX_LIBRARY_DIR}" && \
    break; \
    fi; \
    done; \
    fi && \
    if [ -z "${LIBCXX_LIBRARY_DIR}" ]; then \
    echo "ERROR: Unable to locate libc++ libraries." && \
    echo "Please ensure libc++-17-dev is installed." && \
    exit 1; \
    fi && \
    echo "libc++ include dir: ${LIBCXX_INCLUDE_DIR}" && \
    echo "libc++ library dir: ${LIBCXX_LIBRARY_DIR}" && \
    export USE_LIBCXX=1 && \
    if [ "$LIBCXX_SOURCE" = "Swift toolchain" ]; then \
    export CPLUS_INCLUDE_PATH="${SWIFT_TOOLCHAIN_DIR}/lib/swift:${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include"; \
    fi && \
    export CXXFLAGS="-stdlib=libc++" && \
    export LDFLAGS="-L${LIBCXX_LIBRARY_DIR}" && \
    if [ -n "$LIBRARY_PATH" ]; then \
    export LIBRARY_PATH="${LIBCXX_LIBRARY_DIR}:${LIBRARY_PATH}"; \
    else \
    export LIBRARY_PATH="${LIBCXX_LIBRARY_DIR}"; \
    fi && \
    OPENMP_C_FLAGS="-fopenmp=libomp" && \
    OPENMP_CXX_FLAGS="-fopenmp=libomp" && \
    RESOURCE_INCLUDE="$(realpath "${CLANG_RESOURCE_DIR}/include")" && \
    if [ -n "$OMP_INCLUDE_DIR" ] && [ "$OMP_INCLUDE_DIR" != "$RESOURCE_INCLUDE" ]; then \
    OPENMP_C_FLAGS="$OPENMP_C_FLAGS -I${OMP_INCLUDE_DIR}" && \
    OPENMP_CXX_FLAGS="$OPENMP_CXX_FLAGS -I${OMP_INCLUDE_DIR}"; \
    fi && \
    if [ "$LIBCXX_SOURCE" = "Swift toolchain" ]; then \
    BUILD_CC="${SWIFT_TOOLCHAIN_DIR}/bin/clang" && \
    BUILD_CXX="${SWIFT_TOOLCHAIN_DIR}/bin/clang++" && \
    BUILD_RESOURCE_INCLUDE="${RESOURCE_INCLUDE}" && \
    echo "✓ Using Swift's clang for PyTorch build"; \
    else \
    if [ -f "/usr/lib/llvm-17/bin/clang" ] && [ -f "/usr/lib/llvm-17/bin/clang++" ]; then \
    BUILD_CC="/usr/lib/llvm-17/bin/clang" && \
    BUILD_CXX="/usr/lib/llvm-17/bin/clang++" && \
    BUILD_RESOURCE_INCLUDE="/usr/lib/llvm-17/lib/clang/17/include" && \
    echo "✓ Using system clang-17 for PyTorch build (matches libc++ 17)"; \
    else \
    echo "⚠ System clang-17 not found, falling back to Swift's clang" && \
    BUILD_CC="${SWIFT_TOOLCHAIN_DIR}/bin/clang" && \
    BUILD_CXX="${SWIFT_TOOLCHAIN_DIR}/bin/clang++" && \
    BUILD_RESOURCE_INCLUDE="${RESOURCE_INCLUDE}"; \
    fi; \
    fi && \
    CMAKE_CXX_FLAGS="-stdlib=libc++" && \
    if [ "$LIBCXX_SOURCE" = "Swift toolchain" ]; then \
    CMAKE_PREFIX_PATH="${LLVM_BASE_DIR};/usr"; \
    else \
    CMAKE_PREFIX_PATH="/usr/lib/llvm-17;/usr"; \
    fi && \
    echo "=== Build Configuration ===" && \
    echo "Compiler: ${BUILD_CC}" && \
    echo "C++ Compiler: ${BUILD_CXX}" && \
    echo "libc++ Source: ${LIBCXX_SOURCE}" && \
    echo "libc++ Include: ${LIBCXX_INCLUDE_DIR}" && \
    echo "libc++ Library: ${LIBCXX_LIBRARY_DIR}" && \
    echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}" && \
    echo "===========================" && \
    CC="${BUILD_CC}" CXX="${BUILD_CXX}" cmake .. \
    -DCMAKE_C_COMPILER="${BUILD_CC}" \
    -DCMAKE_CXX_COMPILER="${BUILD_CXX}" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -L${LIBCXX_LIBRARY_DIR} -Wl,-rpath,${LIBCXX_LIBRARY_DIR}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${LIBCXX_LIBRARY_DIR} -Wl,-rpath,${LIBCXX_LIBRARY_DIR}" \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    -DUSE_LIBCXX=ON \
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
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -GNinja

# Stage 4: Build PyTorch (this is the resource-intensive step)
# Using MAX_JOBS=2 to avoid OOM on GitHub Actions runners (7GB RAM limit)
# Adding error logging to capture build failures
RUN set -ex && \
    cd /tmp/pytorch/build && \
    echo "Starting PyTorch build with MAX_JOBS=${MAX_JOBS}..." && \
    cmake --build . --target install -j${MAX_JOBS} 2>&1 | tee /tmp/pytorch-build.log || \
    (echo "=== BUILD FAILED ===" && \
    echo "Last 100 lines of build log:" && \
    tail -n 100 /tmp/pytorch-build.log && \
    exit 1)

# Stage 5: Cleanup after PyTorch build
# cmake --target install already installed to /opt/pytorch, so just cleanup
RUN rm -rf /tmp/pytorch /tmp/pytorch-build.log && \
    echo "PyTorch installation complete" && \
    echo "Verifying PyTorch library installation:" && \
    ls -lh /opt/pytorch/lib/ | head -20

# Try with pre-build torch version
# RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip && \
#    unzip libtorch-shared-with-deps-latest.zip -d /opt/ && mv /opt/libtorch /opt/pytorch

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
