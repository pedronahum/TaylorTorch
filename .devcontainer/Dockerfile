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
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgoogle-glog-dev \
    libgtest-dev \
    libomp-18-dev \
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
    libc++-18-dev \
    libc++abi-18-dev \
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

# Install Swift via Swiftly
RUN curl -O https://download.swift.org/swiftly/linux/swiftly-$(uname -m).tar.gz \
    && tar zxf swiftly-$(uname -m).tar.gz \
    && ./swiftly init --quiet-shell-followup \
    && . "${SWIFTLY_HOME_DIR:-/root/.local/share/swiftly}/env.sh" \
    && hash -r \
    && export PATH="/root/.swiftly/bin:$PATH" \
    && swiftly install ${SWIFT_BRANCH:-${SWIFT_VERSION}} --assume-yes --use \
    && . "${SWIFTLY_HOME_DIR:-/root/.local/share/swiftly}/env.sh" \
    && hash -r \
    && SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)" \
    && mkdir -p "${SWIFTLY_HOME_DIR:-/root/.local/share/swiftly}/toolchains" \
    && ln -sfn "${SWIFT_TOOLCHAIN_PATH}" "${SWIFTLY_HOME_DIR:-/root/.local/share/swiftly}/toolchains/current" \
    && swift --version \
    && rm -rf swiftly swiftly-$(uname -m).tar.gz

# Configure Swift environment
ENV SWIFTLY_HOME=/root/.local/share/swiftly
ENV SWIFTLY_BIN=/root/.swiftly/bin
ENV PATH="${SWIFTLY_BIN}:${SWIFTLY_HOME}/toolchains/current/usr/bin:/usr/bin:${PATH}"
ENV SWIFT_TOOLCHAIN_PATH=${SWIFTLY_HOME}/toolchains/current
ENV SWIFT_TOOLCHAIN=${SWIFT_TOOLCHAIN_PATH}
ENV SWIFT_TOOLCHAIN_DIR=${SWIFT_TOOLCHAIN_PATH}/usr
ENV SWIFT_LIB_DIR=${SWIFT_TOOLCHAIN_DIR}/lib/swift
ENV SWIFT_INCLUDE_DIR=${SWIFT_TOOLCHAIN_DIR}/include
ENV SWIFTPM_SWIFT_EXEC=${SWIFT_TOOLCHAIN_DIR}/bin/swift
ENV CC=clang
ENV CXX=clang++
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

# Verify and locate OpenMP installation
RUN echo "Checking for OpenMP headers and libraries..." && \
    CLANG_RESOURCE_DIR="$(clang++ -print-resource-dir)" && \
    echo "Clang resource dir: ${CLANG_RESOURCE_DIR}" && \
    RESOURCE_OMP="${CLANG_RESOURCE_DIR}/include/omp.h" && \
    for candidate in "${RESOURCE_OMP}" \
                     "/usr/lib/llvm-18/lib/clang/18/include/omp.h" \
                     "/usr/lib/llvm-18/include/omp.h" \
                     "/usr/include/omp.h" \
                     "/usr/lib/x86_64-linux-gnu/openmp/include/omp.h"; do \
        if [ -f "${candidate}" ]; then \
            OMP_INCLUDE_DIR="$(dirname "${candidate}")"; \
            echo "✓ Found omp.h at ${candidate}"; \
            break; \
        fi; \
    done && \
    if [ -z "${OMP_INCLUDE_DIR}" ]; then \
        echo "✗ omp.h not found - searching..."; \
        find /usr -name "omp.h" 2>/dev/null | grep -v android || (echo "omp.h not found!" && exit 1); \
    fi && \
    LLVM_BASE_DIR="$(realpath ${CLANG_RESOURCE_DIR}/../../..)" && \
    echo "LLVM base dir: ${LLVM_BASE_DIR}" && \
    OMP_INCLUDE_DIR="$(realpath "${OMP_INCLUDE_DIR}")" && \
    OMP_LIBRARY="" && \
    for candidate in "${LLVM_BASE_DIR}/lib/libomp.so" "/usr/lib/llvm-18/lib/libomp.so" "/usr/lib/x86_64-linux-gnu/libomp.so" "/usr/lib/x86_64-linux-gnu/libomp.so.5"; do \
        if [ -f "${candidate}" ]; then \
            OMP_LIBRARY="${candidate}"; \
            echo "✓ Found libomp at ${candidate}"; \
            break; \
        fi; \
    done && \
    if [ -z "${OMP_LIBRARY}" ]; then \
        echo "✗ libomp.so not found - searching..."; \
        find /usr -name "libomp.so*" 2>/dev/null | grep -v android || (echo "libomp.so not found!" && exit 1); \
    fi && \
    OMP_LIBRARY="$(realpath "${OMP_LIBRARY}")" && \
    echo "Using OpenMP include: ${OMP_INCLUDE_DIR}" && \
    echo "Using OpenMP library: ${OMP_LIBRARY}" && \
    LIBCXX_INCLUDE_DIR="" && \
    PREFERRED_LIBCXX="${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include/c++/v1" && \
    if [ -d "${PREFERRED_LIBCXX}" ]; then \
        LIBCXX_INCLUDE_DIR="$(realpath "${PREFERRED_LIBCXX}")"; \
    else \
        for candidate in "${SWIFT_TOOLCHAIN_DIR}/include/c++/v1" \
                         "${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include/c++/v1" \
                         "${LLVM_BASE_DIR}/lib/swift/clang/include/c++/v1" \
                         "${LLVM_BASE_DIR}/include/c++/v1" \
                         "/usr/include/c++/v1" \
                         "/usr/lib/llvm-18/include/c++/v1" \
                         "/usr/lib/swift/clang/include/c++/v1"; do \
            if [ -d "${candidate}" ] && [ -f "${candidate}/cstddef" ]; then \
                LIBCXX_INCLUDE_DIR="$(realpath "${candidate}")"; \
                break; \
            fi; \
        done; \
    fi && \
    if [ -z "${LIBCXX_INCLUDE_DIR}" ]; then \
        echo "ERROR: Unable to locate libc++ headers (missing cstddef)."; \
        find "${LLVM_BASE_DIR}" -maxdepth 5 -path "*c++/v1/cstddef" 2>/dev/null | head -20 || true; \
        exit 1; \
    fi && \
    LIBCXX_LIBRARY_DIR="" && \
    for candidate in "${SWIFT_TOOLCHAIN_DIR}/lib/swift/linux" "${SWIFT_TOOLCHAIN_DIR}/lib" "${LLVM_BASE_DIR}/lib" "/usr/lib/llvm-18/lib" "/usr/lib/swift/clang/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib"; do \
        if ls "${candidate}/libc++.so"* >/dev/null 2>&1; then \
            LIBCXX_LIBRARY_DIR="$(realpath "${candidate}")"; \
            break; \
        fi; \
    done && \
    if [ -z "${LIBCXX_LIBRARY_DIR}" ]; then \
        echo "ERROR: Unable to locate libc++ libraries."; \
        find "${LLVM_BASE_DIR}" -name "libc++.so*" 2>/dev/null | head -20 || true; \
        exit 1; \
    fi && \
    echo "libc++ include dir: ${LIBCXX_INCLUDE_DIR}" && \
    echo "libc++ library dir: ${LIBCXX_LIBRARY_DIR}" && \
    export USE_LIBCXX=1 && \
    export CPLUS_INCLUDE_PATH="${LIBCXX_INCLUDE_DIR}:${SWIFT_TOOLCHAIN_DIR}/lib/swift:${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include" && \
    export CXXFLAGS="-nostdinc++ -stdlib=libc++ -isystem ${LIBCXX_INCLUDE_DIR}" && \
    export LDFLAGS="-L${LIBCXX_LIBRARY_DIR}" && \
    if [ -n "$LIBRARY_PATH" ]; then \
      export LIBRARY_PATH="${LIBCXX_LIBRARY_DIR}:${LIBRARY_PATH}"; \
    else \
      export LIBRARY_PATH="${LIBCXX_LIBRARY_DIR}"; \
    fi && \
    RESOURCE_INCLUDE="$(realpath "${CLANG_RESOURCE_DIR}/include")" && \
    OPENMP_C_FLAGS="-fopenmp=libomp" && \
    OPENMP_CXX_FLAGS="-fopenmp=libomp" && \
    if [ -n "${OMP_INCLUDE_DIR}" ] && [ "${OMP_INCLUDE_DIR}" != "${RESOURCE_INCLUDE}" ]; then \
      OPENMP_C_FLAGS="${OPENMP_C_FLAGS} -I${OMP_INCLUDE_DIR}"; \
      OPENMP_CXX_FLAGS="${OPENMP_CXX_FLAGS} -I${OMP_INCLUDE_DIR}"; \
    fi && \
    CMAKE_CXX_FLAGS="-stdlib=libc++ -nostdinc++ -isystem ${RESOURCE_INCLUDE} -isystem ${LIBCXX_INCLUDE_DIR}" && \
    cd pytorch && \
    mkdir -p build && \
    cd build && \
    CC=clang CXX=clang++ cmake .. \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_EXTENSIONS=OFF \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
      -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -L${LIBCXX_LIBRARY_DIR}" \
      -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -L${LIBCXX_LIBRARY_DIR}" \
      -DCMAKE_PREFIX_PATH="${LLVM_BASE_DIR};/usr" \
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
RUN cat <<'EOF' >> ~/.bashrc
if [ -f "${SWIFTLY_HOME}/env.sh" ]; then
  . "${SWIFTLY_HOME}/env.sh"
fi
export CC=clang
export CXX=clang++
export PYTORCH_INSTALL_DIR=/opt/pytorch
export LD_LIBRARY_PATH=/opt/pytorch/lib:$LD_LIBRARY_PATH
EOF

# Default command
CMD ["/bin/bash"]
