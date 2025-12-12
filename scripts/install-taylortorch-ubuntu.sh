#!/bin/bash
################################################################################
# TaylorTorch Installation Script for Ubuntu 24.04
#
# This script installs Swift (latest dev snapshot) and PyTorch via pip.
# Using pip-installed PyTorch provides optimized binaries with MKL/MKLDNN
# support, resulting in ~10x faster training compared to source builds.
#
# Usage:
#   ./install-taylortorch-ubuntu.sh [OPTIONS]
#
# Options:
#   --swift-version VERSION    Specify Swift version (default: main-snapshot-2025-11-03)
#   --pytorch-version VERSION  Specify PyTorch version (default: 2.8.0)
#   --pytorch-dir DIR         PyTorch installation directory (default: auto-detect from pip)
#   --help                    Show this help message
#
# Environment Variables:
#   SWIFT_VERSION             Override Swift version
#   PYTORCH_VERSION           Override PyTorch version
#   PYTORCH_INSTALL_DIR       Override PyTorch installation directory
#
# Example:
#   # Use latest Swift and PyTorch
#   ./install-taylortorch-ubuntu.sh
#
#   # Use specific Swift snapshot
#   ./install-taylortorch-ubuntu.sh --swift-version swift-DEVELOPMENT-SNAPSHOT-2025-10-20-a
#
################################################################################

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Default configuration
SWIFT_VERSION="${SWIFT_VERSION:-main-snapshot-2025-11-03}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.8.0}"
PYTORCH_INSTALL_DIR="${PYTORCH_INSTALL_DIR:-}"  # Will be auto-detected from pip
SKIP_DEPS=false
SKIP_SWIFT=false
SKIP_PYTORCH=false

# Parse command line arguments
show_help() {
    head -n 35 "$0" | tail -n +3 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --swift-version)
            SWIFT_VERSION="$2"
            shift 2
            ;;
        --pytorch-version)
            PYTORCH_VERSION="$2"
            shift 2
            ;;
        --pytorch-dir)
            PYTORCH_INSTALL_DIR="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-swift)
            SKIP_SWIFT=true
            shift
            ;;
        --skip-pytorch)
            SKIP_PYTORCH=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Print configuration
log_section "Configuration"
log_info "Swift Version: $SWIFT_VERSION"
log_info "PyTorch Version: $PYTORCH_VERSION"
log_info "Skip Dependencies: $SKIP_DEPS"
log_info "Skip Swift: $SKIP_SWIFT"
log_info "Skip PyTorch: $SKIP_PYTORCH"

# Check if running as root for system-wide installation
if [[ $EUID -eq 0 ]]; then
    log_warning "Running as root. Installing system-wide."
    SWIFTLY_HOME="/root/.local/share/swiftly"
else
    log_info "Running as user. Installing to user directory."
    SWIFTLY_HOME="$HOME/.local/share/swiftly"
fi

################################################################################
# 1. Install System Dependencies
################################################################################

if [[ "$SKIP_DEPS" == false ]]; then
    log_section "Installing System Dependencies"

    # Check Ubuntu version
    if ! grep -q "Ubuntu 24.04" /etc/os-release 2>/dev/null; then
        log_warning "This script is designed for Ubuntu 24.04. You're running a different version."
        log_warning "Continuing anyway, but you may encounter issues."
    fi

    log_info "Updating package lists..."
    sudo apt-get update

    log_info "Installing Swift dependencies..."
    sudo apt-get install -y \
        build-essential \
        cmake \
        curl \
        git \
        wget \
        ca-certificates \
        binutils \
        gnupg2 \
        libc6-dev \
        libcurl4-openssl-dev \
        libedit2 \
        libgcc-13-dev \
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
        ninja-build

    log_info "Installing Python and PyTorch build dependencies..."
    sudo apt-get install -y \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-yaml \
        libpython3-dev \
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
        libgflags-dev

    # Install libc++ 18 and LLVM 18 (compatible with most recent Swift snapshots)
    # Note: Swift main-snapshot-2025-11-03 uses clang 21, which works with libc++ 18
    log_info "Installing libc++ 18 for compatibility with Swift snapshots..."
    sudo apt-get install -y \
        llvm-18-dev \
        libc++-18-dev \
        libc++abi-18-dev \
        libomp-18-dev

    # Remove libc++ 17 completely to avoid conflicts (CRITICAL!)
    log_info "Removing ALL LLVM 17 packages to prevent version conflicts..."
    sudo apt-get remove -y llvm-17-dev libc++-17-dev libc++abi-17-dev libomp-17-dev llvm-17 'llvm-17-*' 2>/dev/null || true
    sudo apt-get autoremove -y

    # Verify LLVM 17 is gone
    if [ -d "/usr/lib/llvm-17" ]; then
        log_warning "LLVM 17 directory still exists at /usr/lib/llvm-17"
        log_warning "This may cause header search path issues during PyTorch build"
    fi

    log_info "Installing Python packages..."
    pip3 install --break-system-packages \
        numpy \
        pyyaml \
        typing_extensions \
        sympy \
        cffi

    log_success "System dependencies installed"
else
    log_info "Skipping system dependencies installation"
fi

################################################################################
# 2. Install Swift via Swiftly
################################################################################

if [[ "$SKIP_SWIFT" == false ]]; then
    log_section "Installing Swift $SWIFT_VERSION"

    # Check if swiftly is already installed
    if command -v swiftly &> /dev/null; then
        log_info "Swiftly already installed at $(which swiftly)"
    else
        log_info "Installing Swiftly (Swift toolchain manager)..."
        ARCH=$(uname -m)
        curl -f -L -O "https://download.swift.org/swiftly/linux/swiftly-${ARCH}.tar.gz"
        tar zxf "swiftly-${ARCH}.tar.gz"
        ./swiftly init --quiet-shell-followup
        rm "swiftly-${ARCH}.tar.gz"

        # Add swiftly to PATH
        export PATH="$SWIFTLY_HOME/bin:$PATH"
        log_success "Swiftly installed"
    fi

    # Ensure swiftly is in PATH
    export PATH="$SWIFTLY_HOME/bin:$PATH"

    log_info "Installing Swift snapshot: $SWIFT_VERSION"
    swiftly install "$SWIFT_VERSION"
    swiftly use "$SWIFT_VERSION"

    log_info "Swift installation complete:"
    swift --version

    # Set up environment variables
    SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)"
    SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_PATH}/usr"
    SWIFT_BIN_DIR="${SWIFT_TOOLCHAIN_DIR}/bin"

    log_info "Swift toolchain path: $SWIFT_TOOLCHAIN_PATH"
    log_info "Swift bin directory: $SWIFT_BIN_DIR"

    # Create environment file
    log_info "Setting up Swift environment variables..."
    sudo tee /etc/profile.d/swift.sh > /dev/null <<EOF
export PATH="$SWIFTLY_HOME/bin:${SWIFT_BIN_DIR}:\$PATH"
export SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_DIR}"
export SWIFT_TOOLCHAIN_PATH="${SWIFT_TOOLCHAIN_PATH}"
export SWIFT_TOOLCHAIN="${SWIFT_TOOLCHAIN_PATH}"
export SWIFTPM_SWIFT_EXEC="${SWIFT_BIN_DIR}/swift"
EOF

    # Source it for current session
    source /etc/profile.d/swift.sh

    log_success "Swift $SWIFT_VERSION installed and configured"
else
    log_info "Skipping Swift installation"
    # Still need to set up environment
    if command -v swiftly &> /dev/null; then
        export PATH="$SWIFTLY_HOME/bin:$PATH"
        SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)"
        SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_PATH}/usr"
        SWIFT_BIN_DIR="${SWIFT_TOOLCHAIN_DIR}/bin"
    fi
fi

################################################################################
# 3. Verify OpenMP Installation
################################################################################

log_section "Verifying OpenMP Installation"

# Detect Swift toolchain (same logic as PyTorch build section)
SWIFT_BIN="$(which swift 2>/dev/null || true)"
if [ -z "${SWIFT_BIN}" ]; then
    log_error "Swift not found in PATH!"
    log_error "Please run 'source /etc/profile.d/swift.sh' or add Swift to your PATH"
    log_error "Example: export PATH=\"/home/pedro/.local/share/swiftly/bin:\$PATH\""
    exit 1
fi

# Check if we're using swiftly by looking for its directory structure
SWIFTLY_BASE="$(dirname $(dirname ${SWIFT_BIN}))"
if [ -f "${SWIFTLY_BASE}/config.json" ]; then
    # Using swiftly - parse config.json to get active toolchain
    log_info "Detected swiftly installation at ${SWIFTLY_BASE}"

    # Extract the "inUse" field from config.json
    ACTIVE_TOOLCHAIN=$(cat "${SWIFTLY_BASE}/config.json" | sed -n 's/.*"inUse"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')

    if [ -z "${ACTIVE_TOOLCHAIN}" ]; then
        log_error "Could not parse active toolchain from swiftly config.json"
        exit 1
    fi

    SWIFT_TOOLCHAIN_DIR="${SWIFTLY_BASE}/toolchains/${ACTIVE_TOOLCHAIN}/usr"

    if [ ! -d "${SWIFT_TOOLCHAIN_DIR}" ]; then
        log_error "Active toolchain directory not found: ${SWIFT_TOOLCHAIN_DIR}"
        log_error "Swiftly config says toolchain is: ${ACTIVE_TOOLCHAIN}"
        exit 1
    fi

    log_info "Using Swift toolchain: ${ACTIVE_TOOLCHAIN}"
else
    # Not using swiftly - use traditional detection
    SWIFT_TOOLCHAIN_DIR="$(dirname $(dirname ${SWIFT_BIN}))"
fi

CLANG_RESOURCE_DIR="$(${SWIFT_TOOLCHAIN_DIR}/bin/clang++ -print-resource-dir)"
log_info "Clang resource dir: ${CLANG_RESOURCE_DIR}"

# Find omp.h header file
# Try Swift's clang resource dir, then LLVM 17, then system GCC
RESOURCE_OMP="${CLANG_RESOURCE_DIR}/include/omp.h"
OMP_INCLUDE_DIR=""
for candidate in \
    "${RESOURCE_OMP}" \
    "/usr/lib/llvm-18/lib/clang/18/include/omp.h" \
    "/usr/lib/llvm-18/include/omp.h" \
    "/usr/lib/llvm-17/lib/clang/17/include/omp.h" \
    "/usr/lib/llvm-17/include/omp.h" \
    "/usr/lib/gcc/x86_64-linux-gnu/13/include/omp.h" \
    "/usr/include/omp.h" \
    "/usr/lib/x86_64-linux-gnu/openmp/include/omp.h"; do
    if [ -f "${candidate}" ]; then
        OMP_INCLUDE_DIR="$(dirname "${candidate}")"
        log_success "Found omp.h at ${candidate}"
        break
    fi
done

if [ -z "${OMP_INCLUDE_DIR}" ]; then
    log_error "omp.h not found in expected locations"
    log_info "Searching system for omp.h..."
    FOUND_OMP="$(find /usr -name "omp.h" 2>/dev/null | grep -v android | head -1)"
    if [ -n "${FOUND_OMP}" ]; then
        OMP_INCLUDE_DIR="$(dirname "${FOUND_OMP}")"
        log_success "Found omp.h at ${FOUND_OMP}"
    else
        log_error "omp.h not found! Please install libomp-dev or libomp-17-dev"
        exit 1
    fi
fi

# Find libomp.so library file
LLVM_BASE_DIR="$(realpath "${CLANG_RESOURCE_DIR}/../../..")"
log_info "LLVM base dir: ${LLVM_BASE_DIR}"
OMP_LIBRARY=""
for candidate in \
    "${LLVM_BASE_DIR}/lib/libomp.so" \
    "/usr/lib/llvm-18/lib/libomp.so" \
    "/usr/lib/llvm-17/lib/libomp.so" \
    "/usr/lib/x86_64-linux-gnu/libomp.so" \
    "/usr/lib/x86_64-linux-gnu/libomp.so.5" \
    "/usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so"; do
    if [ -f "${candidate}" ]; then
        OMP_LIBRARY="${candidate}"
        log_success "Found OpenMP library at ${candidate}"
        break
    fi
done

if [ -z "${OMP_LIBRARY}" ]; then
    log_error "OpenMP library not found in expected locations"
    log_info "Searching system for libomp.so or libgomp.so..."
    FOUND_OMP_LIB="$(find /usr -name "libomp.so*" -o -name "libgomp.so*" 2>/dev/null | grep -v android | head -1)"
    if [ -n "${FOUND_OMP_LIB}" ]; then
        OMP_LIBRARY="${FOUND_OMP_LIB}"
        log_success "Found OpenMP library at ${FOUND_OMP_LIB}"
    else
        log_error "OpenMP library not found! Please install libomp-dev or libomp-17-dev"
        exit 1
    fi
fi

# Save OpenMP paths
OMP_INCLUDE_DIR="$(realpath "${OMP_INCLUDE_DIR}")"
OMP_LIBRARY="$(realpath "${OMP_LIBRARY}")"

log_success "OpenMP verified and configured"

################################################################################
# 4. Install PyTorch via pip
################################################################################

if [[ "$SKIP_PYTORCH" == false ]]; then
    log_section "Installing PyTorch $PYTORCH_VERSION via pip"

    # Install PyTorch via pip (includes optimized MKL/MKLDNN binaries)
    log_info "Installing PyTorch ${PYTORCH_VERSION} via pip..."
    log_info "This provides pre-built optimized binaries with MKL and MKLDNN support."
    log_info "Performance is ~10x faster than building from source without these optimizations."

    pip3 install --break-system-packages "torch==${PYTORCH_VERSION}"

    # Detect PyTorch installation path
    log_info "Detecting PyTorch installation path..."
    PYTORCH_INSTALL_DIR=$(python3 -c "import torch; print(torch.__path__[0])" 2>/dev/null || true)

    if [ -z "${PYTORCH_INSTALL_DIR}" ]; then
        log_error "Failed to detect PyTorch installation path"
        log_error "Please ensure PyTorch was installed correctly"
        exit 1
    fi

    log_success "PyTorch installed at: ${PYTORCH_INSTALL_DIR}"

    # Verify installation
    log_info "Verifying PyTorch installation..."
    PYTORCH_LIB_DIR="${PYTORCH_INSTALL_DIR}/lib"
    if [ -f "${PYTORCH_LIB_DIR}/libtorch.so" ]; then
        log_success "Found libtorch.so"
    else
        log_error "libtorch.so not found at ${PYTORCH_LIB_DIR}"
        exit 1
    fi

    # Check for MKL/MKLDNN support
    log_info "Checking PyTorch build configuration..."
    python3 -c "import torch; config = torch.__config__.show(); print('MKL:', 'USE_MKL=ON' in config); print('MKLDNN:', 'USE_MKLDNN=ON' in config)"

    log_success "PyTorch ${PYTORCH_VERSION} installed successfully"
else
    log_info "Skipping PyTorch installation"
    # Still need to detect existing installation
    PYTORCH_INSTALL_DIR=$(python3 -c "import torch; print(torch.__path__[0])" 2>/dev/null || true)
    if [ -z "${PYTORCH_INSTALL_DIR}" ]; then
        log_warning "PyTorch not found. Please install it manually or run without --skip-pytorch"
    fi
fi

PYTORCH_LIB_DIR="${PYTORCH_INSTALL_DIR}/lib"

################################################################################
# 5. Verify Installation and Create Environment File
################################################################################

log_section "Verifying Installation"

# Verify Swift
if command -v swift &> /dev/null; then
    log_success "Swift is installed:"
    swift --version
else
    log_error "Swift not found in PATH"
    exit 1
fi

# Verify PyTorch
if [ -f "${PYTORCH_LIB_DIR}/libtorch.so" ]; then
    log_success "PyTorch library found:"
    ls -lh ${PYTORCH_LIB_DIR}/libtorch.so
else
    log_error "PyTorch library not found at ${PYTORCH_LIB_DIR}/libtorch.so"
    exit 1
fi

# Update PyTorch environment file with pip installation path
log_info "Updating PyTorch environment file..."
if sudo -n true 2>/dev/null; then
    sudo tee /etc/profile.d/pytorch.sh > /dev/null <<EOF
export PYTORCH_INSTALL_DIR="${PYTORCH_INSTALL_DIR}"
export SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_DIR}"
export LD_LIBRARY_PATH="${PYTORCH_LIB_DIR}:\${LD_LIBRARY_PATH:-}"
EOF
    log_success "Environment file updated at /etc/profile.d/pytorch.sh"
else
    log_warning "Cannot update /etc/profile.d/pytorch.sh without sudo"
    log_info "Please manually set these environment variables:"
    echo "    export PYTORCH_INSTALL_DIR=\"${PYTORCH_INSTALL_DIR}\""
    echo "    export SWIFT_TOOLCHAIN_DIR=\"${SWIFT_TOOLCHAIN_DIR}\""
    echo "    export LD_LIBRARY_PATH=\"${PYTORCH_LIB_DIR}:\${LD_LIBRARY_PATH:-}\""
fi

# Print environment setup instructions
log_section "Installation Complete"
log_success "Swift and PyTorch successfully installed!"
log_info ""
log_info "PyTorch version: ${PYTORCH_VERSION} (pip-installed with MKL/MKLDNN optimizations)"
log_info "PyTorch location: ${PYTORCH_INSTALL_DIR}"
log_info ""
log_info "To use Swift and PyTorch in new shell sessions, add to your ~/.bashrc:"
log_info ""
echo "    source /etc/profile.d/swift.sh"
echo "    source /etc/profile.d/pytorch.sh"
log_info ""
log_info "Or run the following command:"
log_info ""
echo "    echo 'source /etc/profile.d/swift.sh' >> ~/.bashrc"
echo "    echo 'source /etc/profile.d/pytorch.sh' >> ~/.bashrc"
log_info ""
log_info "To build TaylorTorch, run:"
log_info ""
echo "    export SWIFT_TOOLCHAIN_DIR=\"${SWIFT_TOOLCHAIN_DIR}\""
echo "    export PYTORCH_INSTALL_DIR=\"${PYTORCH_INSTALL_DIR}\""
echo "    export LD_LIBRARY_PATH=\"${PYTORCH_LIB_DIR}:\$LD_LIBRARY_PATH\""
echo "    cd <path-to-TaylorTorch>"
echo "    swift build -c release"
log_info ""
log_success "Installation script completed successfully!"
