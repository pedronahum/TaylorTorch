#!/bin/bash
################################################################################
# TaylorTorch Installation Script for Ubuntu 24.04
#
# This script installs Swift (latest dev snapshot) and PyTorch from source,
# ensuring that PyTorch is compiled with the same clang compiler from Swift.
#
# Usage:
#   ./install-taylortorch-ubuntu.sh [OPTIONS]
#
# Options:
#   --swift-version VERSION    Specify Swift version (default: main-snapshot-2025-11-03)
#   --pytorch-version VERSION  Specify PyTorch version (default: v2.8.0)
#   --pytorch-dir DIR         PyTorch installation directory (default: /opt/pytorch)
#   --max-jobs N              Max parallel jobs for building (default: auto-detect)
#   --help                    Show this help message
#
# Environment Variables:
#   SWIFT_VERSION             Override Swift version
#   PYTORCH_VERSION           Override PyTorch version
#   PYTORCH_INSTALL_DIR       Override PyTorch installation directory
#   MAX_JOBS                  Override max parallel jobs
#
# Example:
#   # Use latest Swift and PyTorch with 4 jobs
#   ./install-taylortorch-ubuntu.sh --max-jobs 4
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
PYTORCH_VERSION="${PYTORCH_VERSION:-v2.8.0}"
PYTORCH_INSTALL_DIR="${PYTORCH_INSTALL_DIR:-/opt/pytorch}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
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
        --max-jobs)
            MAX_JOBS="$2"
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
log_info "PyTorch Install Dir: $PYTORCH_INSTALL_DIR"
log_info "Max Jobs: $MAX_JOBS"
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

    log_info "Installing Python packages for PyTorch..."
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

log_info "Creating PyTorch environment file..."
if sudo -n true 2>/dev/null; then
    # We have passwordless sudo or are root
    sudo tee /etc/profile.d/pytorch.sh > /dev/null <<EOF
export OMP_INCLUDE_DIR="${OMP_INCLUDE_DIR}"
export OMP_LIBRARY="${OMP_LIBRARY}"
export PYTORCH_INSTALL_DIR="${PYTORCH_INSTALL_DIR}"
export SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_DIR}"
export CC=clang
export CXX=clang++
export LD_LIBRARY_PATH="${PYTORCH_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
EOF
else
    # No sudo available - check if file exists and is reasonable
    if [ -f /etc/profile.d/pytorch.sh ]; then
        log_info "PyTorch environment file already exists and no sudo available - skipping update"
    else
        log_error "Cannot create /etc/profile.d/pytorch.sh without sudo permissions"
        exit 1
    fi
fi

# Source it safely (LD_LIBRARY_PATH might not be set yet)
set +u  # Temporarily allow unset variables
source /etc/profile.d/pytorch.sh
set -u  # Re-enable strict mode

log_success "OpenMP verified and configured"

################################################################################
# 4. Build and Install PyTorch
################################################################################

if [[ "$SKIP_PYTORCH" == false ]]; then
    log_section "Building PyTorch $PYTORCH_VERSION"

    # Check if PyTorch is already installed
    if [ -f "${PYTORCH_INSTALL_DIR}/lib/libtorch.so" ]; then
        log_warning "PyTorch already installed at ${PYTORCH_INSTALL_DIR}"
        read -p "Do you want to rebuild PyTorch? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping PyTorch build"
            SKIP_PYTORCH=true
        fi
    fi

    if [[ "$SKIP_PYTORCH" == false ]]; then
        # Clean up before build
        log_info "Checking available disk space..."
        df -h

        # Clone PyTorch
        log_info "Cloning PyTorch repository (version: $PYTORCH_VERSION)..."
        if [ -d "/tmp/pytorch" ]; then
            log_warning "Removing existing /tmp/pytorch directory"
            rm -rf /tmp/pytorch
        fi

        git clone --depth=1 --shallow-submodules --branch ${PYTORCH_VERSION} \
            https://github.com/pytorch/pytorch.git /tmp/pytorch || \
            (log_warning "First clone attempt failed, retrying..." && sleep 10 && \
            git clone --depth=1 --shallow-submodules --branch ${PYTORCH_VERSION} \
            https://github.com/pytorch/pytorch.git /tmp/pytorch)

        # Update submodules
        log_info "Updating PyTorch submodules..."
        cd /tmp/pytorch
        git submodule sync
        git submodule update --init --depth=1 --recursive

        # Configure PyTorch build
        log_section "Configuring PyTorch Build"

        source /etc/profile.d/pytorch.sh
        
        # Isolate OpenMP header to avoid pulling in conflicting GCC headers (limits.h, lwpintrin.h)
        # This is critical because Clang's resource directory must take precedence, but we need omp.h
        if [[ "$OMP_INCLUDE_DIR" == *"/usr/lib/gcc"* ]]; then
            log_info "Isolating OpenMP header from GCC include directory..."
            ISOLATED_OMP_DIR="/tmp/taylortorch_omp"
            mkdir -p "$ISOLATED_OMP_DIR"
            cp "$OMP_INCLUDE_DIR/omp.h" "$ISOLATED_OMP_DIR/"
            OMP_INCLUDE_DIR="$ISOLATED_OMP_DIR"
            log_info "New OpenMP include dir: $OMP_INCLUDE_DIR"
        fi

        log_info "Using OpenMP include: $OMP_INCLUDE_DIR"
        log_info "Using OpenMP library: $OMP_LIBRARY"

        cd /tmp/pytorch
        mkdir -p build
        cd build

        # Detect Swift toolchain
        # CRITICAL: When using swiftly, the swift binary is a wrapper script
        # We must parse swiftly's config.json to find the actual active toolchain
        SWIFT_BIN="$(which swift 2>/dev/null || true)"
        if [ -z "${SWIFT_BIN}" ]; then
            log_error "Swift not found in PATH! Please ensure Swift is installed via Swiftly."
            log_error "Run: source /etc/profile.d/swift.sh"
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
                log_error "You may need to run: swiftly use ${SWIFT_VERSION}"
                exit 1
            fi

            log_info "Detected active Swift toolchain: ${ACTIVE_TOOLCHAIN}"
            log_info "Swift toolchain directory: ${SWIFT_TOOLCHAIN_DIR}"

            # Verify this matches the desired Swift version
            SWIFT_ACTUAL_VERSION="$(${SWIFT_TOOLCHAIN_DIR}/bin/swift --version | head -1)"
            log_info "Swift version: ${SWIFT_ACTUAL_VERSION}"

            if [[ ! "${ACTIVE_TOOLCHAIN}" =~ ${SWIFT_VERSION} ]]; then
                log_error "Active Swift toolchain (${ACTIVE_TOOLCHAIN}) does not match requested version (${SWIFT_VERSION})"
                log_error "Run: swiftly use ${SWIFT_VERSION}"
                exit 1
            fi
        else
            # Not using swiftly - use traditional detection
            SWIFT_TOOLCHAIN_DIR="$(dirname $(dirname ${SWIFT_BIN}))"
            log_info "Detected Swift toolchain dir from PATH: ${SWIFT_TOOLCHAIN_DIR}"
        fi

        # Detect compilers and libc++ from Swift's clang
        CLANG_RESOURCE_DIR="$(${SWIFT_TOOLCHAIN_DIR}/bin/clang++ -print-resource-dir)"
        LLVM_BASE_DIR="$(realpath "${CLANG_RESOURCE_DIR}/../../..")"
        log_info "Swift's clang resource dir: ${CLANG_RESOURCE_DIR}"
        log_info "LLVM base dir: ${LLVM_BASE_DIR}"

        # Find libc++ (C++ standard library)
        # Try Swift's bundled libc++ first, then fall back to system LLVM (17 or 18)
        LIBCXX_INCLUDE_DIR=""
        LIBCXX_SOURCE="unknown"

        if [ -n "${SWIFT_TOOLCHAIN_DIR:-}" ]; then
            for candidate in \
                "${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include/c++/v1" \
                "${SWIFT_TOOLCHAIN_DIR}/include/c++/v1" \
                "${SWIFT_TOOLCHAIN_DIR}/lib/swift_static/clang/include/c++/v1" \
                "${CLANG_RESOURCE_DIR}/../../../include/c++/v1"; do
                if [ -d "${candidate}" ] && [ -f "${candidate}/cstddef" ]; then
                    LIBCXX_INCLUDE_DIR="$(realpath "${candidate}")"
                    LIBCXX_SOURCE="Swift toolchain"
                    log_success "Found libc++ in Swift toolchain at ${LIBCXX_INCLUDE_DIR}"
                    break
                fi
            done
        fi

        if [ -z "${LIBCXX_INCLUDE_DIR}" ]; then
            log_info "Swift toolchain doesn't include libc++, searching system LLVM"
            # CRITICAL: Try LLVM 18 first (compatible with Swift's clang 21)
            for candidate in \
                "/usr/lib/llvm-18/include/c++/v1" \
                "/usr/lib/llvm-17/include/c++/v1" \
                "/usr/include/c++/v1"; do
                if [ -d "${candidate}" ] && [ -f "${candidate}/cstddef" ]; then
                    LIBCXX_INCLUDE_DIR="$(realpath "${candidate}")"
                    # Detect version from path
                    if [[ "${candidate}" == *"llvm-17"* ]]; then
                        LIBCXX_SOURCE="system LLVM 17"
                    elif [[ "${candidate}" == *"llvm-18"* ]]; then
                        LIBCXX_SOURCE="system LLVM 18"
                    else
                        LIBCXX_SOURCE="system"
                    fi
                    log_success "Found libc++ in system at ${LIBCXX_INCLUDE_DIR}"
                    break
                fi
            done
        fi

        if [ -z "${LIBCXX_INCLUDE_DIR}" ]; then
            log_error "Unable to locate libc++ headers."
            log_error "Please ensure libc++-17-dev or libc++-18-dev is installed."
            exit 1
        fi

        # Find libc++ library
        LIBCXX_LIBRARY_DIR=""
        if [ "$LIBCXX_SOURCE" = "Swift toolchain" ] && [ -n "${SWIFT_TOOLCHAIN_DIR:-}" ]; then
            for candidate in \
                "${SWIFT_TOOLCHAIN_DIR}/lib/swift/linux" \
                "${SWIFT_TOOLCHAIN_DIR}/lib"; do
                if ls "${candidate}/libc++.so"* >/dev/null 2>&1; then
                    LIBCXX_LIBRARY_DIR="$(realpath "${candidate}")"
                    log_success "Found libc++ library in Swift toolchain at ${LIBCXX_LIBRARY_DIR}"
                    break
                fi
            done
        fi

        if [ -z "${LIBCXX_LIBRARY_DIR}" ]; then
            # CRITICAL: Try LLVM 18 first (compatible with Swift's clang 21)
            for candidate in \
                "/usr/lib/llvm-18/lib" \
                "/usr/lib/llvm-17/lib" \
                "/usr/lib/x86_64-linux-gnu" \
                "/usr/lib"; do
                if ls "${candidate}/libc++.so"* >/dev/null 2>&1; then
                    LIBCXX_LIBRARY_DIR="$(realpath "${candidate}")"
                    log_success "Found libc++ library at ${LIBCXX_LIBRARY_DIR}"
                    break
                fi
            done
        fi

        if [ -z "${LIBCXX_LIBRARY_DIR}" ]; then
            log_error "Unable to locate libc++ libraries."
            log_error "Please ensure libc++-17-dev or libc++-18-dev is installed."
            exit 1
        fi

        log_info "libc++ include dir: ${LIBCXX_INCLUDE_DIR}"
        log_info "libc++ library dir: ${LIBCXX_LIBRARY_DIR}"

        # Set up build environment
        export USE_LIBCXX=1
        if [ "$LIBCXX_SOURCE" = "Swift toolchain" ] && [ -n "${SWIFT_TOOLCHAIN_DIR:-}" ]; then
            export CPLUS_INCLUDE_PATH="${SWIFT_TOOLCHAIN_DIR}/lib/swift:${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include"
        fi
        # Note: CXXFLAGS and LDFLAGS are set later to use libstdc++ instead of libc++
        export LDFLAGS="-L${LIBCXX_LIBRARY_DIR}"
        if [ -n "${LIBRARY_PATH:-}" ]; then
            export LIBRARY_PATH="${LIBCXX_LIBRARY_DIR}:${LIBRARY_PATH}"
        else
            export LIBRARY_PATH="${LIBCXX_LIBRARY_DIR}"
        fi

        # OpenMP flags
        OPENMP_C_FLAGS="-fopenmp=libomp"
        OPENMP_CXX_FLAGS="-fopenmp=libomp"
        RESOURCE_INCLUDE="$(realpath "${CLANG_RESOURCE_DIR}/include")"
        if [ -n "$OMP_INCLUDE_DIR" ] && [ "$OMP_INCLUDE_DIR" != "$RESOURCE_INCLUDE" ]; then
            OPENMP_C_FLAGS="$OPENMP_C_FLAGS -isystem ${OMP_INCLUDE_DIR}"
            OPENMP_CXX_FLAGS="$OPENMP_CXX_FLAGS -isystem ${OMP_INCLUDE_DIR}"
        fi

        # Select compiler
        # CRITICAL: Always use Swift's clang for compiler compatibility
        # We can use system libc++ but MUST use Swift's clang to avoid header mismatches
        if [ -n "${SWIFT_TOOLCHAIN_DIR:-}" ] && [ -f "${SWIFT_TOOLCHAIN_DIR}/bin/clang" ]; then
            BUILD_CC="${SWIFT_TOOLCHAIN_DIR}/bin/clang"
            BUILD_CXX="${SWIFT_TOOLCHAIN_DIR}/bin/clang++"
            BUILD_RESOURCE_INCLUDE="${RESOURCE_INCLUDE}"
            log_success "Using Swift's clang for PyTorch build (required for compatibility)"
            log_info "Swift clang: ${BUILD_CXX}"
            log_info "Using libc++ from: ${LIBCXX_SOURCE}"
        else
            log_error "Swift toolchain clang not found!"
            log_error "Swift toolchain directory: ${SWIFT_TOOLCHAIN_DIR:-not set}"
            exit 1
        fi

        # CRITICAL: Use libstdc++ instead of libc++ for PyTorch build
        # Clang 21 (from Swift) + libc++ 18 (from Ubuntu) are incompatible versions
        # Clang can use GCC's libstdc++ without issues, and it's more stable
        # We need Swift's clang resource directory for compiler headers (stddef.h, stdint.h, etc.)
        # Use -I instead of -isystem so clang's headers take priority over GCC's intrinsic headers
        # Must pass through CMAKE_CXX_FLAGS and CMAKE_C_FLAGS since CXXFLAGS/CFLAGS are often ignored by CMake
        export CXXFLAGS="-isystem ${RESOURCE_INCLUDE}"
        export CFLAGS="-isystem ${RESOURCE_INCLUDE}"
        CMAKE_CXX_FLAGS="-isystem ${RESOURCE_INCLUDE}"
        CMAKE_C_FLAGS="-isystem ${RESOURCE_INCLUDE}"

        if [ "$LIBCXX_SOURCE" = "Swift toolchain" ]; then
            CMAKE_PREFIX_PATH="${LLVM_BASE_DIR};/usr"
        elif [ "$LIBCXX_SOURCE" = "system LLVM 17" ]; then
            CMAKE_PREFIX_PATH="/usr/lib/llvm-17;/usr"
        elif [ "$LIBCXX_SOURCE" = "system LLVM 18" ]; then
            CMAKE_PREFIX_PATH="/usr/lib/llvm-18;/usr"
        else
            # Generic fallback
            CMAKE_PREFIX_PATH="/usr"
        fi

        log_section "Build Configuration Summary"
        log_info "Compiler: ${BUILD_CC}"
        log_info "C++ Compiler: ${BUILD_CXX}"
        log_info "libc++ Source: ${LIBCXX_SOURCE}"
        log_info "libc++ Include: ${LIBCXX_INCLUDE_DIR}"
        log_info "libc++ Library: ${LIBCXX_LIBRARY_DIR}"
        log_info "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
        log_info "Max Jobs: ${MAX_JOBS}"

        # Run CMake
        log_info "Running CMake configuration..."
        CC="${BUILD_CC}" CXX="${BUILD_CXX}" cmake .. \
            -DCMAKE_C_COMPILER="${BUILD_CC}" \
            -DCMAKE_CXX_COMPILER="${BUILD_CXX}" \
            -DCMAKE_CXX_STANDARD=17 \
            -DCMAKE_CXX_EXTENSIONS=OFF \
            -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
            -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
            -DCMAKE_EXE_LINKER_FLAGS="" \
            -DCMAKE_SHARED_LINKER_FLAGS="" \
            -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
            -DUSE_LIBCXX=OFF \
            -DOpenMP_C_FLAGS="${OPENMP_C_FLAGS}" \
            -DOpenMP_CXX_FLAGS="${OPENMP_CXX_FLAGS}" \
            -DOpenMP_C_LIB_NAMES="libomp" \
            -DOpenMP_CXX_LIB_NAMES="libomp" \
            -DOpenMP_libomp_LIBRARY="${OMP_LIBRARY}" \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${PYTORCH_INSTALL_DIR}" \
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

        log_success "CMake configuration complete"

        # Build PyTorch
        log_section "Building PyTorch (this may take 30-60 minutes)"
        log_info "Starting PyTorch build with MAX_JOBS=${MAX_JOBS}..."

        # Build first (no sudo needed)
        cmake --build . -j${MAX_JOBS} 2>&1 | tee /tmp/pytorch-build.log || \
            (log_error "BUILD FAILED" && \
            log_error "Last 100 lines of build log:" && \
            tail -n 100 /tmp/pytorch-build.log && \
            exit 1)

        log_success "PyTorch build complete"

        # Install to /opt/pytorch (requires sudo)
        log_info "Installing PyTorch to ${PYTORCH_INSTALL_DIR} (requires sudo)..."
        sudo cmake --build . --target install 2>&1 | tee -a /tmp/pytorch-build.log || \
            (log_error "INSTALL FAILED" && \
            log_error "Last 100 lines of build log:" && \
            tail -n 100 /tmp/pytorch-build.log && \
            exit 1)

        log_success "PyTorch installation complete"

        # Cleanup
        log_info "Cleaning up build files..."
        rm -rf /tmp/pytorch /tmp/pytorch-build.log

        log_info "Verifying PyTorch library installation:"
        ls -lh ${PYTORCH_INSTALL_DIR}/lib/ | head -20
    fi

    # Set up library paths
    log_info "Configuring library paths..."
    echo "${PYTORCH_INSTALL_DIR}/lib" | sudo tee /etc/ld.so.conf.d/pytorch.conf > /dev/null
    sudo ldconfig

    log_success "PyTorch configured"
else
    log_info "Skipping PyTorch build"
fi

################################################################################
# 5. Verify Installation
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
if [ -f "${PYTORCH_INSTALL_DIR}/lib/libtorch.so" ]; then
    log_success "PyTorch library found:"
    ls -lh ${PYTORCH_INSTALL_DIR}/lib/libtorch.so
else
    log_error "PyTorch library not found at ${PYTORCH_INSTALL_DIR}/lib/libtorch.so"
    exit 1
fi

# Print environment setup instructions
log_section "Installation Complete"
log_success "Swift and PyTorch successfully installed!"
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
echo "    cd /home/pedro/programming/swift/TaylorTorch"
echo "    swift build"
log_info ""
log_success "Installation script completed successfully!"
