#!/bin/bash
################################################################################
# TaylorTorch Installation Verification
#
# This script verifies that TaylorTorch can be built and run successfully.
# Supports both pip-installed PyTorch (recommended) and source-built PyTorch.
################################################################################

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "TaylorTorch Installation Verification"
echo "=========================================="
echo ""

# Source environment files if they exist
if [ -f /etc/profile.d/swift.sh ]; then
    source /etc/profile.d/swift.sh
fi
if [ -f /etc/profile.d/pytorch.sh ]; then
    set +u
    source /etc/profile.d/pytorch.sh
    set -u
fi

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    LIB_EXT="dylib"
    LIB_PATH_VAR="DYLD_LIBRARY_PATH"
else
    OS="Linux"
    LIB_EXT="so"
    LIB_PATH_VAR="LD_LIBRARY_PATH"
fi

# Step 1: Verify Swift
log_info "Checking Swift installation..."
if ! command -v swift &> /dev/null; then
    log_error "Swift not found in PATH"
    log_info "Try: source /etc/profile.d/swift.sh"
    exit 1
fi
swift --version
log_success "Swift is available"

# Step 2: Verify PyTorch - first try pip, then custom path
log_info "Checking PyTorch installation..."
PYTORCH_DIR=""

# Try pip-installed PyTorch first
if command -v python3 &> /dev/null; then
    PIP_PYTORCH_DIR=$(python3 -c "import torch; print(torch.__path__[0])" 2>/dev/null || true)
    if [ -n "$PIP_PYTORCH_DIR" ] && [ -f "${PIP_PYTORCH_DIR}/lib/libtorch.${LIB_EXT}" ]; then
        PYTORCH_DIR="$PIP_PYTORCH_DIR"
        log_success "PyTorch found via pip at ${PYTORCH_DIR}"

        # Show optimization status
        python3 -c "import torch; config = torch.__config__.show(); print('  MKL:', 'enabled' if 'USE_MKL=ON' in config else 'disabled'); print('  MKLDNN:', 'enabled' if 'USE_MKLDNN=ON' in config else 'disabled')" 2>/dev/null || true
    fi
fi

# Fall back to PYTORCH_INSTALL_DIR or default
if [ -z "$PYTORCH_DIR" ]; then
    PYTORCH_DIR="${PYTORCH_INSTALL_DIR:-/opt/pytorch}"
    if [ -f "${PYTORCH_DIR}/lib/libtorch.${LIB_EXT}" ]; then
        log_success "PyTorch library found at ${PYTORCH_DIR}"
    else
        log_error "PyTorch library not found at ${PYTORCH_DIR}/lib/libtorch.${LIB_EXT}"
        log_info "Install via pip: pip3 install torch==2.8.0"
        exit 1
    fi
fi

# Set environment variables for build
export PYTORCH_INSTALL_DIR="$PYTORCH_DIR"
if [ "$OS" = "Linux" ]; then
    export LD_LIBRARY_PATH="${PYTORCH_DIR}/lib:${LD_LIBRARY_PATH:-}"
else
    export DYLD_LIBRARY_PATH="${PYTORCH_DIR}/lib:${DYLD_LIBRARY_PATH:-}"
fi

log_success "PyTorch library found"

# Step 3: Build TaylorTorch
log_info "Building TaylorTorch..."
if swift build 2>&1; then
    log_success "TaylorTorch built successfully"
else
    log_error "Build failed"
    exit 1
fi

# Step 4: Run tests (optional)
if [ "${RUN_TESTS:-0}" = "1" ]; then
    log_info "Running tests..."
    if swift test 2>&1; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed"
    fi
fi

# Step 5: Check examples can be built
log_info "Verifying example targets..."
for target in MNISTExample ANKIExample KARATEExample; do
    if swift build --target "$target" 2>&1; then
        log_success "$target builds successfully"
    else
        log_warning "$target failed to build"
    fi
done

echo ""
echo "=========================================="
log_success "TaylorTorch installation verified!"
echo "=========================================="
echo ""
echo "PyTorch location: ${PYTORCH_DIR}"
echo ""
echo "You can now:"
echo "  - Build:      swift build"
echo "  - Test:       swift test"
echo "  - Run MNIST:  swift run MNISTExample"
echo ""
