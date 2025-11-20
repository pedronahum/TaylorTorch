#!/bin/bash
################################################################################
# TaylorTorch Prerequisites Checker
#
# This script checks if all prerequisites for building TaylorTorch are installed.
################################################################################

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[MISSING]${NC} $1"; }

echo "=========================================="
echo "TaylorTorch Prerequisites Check"
echo "=========================================="
echo ""

MISSING=0

# Check Ubuntu version
if grep -q "Ubuntu 24.04" /etc/os-release 2>/dev/null; then
    log_success "Ubuntu 24.04 detected"
else
    log_warning "Not running Ubuntu 24.04 - some packages may differ"
fi

# Check Swift
if command -v swift &> /dev/null; then
    VERSION=$(swift --version 2>&1 | head -1)
    log_success "Swift: $VERSION"
else
    log_error "Swift not found"
    MISSING=$((MISSING + 1))
fi

# Check swiftly
if command -v swiftly &> /dev/null; then
    log_success "Swiftly installed at $(which swiftly)"
else
    log_warning "Swiftly not found (optional but recommended)"
fi

# Check PyTorch
PYTORCH_DIR="${PYTORCH_INSTALL_DIR:-/opt/pytorch}"
if [ -f "${PYTORCH_DIR}/lib/libtorch.so" ]; then
    log_success "PyTorch library found at ${PYTORCH_DIR}"
else
    log_error "PyTorch not found at ${PYTORCH_DIR}/lib/libtorch.so"
    MISSING=$((MISSING + 1))
fi

# Check essential build tools
for cmd in cmake ninja git python3 pip3; do
    if command -v $cmd &> /dev/null; then
        log_success "$cmd found"
    else
        log_error "$cmd not found"
        MISSING=$((MISSING + 1))
    fi
done

# Check C++ compiler
if command -v clang++ &> /dev/null; then
    VERSION=$(clang++ --version | head -1)
    log_success "clang++: $VERSION"
else
    log_error "clang++ not found"
    MISSING=$((MISSING + 1))
fi

# Check libstdc++ headers
if [ -d "/usr/include/c++/13" ]; then
    log_success "libstdc++ 13 headers found"
elif [ -d "/usr/include/c++/12" ]; then
    log_warning "libstdc++ 12 headers found (13 recommended)"
else
    log_error "libstdc++ headers not found"
    MISSING=$((MISSING + 1))
fi

# Check OpenMP
OMP_FOUND=0
for omp in "/usr/lib/llvm-18/lib/libomp.so" "/usr/lib/llvm-17/lib/libomp.so" "/usr/lib/x86_64-linux-gnu/libomp.so"; do
    if [ -f "$omp" ]; then
        log_success "OpenMP library found at $omp"
        OMP_FOUND=1
        break
    fi
done
if [ $OMP_FOUND -eq 0 ]; then
    log_error "OpenMP library not found"
    MISSING=$((MISSING + 1))
fi

# Check environment files
if [ -f "/etc/profile.d/swift.sh" ]; then
    log_success "Swift environment file exists"
else
    log_warning "Swift environment file not found at /etc/profile.d/swift.sh"
fi

if [ -f "/etc/profile.d/pytorch.sh" ]; then
    log_success "PyTorch environment file exists"
else
    log_warning "PyTorch environment file not found at /etc/profile.d/pytorch.sh"
fi

# Check LD_LIBRARY_PATH
if [[ "${LD_LIBRARY_PATH:-}" == *"pytorch"* ]]; then
    log_success "PyTorch in LD_LIBRARY_PATH"
else
    log_warning "PyTorch not in LD_LIBRARY_PATH (may need to source /etc/profile.d/pytorch.sh)"
fi

echo ""
echo "=========================================="
if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}All prerequisites satisfied!${NC}"
    echo "You can build TaylorTorch with: swift build"
    exit 0
else
    echo -e "${RED}Missing $MISSING prerequisite(s)${NC}"
    echo "Run ./scripts/install-taylortorch-ubuntu.sh to install missing dependencies"
    exit 1
fi
