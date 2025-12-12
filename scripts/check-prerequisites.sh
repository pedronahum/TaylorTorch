#!/bin/bash
################################################################################
# TaylorTorch Prerequisites Checker
#
# This script checks if all prerequisites for building TaylorTorch are installed.
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
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[MISSING]${NC} $1"; }

echo "=========================================="
echo "TaylorTorch Prerequisites Check"
echo "=========================================="
echo ""

MISSING=0

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    log_success "macOS detected"
elif grep -q "Ubuntu 24.04" /etc/os-release 2>/dev/null; then
    OS="Linux"
    log_success "Ubuntu 24.04 detected"
elif [ -f /etc/os-release ]; then
    OS="Linux"
    log_warning "Non-Ubuntu Linux detected - some packages may differ"
else
    OS="Unknown"
    log_warning "Unknown OS - some checks may not apply"
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

# Check PyTorch - first try pip installation, then fall back to custom path
PYTORCH_FOUND=false

# Try to detect pip-installed PyTorch
if command -v python3 &> /dev/null; then
    PIP_PYTORCH_DIR=$(python3 -c "import torch; print(torch.__path__[0])" 2>/dev/null || true)
    if [ -n "$PIP_PYTORCH_DIR" ] && [ -f "${PIP_PYTORCH_DIR}/lib/libtorch.so" -o -f "${PIP_PYTORCH_DIR}/lib/libtorch.dylib" ]; then
        log_success "PyTorch found via pip at ${PIP_PYTORCH_DIR}"
        PYTORCH_FOUND=true

        # Check for optimizations
        python3 -c "import torch; config = torch.__config__.show(); mkl='USE_MKL=ON' in config; mkldnn='USE_MKLDNN=ON' in config; print('  MKL:', 'enabled' if mkl else 'disabled'); print('  MKLDNN:', 'enabled' if mkldnn else 'disabled')" 2>/dev/null || true
    fi
fi

# Fall back to PYTORCH_INSTALL_DIR or default paths
if [ "$PYTORCH_FOUND" = false ]; then
    PYTORCH_DIR="${PYTORCH_INSTALL_DIR:-/opt/pytorch}"
    if [ -f "${PYTORCH_DIR}/lib/libtorch.so" ] || [ -f "${PYTORCH_DIR}/lib/libtorch.dylib" ]; then
        log_success "PyTorch library found at ${PYTORCH_DIR}"
        PYTORCH_FOUND=true
    fi
fi

if [ "$PYTORCH_FOUND" = false ]; then
    log_error "PyTorch not found"
    log_info "Install via pip: pip3 install torch==2.8.0"
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

# Linux-specific checks
if [ "$OS" = "Linux" ]; then
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
fi

# macOS-specific checks
if [ "$OS" = "macOS" ]; then
    # Check for libomp (via Homebrew)
    if [ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ] || [ -f "/usr/local/opt/libomp/lib/libomp.dylib" ]; then
        log_success "OpenMP library found (via Homebrew)"
    else
        log_warning "OpenMP not found - install with: brew install libomp"
    fi

    # Check SDKROOT
    if [ -n "${SDKROOT:-}" ]; then
        log_success "SDKROOT set to ${SDKROOT}"
    else
        SDKROOT=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)
        if [ -n "$SDKROOT" ]; then
            log_success "SDKROOT available at ${SDKROOT}"
        else
            log_warning "SDKROOT not set and xcrun failed"
        fi
    fi
fi

# Check environment files (Linux only)
if [ "$OS" = "Linux" ]; then
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
fi

# Check LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
if [ "$OS" = "Linux" ]; then
    if [[ "${LD_LIBRARY_PATH:-}" == *"torch"* ]] || [[ "${LD_LIBRARY_PATH:-}" == *"pytorch"* ]]; then
        log_success "PyTorch in LD_LIBRARY_PATH"
    else
        log_warning "PyTorch not in LD_LIBRARY_PATH (may need to source /etc/profile.d/pytorch.sh)"
    fi
elif [ "$OS" = "macOS" ]; then
    if [[ "${DYLD_LIBRARY_PATH:-}" == *"torch"* ]] || [[ "${DYLD_LIBRARY_PATH:-}" == *"pytorch"* ]]; then
        log_success "PyTorch in DYLD_LIBRARY_PATH"
    else
        log_warning "PyTorch may not be in DYLD_LIBRARY_PATH"
    fi
fi

echo ""
echo "=========================================="
if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}All prerequisites satisfied!${NC}"
    echo "You can build TaylorTorch with: swift build"
    exit 0
else
    echo -e "${RED}Missing $MISSING prerequisite(s)${NC}"
    if [ "$OS" = "Linux" ]; then
        echo "Run ./scripts/install-taylortorch-ubuntu.sh to install missing dependencies"
    else
        echo "Install missing dependencies and try again"
    fi
    exit 1
fi
