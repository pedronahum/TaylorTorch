# TaylorTorch Installation Scripts

This directory contains installation scripts for setting up TaylorTorch on various platforms.

## Ubuntu 24.04 Installation

The [install-taylortorch-ubuntu.sh](install-taylortorch-ubuntu.sh) script automates the complete setup of TaylorTorch on Ubuntu 24.04, including:

1. Installing system dependencies (Swift and PyTorch build requirements)
2. Installing Swift via Swiftly (using the latest development snapshot)
3. Building PyTorch from source with the same clang compiler as Swift
4. Configuring environment variables and library paths

### Quick Start

```bash
# Run with default settings (Swift 2025-11-03 snapshot, PyTorch v2.8.0)
./scripts/install-taylortorch-ubuntu.sh

# Or with custom settings
./scripts/install-taylortorch-ubuntu.sh --swift-version main-snapshot --max-jobs 4
```

### Usage Options

```bash
./install-taylortorch-ubuntu.sh [OPTIONS]

Options:
  --swift-version VERSION    Specify Swift version (default: main-snapshot-2025-11-03)
  --pytorch-version VERSION  Specify PyTorch version (default: v2.8.0)
  --pytorch-dir DIR         PyTorch installation directory (default: /opt/pytorch)
  --max-jobs N              Max parallel jobs for building (default: auto-detect)
  --skip-deps               Skip installing system dependencies
  --skip-swift              Skip Swift installation
  --skip-pytorch            Skip PyTorch build
  --help                    Show help message
```

### Examples

#### Install with specific Swift snapshot
```bash
./scripts/install-taylortorch-ubuntu.sh --swift-version swift-DEVELOPMENT-SNAPSHOT-2025-10-20-a
```

#### Install with 4 parallel build jobs (for systems with limited RAM)
```bash
./scripts/install-taylortorch-ubuntu.sh --max-jobs 4
```

#### Skip dependencies if already installed
```bash
./scripts/install-taylortorch-ubuntu.sh --skip-deps
```

#### Use environment variables
```bash
export SWIFT_VERSION="main-snapshot"
export PYTORCH_VERSION="v2.8.0"
export MAX_JOBS=8
./scripts/install-taylortorch-ubuntu.sh
```

### What the Script Does

1. **System Dependencies Installation**
   - Installs build tools (gcc, cmake, ninja)
   - Installs clang-18 and LLVM libraries
   - Installs Python and PyTorch build dependencies
   - Installs OpenMP, protobuf, and other required libraries

2. **Swift Installation**
   - Downloads and installs Swiftly (Swift toolchain manager)
   - Installs the specified Swift development snapshot
   - Configures Swift environment variables in `/etc/profile.d/swift.sh`

3. **OpenMP Verification**
   - Locates OpenMP headers and libraries
   - Ensures compatibility with clang compiler
   - Configures environment for PyTorch build

4. **PyTorch Build**
   - Clones PyTorch repository at specified version
   - Uses Swift's clang compiler for C++ ABI compatibility
   - Builds PyTorch with libstdc++ (GCC's C++ standard library)
   - Builds PyTorch with CPU-only support, optimized for TaylorTorch
   - Installs PyTorch to `/opt/pytorch` (or custom directory)

5. **Environment Configuration**
   - Sets up library paths in `/etc/ld.so.conf.d/pytorch.conf`
   - Creates environment files for easy setup
   - Verifies installation

### Key Features

- **Compiler Compatibility**: Ensures PyTorch is compiled with Swift's clang and libstdc++, preventing C++ ABI mismatches
- **Tested Swift Snapshot**: Uses Swift development snapshot from 2025-11-03 by default (known stable version)
- **Optimized Build**: Configures PyTorch with minimal dependencies (CPU-only, no Python, no distributed)
- **Environment Management**: Automatically configures all necessary environment variables
- **Error Handling**: Comprehensive error checking and logging throughout the process
- **Resumable**: Can skip already-completed steps with `--skip-*` flags

### Post-Installation

After the script completes, add the environment files to your shell configuration:

```bash
echo 'source /etc/profile.d/swift.sh' >> ~/.bashrc
echo 'source /etc/profile.d/pytorch.sh' >> ~/.bashrc
source ~/.bashrc
```

Then build TaylorTorch:

```bash
cd /home/pedro/programming/swift/TaylorTorch
swift build
```

### Required Environment Variables

**IMPORTANT**: The following environment variables must be set before building TaylorTorch:

```bash
export SWIFT_TOOLCHAIN_DIR="/home/pedro/.local/share/swiftly/toolchains/main-snapshot-2025-11-03/usr"
export PYTORCH_INSTALL_DIR="/opt/pytorch"
export PATH="/home/pedro/.local/share/swiftly/bin:$PATH"
```

These are automatically set when you source `/etc/profile.d/swift.sh` and `/etc/profile.d/pytorch.sh`. If you get errors like `'swift/bridging' file not found`, ensure these environment variables are properly set.

For Docker or CI environments, you can also set these in your Dockerfile or workflow:
```bash
# Source environment files
. /etc/profile.d/swift.sh
. /etc/profile.d/pytorch.sh
```

### System Requirements

- Ubuntu 24.04 (may work on other versions but untested)
- At least 8GB RAM (16GB recommended for faster builds)
- At least 20GB free disk space
- Internet connection for downloading dependencies

### Build Time Estimates

- System dependencies: 5-10 minutes
- Swift installation: 2-5 minutes
- PyTorch build: 30-90 minutes (depending on CPU and `--max-jobs` setting)

### Troubleshooting

#### Out of Memory during PyTorch build
Reduce the number of parallel jobs:
```bash
./scripts/install-taylortorch-ubuntu.sh --max-jobs 2
```

#### Swift version not found
Check available versions at https://www.swift.org/download/
```bash
# Use a specific snapshot
./scripts/install-taylortorch-ubuntu.sh --swift-version swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a
```

#### Permission denied errors
The script uses `sudo` for system-wide installation. Ensure you have sudo privileges:
```bash
sudo -v  # Test sudo access
./scripts/install-taylortorch-ubuntu.sh
```

#### C++ standard library issues
The script uses libstdc++ (GCC's C++ standard library). Ensure gcc-13 packages are installed:
```bash
sudo apt-get install libstdc++-13-dev libgcc-13-dev
```

### Environment Variables Set by Script

The script creates two environment files:

**`/etc/profile.d/swift.sh`**:
```bash
export PATH="/root/.local/share/swiftly/bin:${SWIFT_BIN_DIR}:$PATH"
export SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_DIR}"
export SWIFT_TOOLCHAIN_PATH="${SWIFT_TOOLCHAIN_PATH}"
export SWIFT_TOOLCHAIN="${SWIFT_TOOLCHAIN_PATH}"
export SWIFTPM_SWIFT_EXEC="${SWIFT_BIN_DIR}/swift"
```

**`/etc/profile.d/pytorch.sh`**:
```bash
export OMP_INCLUDE_DIR="${OMP_INCLUDE_DIR}"
export OMP_LIBRARY="${OMP_LIBRARY}"
export PYTORCH_INSTALL_DIR="/opt/pytorch"
export CC=clang
export CXX=clang++
export LD_LIBRARY_PATH="/opt/pytorch/lib:$LD_LIBRARY_PATH"
```

### Differences from Dockerfile

The installation script is functionally equivalent to the Dockerfile but designed for standalone use:

- Interactive prompts when rebuilding existing installations
- Better error messages and colored output
- Flexible command-line options
- Ability to skip already-completed steps
- Designed for both root and user installations

### Contributing

If you encounter issues or have improvements, please submit an issue or PR to the TaylorTorch repository.

## License

This script is part of TaylorTorch and is licensed under the Apache 2.0 License. See [../LICENSE](../LICENSE) for details.
