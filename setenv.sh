#!/bin/bash
export CC=clang
export CXX=clang++
export SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)"
export SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_PATH}/usr" 
export SWIFT_BRIDGING_INCLUDE_DIR=${SWIFT_TOOLCHAIN_PATH}/usr/include
export SWIFT_BIN_DIR="${SWIFT_TOOLCHAIN_DIR}/bin" 
export PYTORCH_INSTALL_DIR=./pytorch

    # echo "export PATH=/root/.swiftly/bin:${SWIFT_BIN_DIR}:\$PATH" >> /etc/profile.d/swift.sh && \
    # New, correct line
    # echo "export PATH=/root/.local/share/swiftly/bin:${SWIFT_BIN_DIR}:\$PATH" >> /etc/profile.d/swift.sh && \
    # echo "export SWIFT_TOOLCHAIN_DIR=${SWIFT_TOOLCHAIN_DIR}" >> /etc/profile.d/swift.sh && \
    # echo "export SWIFT_TOOLCHAIN_PATH=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    # echo "export SWIFT_TOOLCHAIN=${SWIFT_TOOLCHAIN_PATH}" >> /etc/profile.d/swift.sh && \
    # echo "export SWIFTPM_SWIFT_EXEC=${SWIFT_BIN_DIR}/swift" >> /etc/profile.d/swift.sh && \
    # echo "export PYTORCH_INSTALL_DIR=/opt/pytorch" >> /etc/profile.d/swift.sh && \
    # echo "export CC=clang" >> /etc/profile.d/swift.sh && \
    # echo "export CXX=clang++" >> /etc/profile.d/swift.sh


#export SWIFT_TOOLCHAIN_DIR=$(which swift)
#export SWIFT_TOOLCHAIN_ROOT=${SWIFT_TOOLCHAIN_DIR%/bin/swift}

