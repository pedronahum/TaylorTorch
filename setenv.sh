#!/usr/bin/env bash
# Base toolchain configuration -------------------------------------------------
export SWIFT_TOOLCHAIN_PATH="$(swiftly use --print-location)"
export SWIFT_TOOLCHAIN_DIR="${SWIFT_TOOLCHAIN_PATH}/usr"
export CC="${SWIFT_TOOLCHAIN_DIR}/bin/clang"
export CXX="${SWIFT_TOOLCHAIN_DIR}/bin/clang++"
export SWIFT_TOOLCHAIN="${SWIFT_TOOLCHAIN_PATH}"
export SWIFT_BRIDGING_INCLUDE_DIR="${SWIFT_TOOLCHAIN_PATH}/usr/include"
export SWIFT_BIN_DIR="${SWIFT_TOOLCHAIN_DIR}/bin"
export PYTORCH_INSTALL_DIR="./pytorch"
