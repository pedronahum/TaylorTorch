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

# Step 2: Locate compiler and toolchain directories ---------------------------
export CLANG_RESOURCE_DIR="$(clang++ -print-resource-dir)"
export LLVM_BASE_DIR="$(realpath "${CLANG_RESOURCE_DIR}/../../..")"
echo "Clang resource dir: ${CLANG_RESOURCE_DIR}"
echo "LLVM base dir:      ${LLVM_BASE_DIR}"
echo "Swift toolchain dir: ${SWIFT_TOOLCHAIN_DIR}"

# Step 3: Locate libc++ headers ------------------------------------------------
libcxx_include_candidates=(
  "${SWIFT_TOOLCHAIN_DIR}/lib/swift/clang/include/c++/v1"
  "${SWIFT_TOOLCHAIN_DIR}/include/c++/v1"
  "${SWIFT_TOOLCHAIN_DIR}/lib/swift_static/clang/include/c++/v1"
  "${CLANG_RESOURCE_DIR}/../../../include/c++/v1"
)

LIBCXX_INCLUDE_DIR=""
LIBCXX_SOURCE="Swift toolchain"
for candidate in "${libcxx_include_candidates[@]}"; do
  if [[ -d "${candidate}" && -f "${candidate}/cstddef" ]]; then
    LIBCXX_INCLUDE_DIR="$(realpath "${candidate}")"
    echo "✓ Found libc++ in Swift toolchain at ${LIBCXX_INCLUDE_DIR}"
    break
  fi
done

if [[ -z "${LIBCXX_INCLUDE_DIR}" ]]; then
  echo "ℹ Swift toolchain lacks libc++, searching system LLVM 17..."
  LIBCXX_SOURCE="system LLVM 17"
  system_candidates=(
    "/usr/lib/llvm-17/include/c++/v1"
    "/usr/include/c++/v1"
  )
  for candidate in "${system_candidates[@]}"; do
    if [[ -d "${candidate}" && -f "${candidate}/cstddef" ]]; then
      LIBCXX_INCLUDE_DIR="$(realpath "${candidate}")"
      echo "✓ Found libc++ headers at ${LIBCXX_INCLUDE_DIR}"
      break
    fi
  done
fi

if [[ -z "${LIBCXX_INCLUDE_DIR}" ]]; then
  echo "ERROR: Unable to locate libc++ headers. Please install libc++-17-dev." >&2
  exit 1
fi

# # Step 4: Locate libc++ libraries ---------------------------------------------
declare -a libcxx_library_candidates=()
if [[ "${LIBCXX_SOURCE}" == "Swift toolchain" ]]; then
  libcxx_library_candidates+=(
    "${SWIFT_TOOLCHAIN_DIR}/lib/swift/linux"
    "${SWIFT_TOOLCHAIN_DIR}/lib"
  )
fi
libcxx_library_candidates+=(
  "/usr/lib/llvm-17/lib"
  "/usr/lib/x86_64-linux-gnu"
  "/usr/lib"
)

LIBCXX_LIBRARY_DIR=""
for candidate in "${libcxx_library_candidates[@]}"; do
  if [[ -d "${candidate}" ]] && compgen -G "${candidate}/libc++.so*" > /dev/null; then
    LIBCXX_LIBRARY_DIR="$(realpath "${candidate}")"
    echo "✓ Found libc++ libraries at ${LIBCXX_LIBRARY_DIR}"
    break
  fi
done

if [[ -z "${LIBCXX_LIBRARY_DIR}" ]]; then
  echo "ERROR: Unable to locate libc++ libraries. Please install libc++-17-dev." >&2
  exit 1
fi

export LIBCXX_INCLUDE_DIR
export LIBCXX_LIBRARY_DIR
export LIBCXX_SOURCE
export USE_LIBCXX=1
echo "libc++ include dir: ${LIBCXX_INCLUDE_DIR}"
echo "libc++ library dir: ${LIBCXX_LIBRARY_DIR}"
