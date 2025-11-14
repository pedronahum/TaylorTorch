#!/bin/bash
export LIB_TORCH_VERSION=libtorch-shared-with-deps-2.8.0
export OUTPUT_PATH="./pytorch"

if [ -d "${OUTPUT_PATH}" ]; then
  echo "Removing existing ${OUTPUT_PATH}"
  rm -rf "${OUTPUT_PATH}"
fi

wget https://download.pytorch.org/libtorch/cpu/${LIB_TORCH_VERSION}%2Bcpu.zip && unzip ${LIB_TORCH_VERSION}+cpu.zip -d . && mv ./libtorch ${OUTPUT_PATH}
rm ${LIB_TORCH_VERSION}+cpu.zip
