# ATen Bridge

This directory contains Swift helpers that sit directly on top of the C++ ATen shims located in `Sources/ATenCXX/`.

- **Core/Device.swift** – Describes the Swift-facing device abstraction (`Device` enum) that mirrors ATen’s CPU/CUDA/MPS selectors. Layers and tensor factories use it to place data on the correct accelerator.

Most of the heavy lifting is performed by the generated headers in `Sources/ATenCXX`. Swift files here provide ergonomic wrappers (computed properties, small extensions) so higher-level modules can remain Swift-native while delegating computation to libtorch.
