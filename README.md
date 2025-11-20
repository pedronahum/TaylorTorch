<img src="assets/TaylorTorch_logo_horizontal_dark_v4.png" alt="TaylorTorch" width="1000">

[![macOS CI](https://github.com/pedronahum/TaylorTorch/actions/workflows/macos-ci.yml/badge.svg)](https://github.com/pedronahum/TaylorTorch/actions/workflows/macos-ci.yml)
[![Ubuntu CI](https://github.com/pedronahum/TaylorTorch/actions/workflows/ubuntu-ci.yml/badge.svg)](https://github.com/pedronahum/TaylorTorch/actions/workflows/ubuntu-ci.yml)
[![Documentation](https://github.com/pedronahum/TaylorTorch/actions/workflows/deploy-docc.yml/badge.svg)](https://pedronahum.github.io/TaylorTorch/documentation/torch/)

# Introduction
TaylorTorch is a modern Swift wrapper for LibTorch, designed to bring the elegance of Swift to the power of PyTorch's C++ backend. It is built from the ground up to feel idiomatic to Swift developers and to leverage the language's first-class automatic differentiation capabilities for seamless gradient computation. By embracing a protocol-oriented design, TaylorTorch provides a flexible “front-to-back” experience so you can compose complex models in pure Swift.

# Project Motivation and Vision
The journey of TaylorTorch begins with a deep appreciation for the pioneering work of the Swift for TensorFlow (S4TF) project. S4TF first introduced Differentiable Programming as a first-class feature in Swift, and although the project is now archived, its ambitious vision left a lasting impact on the community.

Fortunately, the story of automatic differentiation in Swift didn't end there. Thanks to the continued efforts of the Swift community and the dedicated team at PassiveLogic, the language's auto-diff capabilities have not only been preserved but have matured and strengthened significantly.

Witnessing these advancements, and in an era of rapid progress with technologies like Large Language Models (LLMs), I set a personal challenge: to see if the original vision of S4TF—a powerful, end-to-end deep learning framework in Swift—could be resurrected. TaylorTorch is the result of that challenge, aiming to rebuild that capability but this time powered by the robust and widely-used LibTorch backend.

This project is a foundational layer, and its future can be extended in countless ways—from adding higher-level APIs like torch.nn to developing integrations with other scientific computing libraries. The roadmap is intentionally open, as I eagerly await feedback, contributions, and ideas from the Swift community to guide its evolution.

# Features & Status: Experimental, but Batteries Included 🔋

While TaylorTorch is an experimental alpha release, it comes packed with features to let you start building powerful models right away. The core goal is to provide comprehensive access to the underlying Torch ATen tensor functionalities directly within Swift, giving you a massive library of operations at your fingertips.

This release includes a robust set of common neural network layers, ready for you to compose your own architectures:

* Core Layers: Linear, Conv (1D, 2D, 3D), Dropout

* Normalization: BatchNorm, LayerNorm, GroupNorm

* Attention Mechanisms: Multi-Head Attention, SiLU/ELU/GELU activations

* Recurrent Cells: LSTMCell, GRUCell

### Layers API overview

TaylorTorch adopts Swift’s `Layer` protocol (an alias of `Module`) as the
foundation for the high-level API. Every layer that you will find under
`Sources/Torch/Modules/Layers` is both differentiable and composable:

* **Initialisation** – weight tensors now follow the `[in, out]` convention for
  dense projections (`Linear`, `Dense`) and `[Cin, Cout]` style shapes for
  convolutional blocks, mirroring the semantics used across the tests.
* **Builder syntax** – `Sequential` is a single generic type that accepts any
  `Layer` instance. The nested chains created by the result builder eliminate
  the need to manually spell out `Sequential<Chain<…>>`. The examples and tests
  now use

  ```swift
  let head = Sequential {
    Dense(inputSize: 70, outputSize: 64)
    ReLU()
    Dense(inputSize: 64, outputSize: 32)
  }
  ```

  which matches the API available inside `GraphNetwork`, Transformers, and the
  examples.
* **Differentiable semantics** – most layers provide handwritten
  `TangentVector`s to keep Swift’s autodiff compiler happy. When you compose
  layers in `Sequential` or within more specialised modules (e.g. the graph
  stack in the Karate example), their gradients can be consumed by optimisers
  such as `SGD` or `Adam` directly.
* **Graph utilities** – the layers in `Modules/Graph` include batching helpers,
  GNN-style message passing (`GraphNetwork`), and pooling utilities. They expect
  host indices (Swift arrays) for sender/receiver lists, and the README examples
  demonstrate the required shapes.

## First-Class Graph Learning
Beyond standard layers, TaylorTorch includes initial building blocks for Graph Neural Networks (GNNs). Inspired by the excellent work from DeepMind's Graph Nets and Jraph libraries, these components are designed to make graph-based machine learning a first-class citizen in the Swift ecosystem.

## Hands-On Examples
To help you get started, the library includes several working examples that showcase its capabilities:

* MNIST: A classic image classification task using convolutional layers (Conv2d).

* ANKI English to Spanish Translation: A sequence-to-sequence model demonstrating the use of multi-head attention.

* Karate Club Classification: A 2-class node classification problem solved with Graph Networks, showing how to leverage the GNN components.

These examples serve as a practical guide and a starting point for your own projects.

# System Requirements & Installation Notes ⚠️

Please read this section carefully before building the project.

## Supported Platforms

TaylorTorch has been tested on the following configurations:

| Platform | Swift Version | PyTorch | Status |
|----------|---------------|---------|--------|
| **macOS** (Apple Silicon) | 6.3-dev (main-snapshot-2025-11-03) | 2.8.0 (CPU) | ✅ Fully supported |
| **Ubuntu 24.04** | 6.3-dev (main-snapshot-2025-11-03) | 2.8.0 (CPU) | ✅ Supported (with [known issues](KNOWN_ISSUES.md)) |

GPU support (CUDA/Metal) is not yet implemented.

## Installation

### macOS

For macOS installation, refer to the [GitHub Actions workflow](.github/workflows/macos-ci.yml) which demonstrates the complete build process including:
- Installing Swift development snapshots via Swiftly
- Building PyTorch from source with compatible compilers
- Configuring environment variables

### Ubuntu 24.04

For Ubuntu, we provide an automated installation script:

```bash
./scripts/install-taylortorch-ubuntu.sh
```

This script handles:
- Installing all system dependencies
- Installing Swift via Swiftly
- Building PyTorch from source with the correct compiler (Swift's clang with libstdc++)
- Configuring environment variables

See [scripts/README.md](scripts/README.md) for detailed usage options and troubleshooting.

For CI/Docker environments, refer to the [Ubuntu CI workflow](.github/workflows/ubuntu-ci.yml) and the [Dockerfile](Dockerfile).

## Required Environment Variables

Before building TaylorTorch, you must set these environment variables:

```bash
export SWIFT_TOOLCHAIN_DIR="/path/to/swiftly/toolchains/main-snapshot-2025-11-03/usr"
export PYTORCH_INSTALL_DIR="/opt/pytorch"  # or your PyTorch install location
export PATH="/path/to/swiftly/bin:$PATH"
```

On Ubuntu, these are automatically configured when you source the environment files created by the install script:

```bash
source /etc/profile.d/swift.sh
source /etc/profile.d/pytorch.sh
```

## Known Issues

Linux builds may encounter some Swift autodiff-related issues. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for:
- SIL linker crashes with C library math functions (exp, log, sqrt, pow)
- Autodiff crashes with for-in loops
- Adam optimizer KeyPath issues with complex models

These issues are specific to Linux and do not affect macOS builds.

# What does TaylorTorch looks like?

```swift
import Torch

var model = Sequential {
    // ── Block 1 ──────────────────────────────────────────────────────────────
    Conv2D(
      kaimingUniformInChannels: 1, outChannels: 32,
      kernelSize: (3, 3), padding: (1, 1))
    Dropout(probability: 0.025)  
    BatchNorm(featureCount: 32)  // NCHW => axis 1
    ReLU()
    AvgPool2D(kernelSize: (2, 2), stride: (2, 2))

    // ── Block 2 ──────────────────────────────────────────────────────────────
    Conv2D(
      kaimingUniformInChannels: 32, outChannels: 64,
      kernelSize: (3, 3), padding: (1, 1))
    BatchNorm(featureCount: 64)
    ReLU()
    AvgPool2D(kernelSize: (2, 2), stride: (2, 2))

    // ── Head ─────────────────────────────────────────────────────────────────
    Flatten(startDim: 1)  // [N, 64*7*7] = [N, 3136]
    Linear(inputSize: 64 * 7 * 7, outputSize: 256)
    Dropout(probability: 0.025)
    ReLU()
    Linear(inputSize: 256, outputSize: 10)
  }
```

```swift
import Torch

let model = Sequential {
  Dense(inputSize: 3, outputSize: 2)
  ReLU()
}

let x = Tensor(array: [
  0.5, -1.0, 2.0,
  1.5,  0.0, -0.5,
], shape: [2, 3], dtype: .float64)
let y = model(x)
print(y)
```

# What's Next? (The Unreleased Tracks) 🚀

TaylorTorch is a passion project developed in my free time, which means balancing development with important things like sleep and sports! The library has many potential avenues for growth, and the path forward will be heavily influenced by community feedback and contributions.

Here are some of the exciting directions we could explore:

* Expanded Operator Coverage: Systematically increase the number of covered ATen tensor operators to provide more comprehensive access to the LibTorch backend.

* Robust Test Coverage: Implement a thorough testing suite for both the Swift front-end and the C++ bridging code to ensure stability and reliability.

* GPU / Metal Support: Unlock high-performance training and inference by adding support for GPU acceleration via Metal on Apple Silicon, which would be a game-changer for larger models.

* Richer Model Zoo: Add more advanced layers and end-to-end examples, such as a full Transformer or a Vision Transformer (ViT).

* Ecosystem Interoperability: Integrate support for standards like DLPack to allow for zero-copy tensor sharing with other libraries (like NumPy or JAX), making it easier to use weights and data from different ecosystems.

* Exploring New Backends: Investigate a potential future version that moves away from LibTorch and instead uses a more native backend like Apple's MLX (Swift) for a deeply integrated experience on Apple hardware.

If any of these ideas excite you, feel free to open an issue or submit a pull request. Your contributions are what will shape the next era of TaylorTorch!

# A Testbed for Swift's Automatic Differentiation

Developing TaylorTorch has been a fascinating journey into the depths of Swift's automatic differentiation system. The library's multi-nested architecture, combined with its heavy reliance on the Differentiable protocol, makes it a powerful stress test for the Swift compiler.

Along the way, I encountered and resolved several interesting SIL (Swift Intermediate Language) generation issues related to differentiation. A key insight was that for many complex Differentiable structs, compiler-related problems could be consistently solved by explicitly implementing the associated TangentVector.

Given its structure, this library can serve as a valuable test suite for the Swift community. It provides a real-world, complex use case to help identify, debug, and improve the compiler's handling of automatic differentiation. The ultimate goal is to contribute to making differentiable programming a more robust and first-class citizen in the Swift ecosystem.
