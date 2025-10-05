# Torch Module Overview

Swift sources under `Sources/Torch/` form the public façade of TaylorTorch. They wrap low-level ATen bindings, expose differentiable building blocks, and co-ordinate parameter traversal plus optimiser support. The directories below explain the division of responsibilities:

- **ATen/** – lightweight Swift shims over the C++ ATen runtime (device descriptors, tensor convenience APIs, bridging helpers).
- **Core/** – shared protocols (`ParameterIterable`, `EuclideanModel`) and tangent-algebra utilities that make every layer optimisable and differentiable.
- **Modules/** – user-facing differentiable components: layers, combinators, builder DSL helpers, and context plumbing.
- **Optimizers/** – first-party optimiser implementations that operate on any `EuclideanModel` using the core traversal primitives.

Each subfolder contains its own README describing the files in detail. Start with `Modules/` to explore trainable layers, or dive into `Core/` if you need to add new differentiable model primitives.
