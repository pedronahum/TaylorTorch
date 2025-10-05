# Modules Overview

The `Modules` subtree assembles everything above core tensor primitives into reusable differentiable components. Public entry points live here, while specialised folders keep related concerns grouped together:

- **Layer.swift** – Declares the `Layer` protocol that every trainable component adopts. It extends `EuclideanModel`, exposes both pure and contextual calls, and ensures parameters are discoverable via key paths.
- **Initializers.swift** – Shared tensor initialiser routines (e.g. Glorot) used by higher-level modules when constructing weights.
- **Layers/** – Concrete trainable layers and shape utilities (linear blocks, convolutions, embeddings, activations, normalisation, flatten/reshape/permute, squeeze/unsqueeze, etc.). See its README for the full catalogue.
- **Combinators/** – Higher-order wiring helpers (`Residual`, `ParallelAdd`, `Concat`, `Identity`) that operate on other layers while staying differentiable.
- **Builders/** – The `LayerBuilder` DSL that turns declarative block syntax into nested `Sequential` pipelines.
- **Context/** – Defines `ForwardContext`, the training/inference flag threaded through contextual layer calls.
- **Shape/** – Reserved slot for non-layer shape helpers should they be needed in future.

Most downstream code interacts with this directory: import `Torch`, pick the layers you need, and compose them using combinators or builder blocks. Optimisers and core algebra (found elsewhere) operate uniformly on any module defined here because every type conforms to the shared `Layer` contract.
