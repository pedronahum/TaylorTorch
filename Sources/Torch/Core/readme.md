# Core Primitives

The files in this directory define the protocols and algebra that make TaylorTorch models composable and optimisable.

- **ParameterIterable.swift** – Declares the key-path based traversal protocol used to enumerate tensors inside a model. Both models and their tangent vectors conform so optimisers can walk a structure once and reuse the same paths.
- **EuclideanModel.swift** – Builds on `ParameterIterable` to provide default implementations for tangent zero, in-place `move(by:)`, and `flattenedParameters()`. Any layer or model that adopts `EuclideanModel` automatically gains differentiable parameter arithmetic.
- **TangentVectorAlgebra.swift** – Shared helpers for combining tangent vectors, including element-wise addition/subtraction and convenience initialisers.

If you introduce a new container type (e.g. custom module or optimiser state), conform it to these protocols so that gradients and parameter updates continue to flow uniformly through the system.
