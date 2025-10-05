# Optimizers

TaylorTorch ships a small optimiser toolkit that operates on any `EuclideanModel`.

- **Optimizers.swift** – Implements SGD, Adam, and supporting update routines. Each optimiser accepts references to model parameters via `ParameterIterable`, keeps lightweight per-parameter state, and exposes a `step(_:)` method that applies tangents produced by reverse-mode autodiff.

Because optimisers consume the same traversal metadata as layers, you can plug them into custom training loops without worrying about model structure. Extend this file when you need additional optimisation algorithms.
