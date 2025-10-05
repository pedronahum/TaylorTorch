# Builders

The builder DSL provides a lightweight way to author composite modules without writing nested `Sequential` initialisers by hand.

- **LayerBuilder.swift** – Declares the `LayerBuilder` result builder and the `SequentialBlock` wrapper. Using it lets you write:
  ```swift
  let block = SequentialBlock {
    Linear(weight: w1, bias: b1)
    ReLU()
    Linear(weight: w2, bias: b2)
  }
  ```
  The builder expands the block into nested `Sequential` types while preserving parameter traversal and differentiability.

Extend this file if you need additional arities or custom combinators that should participate in the same DSL.
