# Tensor Module Overview

Swift-side tensor APIs live here. The folder layers lightweight, value-semantic wrappers on top of the C++ `TTSTensor` bridge so day-to-day code reads like Swift while retaining ATen performance. Each file extends `Tensor` (or related helpers) with a focused capability, covering construction, shape manipulation, indexing, math, async workflows, and ergonomic sugar.

Every public-facing symbol now carries Swift documentation comments so Xcode and IDE quick help reflect the same detail captured below.

## Core Wrapper (`Tensor.swift`)
- Stores the underlying `TTSTensor`, keeping the Swift façade `Sendable`.
- Canonical factories (`empty`, `zeros`, `ones`, `full`) plus scalar conveniences for the basic tensor type.
- Metadata access (`rank`, `shape`, `dtype`, `device`) and conversions across dtypes/devices via `to(...)` helpers.
- Minimal arithmetic (`adding`, scalar overloads) used by higher-level operator layers.

## Construction Helpers
- **`Tensor+Creation.swift`** – Random initialisers (`rand`, `randn`) and range utilities (`arange`, `linspace`).
- **`Tensor+FactoriesVariadic.swift`** – Variadic wrappers so you can call `Tensor.zeros(2, 3)` without allocating an array.
- **`Tensor+Literals.swift`** – Literal-style helper functions `tensor(_:shape:)` and varargs variants for quick array-backed tensors.
- **`Tensor+Builders.swift`** – Result builders (`tensor1D`, `tensor2D`) that assemble tensors from nested Swift literals with early shape validation.

## Host Interop & Serialization
- **`Tensor+ArrayIO.swift`** – Copies between row-major Swift arrays and tensors, with special handling for Bool.
  
- **`Tensor+HostBuffer.swift`** – `withHostBuffer` / `withMutableHostBuffer` utilities that lend contiguous CPU buffers to closures, copying only when necessary.
- **`CodableTensor.swift`** – Serializable wrapper that captures shape, dtype, device, and raw bytes for persistence or IPC.

## Shape, Layout, and Broadcasting
- **`Tensor+Broadcast.swift`** – `expanded`, `expanded(as:)`, and `broadcasted` helpers that follow PyTorch broadcasting semantics.
- **`Tensor+Shape.swift`** – Transpose, permute, reshape, flatten, squeeze, and unsqueeze.
- **`Tensor+ReshapeInference.swift`** – `reshaped(inferring:)` validates a single `-1` placeholder before delegating to reshape.
- **`Tensor+Unfold.swift`** – Sliding-window views over a dimension.
- **`Tensor+Layout.swift`** – Storage introspection (`count`, `strides`, `isContiguous`) plus `contiguous()`.
- **`Tensor+Description.swift`** – `CustomStringConvertible` implementation for tidy summaries of small tensors.

## Axis Typing & Collection Views
- **`Axis.swift`** – Strongly typed axis identifiers (`.batch`, `.channel`, `.last`, etc.) that resolve to integer dims per tensor rank.
- **`Tensor+AxisSugar.swift`** – Axis-aware overloads (`select(dim:)`, `narrow(dim:)`, `slice(dim:)`, `transposed`, `sum`, `mean`) that accept `Axis` instead of raw integers.
- **`Tensor1D.swift`** – `Tensor1D<T>` wrapper conforming to `RandomAccessCollection`, `MutableCollection`, and array literal protocols for 1-D tensors.
- **`Tensor+Collection.swift`** – Snapshot-style `TensorElements` collection for iterating over rank-1 tensors without mutating the backing storage.

## Indexing, Views, and Combination
- **`Tensor+Indexing.swift`** – Core selection (`select`, `narrow`, `slice`), rich subscripting forms, and chunking helpers (`split`, `chunk`).
- **`Tensor+AdvancedIndexing.swift`** – Host-side `indexSelect` and 2-D sugar subscripts.
- **`Tensor+MultiAxisSubscript.swift`** – `TensorIndex` enum and variadic subscript supporting integers, ranges, ellipsis, and new axes for NumPy-style slicing.
- **`Tensor+Join.swift`** – Concatenate (`cat`) and stack (`stack`) utilities.
- **`Tensor+Indexing+Differentiable.swift`** – Reverse-mode derivatives for `select`, `narrow`, and `slice`, ensuring gradients scatter back into the source tensor (`TensorIndexingDifferentiationTests`).

## Math, Logic, and Operators
- **`Tensor+Math.swift`** – Unary transforms, tensor/scalar binary ops, reductions (`sum`, `mean`, etc.), linear algebra (`matmul`, `dot`), and comparison helpers.
- **`Tensor+Mask.swift`** – Masked fills/selects plus boolean reductions (`any`, `all`, `nonzero`).
- **`Tensor+Reduce.swift`** – Dimensional `min`, `max`, `argmin`, `argmax`, and ranked selections (`topk`, `sort`) returning aligned value/index tensors via `TensorPair`.
- **`Tensor+Operators.swift`** – Conventional arithmetic operators (`+`, `-`, `*`, `/`) and compound assignments for tensor–tensor or tensor–scalar combinations.
- **`Tensor+ComparisonOperators.swift`** – Custom `.==`, `.<`, `.<=`, `.>`, `.>=` operators that keep element-wise semantics without clashing with Swift numerics.
- **`Tensor+Equatable.swift`** – `Equatable` conformance and `isClose` for tolerance-based equality checks.

## Differentiation Support
- **`Tensor+Differentiation.swift`** – Bridges `Tensor` into Swift's `Differentiable` and `AdditiveArithmetic` ecosystems with a simple tangent definition and `move(by:)`.
- **`Tensor+Differentiable.swift`** – Pullbacks for tensor–tensor and tensor–scalar `adding` overloads that collapse broadcasted gradients back to each operand.
- **`Tensor+Broadcast+Differentiable.swift`** – Derivatives for `expanded`, `expanded(as:)`, and `broadcasted`, implemented via `_reduceLike` and validated by `TensorBroadcastDifferentationTests`.
- **`Tensor+Math+Differentiable.swift`** – Reverse-mode derivatives for core unary math ops (`negated`, `abs`, `relu`, `exp`, `log`, `sqrt`) so AD aware code paths remain smooth (`TensorMathDifferentiationTests`).
- **`Tensor+Shape+Differentiable.swift`** – Pullbacks for transpose, permute, reshape, squeeze, and unsqueeze that restore the original layout while summing along broadcast axes (`TensorShapeDifferentiableTests`).

## Async & Device Utilities
- **`Tensor+Async.swift`** – Availability checks (`isAvailable`) and `moved(to:nonBlocking:)` async helper that hops transfers onto background queues and throws when devices are unavailable.

## Scalar and Numeric Bridges
- **`ScalarTensor.swift`** – `ScalarTensor<T>` wrapper that promotes single-element tensors to full `Comparable`, `Numeric`, and `AdditiveArithmetic` conformance for integration with Swift generics.

## Specialized Convenience Types
- **`Tensor1D.swift`** and **`Tensor+Collection.swift`** (see above) provide collection semantics for rank-1 tensors.
- **`Tensor+Builders.swift`** (above) adds DSL-like builders.

## Testing & Debugging
- **`Tests/TensorTests`** – Swift Testing suite covering factories, shape/broadcasting, indexing + differentiation, math/reductions, host interop, and operators.
- **`TemporaryDebugTests.swift`** – Ad-hoc debug cases kept alongside the main suite for quick repros while features stabilize.

## Recommendations for Even More Swifty Ergonomics
1. **Async sequences for streaming data** – Expose tensor batches as `AsyncSequence` (or `AsyncThrowingSequence`) to integrate seamlessly with structured concurrency.
2. **Zero-copy host views** – Once a safe `data_ptr` shim lands in C++, surface `MutableCollection` views that borrow CPU storage instead of copying, tightening parity with Swift’s inout patterns.
3. **Stronger literal support** – Extend result builders to higher ranks and infer element types automatically, making complex literals and ragged-shape diagnostics even friendlier.
4. **Protocol conformance expansion** – Explore adopting `AdditiveArithmetic`, `VectorProtocol`, or future Swift Numerics traits directly on `Tensor` (not just `ScalarTensor`) where semantics align, enabling reuse of generic numeric algorithms.
5. **Better axis inference** – Pair the `Axis` type with compile-time shape metadata (using Swift macros) so common operations can be statically checked and auto-completed.
