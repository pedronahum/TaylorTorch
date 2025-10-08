# Layers

TaylorTorch’s `Modules/Layers` directory houses the differentiable building blocks used to compose models. Every type conforms to the shared `Layer` protocol, so:

- Parameters are discoverable through key-path traversal (`ParameterIterable`).
- Reverse-mode derivatives are available out of the box (custom pullbacks are supplied where broadcasting or masking require them).
- Context-aware variants can react to `ForwardContext(training:)`, while the pure `callAsFunction(_:)` entry point stays deterministic and side-effect free for inference.

```swift
import Torch

let ctx = ForwardContext(training: true)

var embed = Embedding(numEmbeddings: 10, embeddingDim: 4)
var block = Sequential(
  embed,
  Permute([0, 2, 1]),
  Flatten(startDim: 1),
  Linear(weight: W, bias: b)
)

let logits = block(tokens, context: ctx)
```

## Core feed-forward blocks

- `Linear.swift` – Dense affine transformation (`x.matmul(W.T) + b`) with differentiable parameter updates.
- `Dense.swift` – Convenience wrapper around `Linear` that wires in an activation closure (ReLU, GELU, etc.) and ships Glorot initialisers.
- `Sequential.swift` / `Sequential+Context.swift` – Generic two-layer composition. The contextual overload makes composite stacks (Dropout → BatchNorm → Linear) respect `ForwardContext`.
- `Embedding.swift` – Trainable lookup table that gathers integer IDs into dense vectors, supports optional `paddingIndex`, and scatters gradients with accumulation semantics.

## Convolution and pooling

- `Conv2D.swift` – 2D convolution with stride, padding, dilation, groups, and configurable data format (NCHW/NHWC).
- `DepthwiseSeparableConv2D.swift` – MobileNet-style depthwise + pointwise stack with shared `ForwardContext` support and optional fused activations.
- `Pooling.swift` – `MaxPool2D` and `AvgPool2D` layers that mirror the ATen operators while providing context-aware composition.
- `DataFormat.swift` – Shared enum describing tensor memory layouts so convolution, pooling, and normalization layers agree on axis order.

## Normalisation suites

- `BatchNorm.swift` – `BatchNorm1D`/`2D` with affine parameters, running statistics, and contextual behaviour (updates only while `training == true`).
- `LayerNorm.swift` – Stateless per-sample normalisation across trailing axes with custom VJP to handle broadcasting.
- `GroupNorm.swift` – Channel grouping normalisation with NHWC/NCHW support and padding-aware gradients.
- `RMSNorm.swift` – Root-mean-square normalisation with learnable scale/offset, commonly used in transformer blocks.

## Activation layers

- `Activations.swift` – Parameter-free wrappers for smooth activations (ReLU, Tanh, Sigmoid, SiLU, Softplus, GELU exact/approx). Exposed both as functional helpers (`Activations.relu(x)`) and `Layer` types (e.g. `ReLU()`).
- `HardActivations.swift` – Piecewise-linear variants (`HardTanh`, `HardSigmoid`, `HardSwish`) plus `ELU`, all differentiable and compatible with sequential composition.

## Shape & layout utilities

- `Flatten.swift` – Collapses a contiguous range of dimensions (default: all but batch) while keeping AD-friendly reshaping.
- `Reshape.swift` – Stateless reshape that supports a single `-1` inferred dimension and validates element counts.
- `Permute.swift` – Axis reordering layer with permutation validation and inverse-gradient semantics.
- `Squeeze.swift` / `Unsqueeze` – Drop or insert singleton dimensions, handy for glueing CNN and MLP components.

## Miscellaneous helpers

- `Dropout.swift` – Inverted dropout with optional deterministic mask factory for testing.
- `GroupNorm.swift`, `LayerNorm.swift`, `RMSNorm.swift` (see above) – provide the smoothing necessary for transformer-style architectures.
- `Identity` lives alongside these helpers (see `Modules/Combinators`) for no-op placeholders and residual wiring.

## Support folders inside `Modules/`

- `Builders/` – Contains `LayerBuilder.swift`, the result-builder DSL that lowers declarative blocks into nested `Sequential` chains.
- `Combinators/` – Higher-order modules (`Identity`, `Residual`, `ParallelAdd`, `Concat`) for wiring together submodels without leaving the differentiable world.
- `Context/` – Defines `ForwardContext`, the lightweight training/inference flag that contextual layers use to toggle behaviour.
- `Shape/` – Reserved for additional pure shape helpers that complement the layer-level utilities above.

## Working with ForwardContext

Any layer implementing the contextual `call(_:context:)` overload can respond to training/eval mode. Prefer `layer(x, context: ctx)` inside training loops and the pure `layer(x)` otherwise. Sequential containers forward `ForwardContext` automatically so deeper stacks stay in sync.

## Implementation patterns

- Stateful layers wrap mutable tensors in `_TensorBox` to remain differentiable while bypassing Swift’s copy-on-write semantics.
- Gradient-critical utilities (GELU, RMSNorm, Embedding, etc.) supply custom VJPs so broadcasting, scatter-add, or masking semantics are handled explicitly.
- All layers provide a `TangentVector` that mirrors stored parameters; optimisers supplied elsewhere in TaylorTorch use these key paths to perform updates without bespoke plumbing.

This collection covers the common NN toolbox—linear blocks, embeddings, activations, normalisation, convolution/pooling, and shape transforms—while maintaining “Swift native” ergonomics for differentiable programming.
