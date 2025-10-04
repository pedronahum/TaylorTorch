# Layers

TaylorTorch’s `Modules/Layers` folder houses the differentiable building blocks used to assemble models. Every layer conforms to the common `Layer` protocol, meaning parameters are discoverable through key-path traversal, gradients flow by default, and context-aware variants can react to `ForwardContext(training:)` for behaviour that differs between training and evaluation.

## Using the layers

```swift
import Torch

let ctx = ForwardContext(training: true)

var linear = Linear(weight: W, bias: b)
let logits = linear(inputs)

var bn = BatchNorm1D(numFeatures: logits.shape[1])
let normalized = bn.call(logits, context: ctx) // updates running stats only when ctx.training == true

var block = Sequential(
  Linear(weight: l1W, bias: l1B),
  Dense(inFeatures: logits.shape[1], outFeatures: 10, activation: Activations.relu)
)
let predictions = block(normalized, context: ctx)
```

`ForwardContext` threads through any type that implements the contextual `call(_:context:)` overload (BatchNorm, Dropout, Sequential, Dense, etc.). Inference code can continue to rely on the pure `callAsFunction(_:)` entry point when training-specific state updates are not required.

## File tour

- `BatchNorm.swift` – Implements `BatchNorm1D` and `BatchNorm2D` with affine parameters, running mean/variance tracking, and custom reverse-mode rules so training updates running statistics while inference uses frozen values. Supports both NCHW and NHWC layouts via `DataFormat`.
- `Conv2D.swift` – Defines the 2D convolution layer with configurable stride, padding, dilation, groups, and layout (`DataFormat`). Includes parameter key-paths and AD plumbing so it integrates with optimizers and Sequential combinators.
- `DataFormat.swift` – Shared enum describing tensor memory layouts (`.nchw`, `.nhwc`) used by convolution, pooling, and normalization layers for consistent layout transforms.
- `Dense.swift` – Wraps a `Linear` layer with a differentiable activation closure. Provides convenience initialisers (including Glorot) and an `Activations` namespace with common differentiable functions.
- `Dropout.swift` – Inverted-dropout layer that samples masks during training, scales survivors by `1 / (1 - rate)`, and becomes the identity for inference. Accepts an optional deterministic mask factory for testing.
- `Linear.swift` – Core fully connected layer exposing matrix multiplication plus bias. Serves as the building block for `Dense`, `Sequential`, and custom modules.
- `Pooling.swift` – Contains `MaxPool2D` and `AvgPool2D` layers with stride, padding, dilation, ceil-mode, and layout control. Handles NHWC inputs by permuting to the NCHW kernels exposed by ATen.
- `Sequential.swift` – Generic two-layer composition that preserves parameter traversal and differentiability, making it easy to chain smaller layers into blocks.
- `Sequential+Context.swift` – Extends `Sequential` so the contextual `call(_:context:)` cascades through both sublayers, letting dropout, batch norm, and other context-aware modules cooperate inside composite stacks.

## Patterns to keep in mind

- Layers that mutate internal state (BatchNorm) wrap that state in `_TensorBox` so mutations stay outside Swift’s value semantics while remaining differentiable where appropriate.
- If a layer supports both pure and contextual calls, prefer `layer(x, context: ctx)` during training and `layer(x)` for inference to avoid unnecessary state updates.
- All layers expose parameter key-paths via `ParameterIterable`, so optimizers supplied elsewhere in TaylorTorch can traverse and update weights without extra boilerplate.

This directory therefore provides everything needed to assemble common MLP and CNN blocks while keeping differentiable programming ergonomics consistent across the project.
