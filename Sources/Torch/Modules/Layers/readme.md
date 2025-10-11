# Layers

This folder now contains the distilled set of differentiable building blocks that ship with TaylorTorch. Every Swift file defines one or more types that conform to `Layer` (or `ParameterlessLayer`) and provide reverse-mode `callAsFunction(_:)` implementations paired with explicit pullbacks when broadcasting, masking, or view reshaping is involved.

## Files at a glance

| File | Highlights |
| --- | --- |
| `Activations.swift` | Smooth activation layers (ReLU, LeakyReLU, SiLU, ELU, GELU, Tanh, Sigmoid, Softplus, Softmax/LogSoftmax) plus differentiable helpers used across tests. |
| `BatchNorm.swift` | Feature-wise batch normalisation with running statistics, affine parameters, and training/inference behaviour driven by `withLearningPhase`. |
| `Dense.swift` | A lightweight MLP block (`Linear` + activation closure) that now uses the `[in, out]` weight layout to match the tests and examples. |
| `Dropout.swift` | Inverted dropout conforming to `ParameterlessLayer`, gated entirely by `Context.local.learningPhase`. |
| `GroupNorm.swift` | Group normalisation over arbitrary axes with shape-safe regrouping and hand-written pullbacks. |
| `LayerNorm.swift` | Per-sample normalisation across selectable axes; gradients stay stable thanks to explicit broadcasting logic. |
| `Linear.swift` | Core affine transform (`x.matmul(W) + b`) with a custom tangent vector so optimisers (`SGD`, `Adam`) can consume gradients directly. |
| `MultiHeadAttention.swift` | Transformer attention primitive: head splitting/combining utilities, differentiable `Input` wrappers, and numerically stable softmax. |
| `NNLayers.swift` | Convolution/pooling utilities (`Conv2D`, `MaxPool2D`, `AvgPool2D`, `Flatten`) exposing parameterless tangents while driving ATen kernels. |
| `Recurrent.swift` | RNN cell scaffolds plus `BasicRNNCell`, `LSTMCell`, and `GRUCell` implementations with mergeable state tangents. |
| `Sequential.swift` | Building blocks for chaining layers driven by the `@SequentialBuilder` result builder, wrapping a single generic `Sequential<Body>`. |

## Implementation notes

- **Parameterless layers** (Dropout, MaxPool2D, AvgPool2D, Flatten, Identity) explicitly set `TangentVector == EmptyTangentVector`, ensuring their pullbacks align with the `Layer` protocol.
- **Shape/device reads** that are only needed for validation are wrapped in `withoutDerivative(at:)`, keeping the AD graph lean while still allowing runtime checks.
- **Manual pullbacks** implemented via `valueWithPullback` appear wherever tensor views, masking, or scatter-style ops would otherwise lose gradient information (attention, normalisation, Softplus, pooling).

## Quick example

```swift
import Torch

let encoder = Sequential {
  Linear(inputSize: 128, outputSize: 256)
  Activations.relu
  LayerNorm(featureCount: 256)
  Dropout(probability: 0.1)
}

let x = Tensor.randn(shape: [32, 128])
let y = encoder(x)  // differentiable end-to-end
```

Together these sources form the essential toolkit—dense blocks, attention, recurrent state, convolution + pooling, normalisation, activations, dropout, and sequential composition—ready to plug into `GraphNetwork`, Transformers, or any custom TaylorTorch model.
