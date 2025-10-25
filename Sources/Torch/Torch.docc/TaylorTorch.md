# ``Torch``

Swift bindings for LibTorch with automatic differentiation support for building and training deep learning models.

## Overview

TaylorTorch provides idiomatic Swift bindings to PyTorch's LibTorch C++ library, enabling you to build, train, and deploy deep learning models entirely in Swift. With full integration into Swift's powerful automatic differentiation system, TaylorTorch brings type-safe, performant deep learning to the Swift ecosystem.

### Key Features

- **Native Swift API**: Idiomatic Swift interfaces over PyTorch's battle-tested implementations
- **Automatic Differentiation**: First-class integration with Swift's `@differentiable` attribute
- **Type Safety**: Compile-time checking prevents common deep learning errors
- **GPU Acceleration**: Full CUDA and MPS (Metal Performance Shaders) support
- **Rich Layer Library**: Pre-built layers for CNNs, RNNs, Transformers, and Graph Neural Networks
- **PyTorch Compatibility**: Seamlessly interoperate with PyTorch models and weights

## Quick Start

Build your first neural network in just a few lines:

```swift
import Torch

// Create a simple feedforward network
let model = Sequential {
    Linear(inputSize: 784, outputSize: 512)
    ReLU()
    Dropout(probability: 0.2)
    Linear(inputSize: 512, outputSize: 10)
}

// Training loop with automatic differentiation
let optimizer = Adam(parameters: model.parameters, learningRate: 0.001)

for epoch in 0..<numEpochs {
    // Forward pass
    let predictions = model(inputs)
    let loss = crossEntropy(predictions, targets)

    // Backward pass - Swift's autodiff computes gradients automatically
    let gradients = gradient(at: model) { m in
        m(inputs).loss(targets)
    }

    // Update weights
    optimizer.update(&model, along: gradients)
}
```

## Core Concepts

### Tensors

``Tensor`` is the fundamental data structure in TaylorTorch, representing multi-dimensional arrays of data. Tensors provide:

- Efficient storage on CPU, CUDA GPUs, or Metal
- Automatic broadcasting for element-wise operations
- Comprehensive mathematical operations
- Seamless integration with Swift's automatic differentiation

```swift
// Create tensors
let x = Tensor.zeros(shape: [2, 3], dtype: .float32)
let y = Tensor.randn(shape: [2, 3])

// Element-wise operations with automatic broadcasting
let z = x * 2.0 + y

// Reduction operations
let sum = z.sum()
let mean = z.mean(along: Axis(0))
```

### Layers and Modules

The ``Layer`` protocol defines the interface for all neural network components. Layers are:

- **Differentiable**: Automatic gradient computation through Swift's autodiff
- **Composable**: Build complex architectures from simple building blocks
- **Stateful**: Store learnable parameters like weights and biases

```swift
struct CustomLayer: Layer {
    var weight: Tensor
    var bias: Tensor

    init(inputSize: Int, outputSize: Int) {
        self.weight = Tensor.kaimingUniform([inputSize, outputSize])
        self.bias = Tensor.zeros([outputSize])
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        return input.matmul(weight) + bias
    }
}
```

### Sequential Model Composition

The ``Sequential`` container uses Swift's result builders to create elegant model definitions:

```swift
let resnetBlock = Sequential {
    Conv2D(inChannels: 64, outChannels: 64, kernelSize: (3, 3), padding: (1, 1))
    BatchNorm(featureCount: 64)
    ReLU()
    Conv2D(inChannels: 64, outChannels: 64, kernelSize: (3, 3), padding: (1, 1))
    BatchNorm(featureCount: 64)
}
```

### Optimization

TaylorTorch includes popular optimization algorithms for training neural networks:

```swift
// Stochastic Gradient Descent with momentum
let sgd = SGD(
    parameters: model.parameters,
    learningRate: 0.01,
    momentum: 0.9,
    weightDecay: 0.0001
)

// Adam optimizer
let adam = Adam(
    parameters: model.parameters,
    learningRate: 0.001,
    betas: (0.9, 0.999)
)
```

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:BuildingModels>
- ``Tensor``
- ``Layer``
- ``Sequential``

### Creating Tensors

- ``Tensor/empty(shape:dtype:device:)``
- ``Tensor/zeros(shape:dtype:device:)``
- ``Tensor/ones(shape:dtype:device:)``
- ``Tensor/full(_:shape:device:)``
- ``Tensor/randn(shape:dtype:device:)``
- ``Tensor/rand(shape:dtype:device:)``

### Tensor Properties and Operations

- ``Tensor/shape``
- ``Tensor/dtype``
- ``Tensor/device``
- ``Tensor/rank``
- ``Tensor/to(dtype:)``
- ``Tensor/to(device:)``

### Basic Layers

- ``Linear``
- ``Dense``
- ``Dropout``
- ``Flatten``
- ``Reshape``

### Activation Functions

- ``ReLU``
- ``GELU``
- ``SiLU``
- ``Sigmoid``
- ``Tanh``

### Convolutional Layers

- ``Conv1D``
- ``Conv2D``
- ``Conv3D``
- ``AvgPool2D``
- ``MaxPool2D``

### Normalization

- ``BatchNorm``
- ``LayerNorm``
- ``GroupNorm``

### Attention and Transformers

- ``MultiHeadAttention``
- ``Transformer``
- ``PositionalEncoding``

### Recurrent Layers

- ``LSTMCell``
- ``GRUCell``

### Graph Neural Networks

- <doc:GraphNeuralNetworks>
- ``GraphNetwork``
- ``GraphPooling``

### Optimization

- ``Optimizer``
- ``SGD``
- ``Adam``

### Loss Functions

- ``Loss``
- ``CrossEntropyLoss``
- ``MSELoss``

### Data Loading

- ``Dataset``
- ``DataLoader``
- ``MNISTDataset``
- ``CIFAR10Dataset``

### Examples

- <doc:Examples-MNIST>
- <doc:Examples-Translation>
- <doc:Examples-GraphNetworks>

### Advanced Topics

- <doc:AutomaticDifferentiation>
- <doc:CustomLayers>
- <doc:MigrationFromPyTorch>
