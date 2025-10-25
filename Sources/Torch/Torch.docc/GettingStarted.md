# Getting Started with TaylorTorch

Build your first deep learning model in Swift with TaylorTorch.

## Overview

This guide walks you through installing TaylorTorch, understanding its core concepts, and building your first neural network. By the end, you'll have a working model that you can train and evaluate.

## Installation

### Swift Package Manager

Add TaylorTorch to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/pedronahum/TaylorTorch.git", from: "1.0.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["Torch"]
)
```

### Prerequisites

TaylorTorch requires:
- Swift 5.9 or later
- LibTorch 2.0+ (automatically linked)
- CUDA 11.8+ (optional, for GPU support)
- Metal (automatic on macOS for GPU acceleration)

## Your First Neural Network

Let's build a simple neural network to classify handwritten digits from the MNIST dataset.

### Step 1: Import TaylorTorch

```swift
import Torch
```

### Step 2: Create the Model

Use the ``Sequential`` builder to compose layers:

```swift
let model = Sequential {
    // Flatten 28x28 images to vectors
    Flatten(startDim: 1)

    // Hidden layer with 128 neurons
    Linear(inputSize: 784, outputSize: 128)
    ReLU()

    // Dropout for regularization
    Dropout(probability: 0.2)

    // Output layer with 10 classes (digits 0-9)
    Linear(inputSize: 128, outputSize: 10)
}
```

### Step 3: Prepare the Data

Load and preprocess the MNIST dataset:

```swift
// Load MNIST data
let dataset = try MNISTDataset(
    trainTest: .train,
    batchSize: 32,
    flattening: false  // Keep as 28x28 images
)

let dataLoader = DataLoader(dataset: dataset, batchSize: 32, shuffle: true)
```

### Step 4: Set Up Training

Configure the optimizer and loss function:

```swift
// Create optimizer
let optimizer = Adam(
    parameters: model.parameters,
    learningRate: 0.001
)

// Training loop
for epoch in 1...5 {
    var epochLoss: Float = 0
    var batchCount = 0

    for batch in dataLoader {
        let (images, labels) = batch

        // Forward pass
        let predictions = model(images)
        let loss = crossEntropy(predictions, labels)

        // Backward pass using Swift's autodiff
        let gradients = gradient(at: model) { m in
            let pred = m(images)
            return crossEntropy(pred, labels)
        }

        // Update weights
        optimizer.update(&model, along: gradients)

        epochLoss += loss.scalarValue
        batchCount += 1
    }

    let avgLoss = epochLoss / Float(batchCount)
    print("Epoch \(epoch): Loss = \(avgLoss)")
}
```

### Step 5: Evaluate the Model

Test your trained model:

```swift
// Switch to evaluation mode
model.inferring(from:)

// Load test data
let testDataset = try MNISTDataset(trainTest: .test, batchSize: 100)
var correct = 0
var total = 0

for batch in testDataset {
    let (images, labels) = batch

    // Get predictions
    let predictions = model(images)
    let predicted = predictions.argmax(axis: 1)

    // Calculate accuracy
    correct += (predicted == labels).sum().scalarValue
    total += labels.shape[0]
}

let accuracy = Float(correct) / Float(total) * 100
print("Test Accuracy: \(accuracy)%")
```

## Understanding the Components

### Tensors

``Tensor`` is the fundamental data structure, representing multi-dimensional arrays:

```swift
// Create a 2x3 tensor
let x = Tensor([[1, 2, 3],
                [4, 5, 6]], dtype: .float32)

// Shape operations
print(x.shape)  // [2, 3]
print(x.rank)   // 2

// Element-wise operations
let y = x * 2 + 1
let z = x.matmul(y.transpose())
```

### Layers

Layers implement the ``Layer`` protocol and define the ``callAsFunction(_:)`` method:

```swift
struct MyLayer: Layer {
    var weight: Tensor

    init(size: Int) {
        self.weight = Tensor.randn([size, size])
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        return input.matmul(weight)
    }
}
```

The `@differentiable` attribute enables automatic differentiation - Swift automatically computes gradients for backpropagation.

### Optimizers

Optimizers update model parameters based on computed gradients:

```swift
// SGD with momentum
let sgd = SGD(
    parameters: model.parameters,
    learningRate: 0.01,
    momentum: 0.9
)

// Adam (adaptive learning rates)
let adam = Adam(
    parameters: model.parameters,
    learningRate: 0.001
)
```

## Next Steps

Now that you've built your first model, explore more advanced topics:

- <doc:BuildingModels> - Learn about different layer types and architectures
- <doc:AutomaticDifferentiation> - Deep dive into Swift's autodiff system
- <doc:Examples-MNIST> - Complete MNIST CNN example
- <doc:Examples-Translation> - Build a sequence-to-sequence translator
- <doc:GraphNeuralNetworks> - Work with graph-structured data

## Common Patterns

### GPU Acceleration

Move your model and data to GPU for faster training:

```swift
// Move model to CUDA
let model = model.to(device: .cuda(0))

// Move data to GPU in training loop
let images = images.to(device: .cuda(0))
let labels = labels.to(device: .cuda(0))
```

On macOS, use Metal Performance Shaders:

```swift
let model = model.to(device: .mps)
```

### Saving and Loading Models

Save your trained model's weights:

```swift
// Save model parameters
let parameters = model.parameters
try parameters.save(to: "model.pt")

// Load parameters
let loaded = try Parameters.load(from: "model.pt")
model.parameters = loaded
```

### Custom Training Loops

For more control over training:

```swift
for epoch in 0..<numEpochs {
    // Training phase
    withLearningPhase(.training) {
        for batch in trainLoader {
            // ... training code
        }
    }

    // Validation phase
    withLearningPhase(.inference) {
        for batch in validLoader {
            // ... validation code (no gradient computation)
        }
    }
}
```

## Troubleshooting

### Shape Mismatches

Use `print(tensor.shape)` to debug shape issues:

```swift
let x = Tensor.randn([32, 784])
print("Input shape: \(x.shape)")

let output = model(x)
print("Output shape: \(output.shape)")
```

### Memory Issues

For large models or datasets:

```swift
// Process in smaller batches
let dataLoader = DataLoader(dataset: dataset, batchSize: 16)

// Clear GPU memory periodically
Tensor.clearCaches()
```

### Gradient Issues

Check that layers are marked `@differentiable`:

```swift
@differentiable  // Required for backpropagation!
func callAsFunction(_ input: Tensor) -> Tensor {
    return weight * input + bias
}
```

## See Also

- ``Tensor`` - Core tensor operations
- ``Layer`` - Layer protocol documentation
- ``Sequential`` - Model composition
- ``Optimizer`` - Optimization algorithms
