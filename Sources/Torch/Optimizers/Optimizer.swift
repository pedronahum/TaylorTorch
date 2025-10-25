// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import _Differentiation

/// A numerical optimizer for training neural networks.
///
/// Optimizers apply an optimization algorithm to iteratively update model parameters
/// by following the negative gradient direction. All optimizers in TaylorTorch conform
/// to this protocol and support automatic differentiation through Swift's
/// `@differentiable` attribute.
///
/// ## Overview
///
/// Training a neural network involves:
/// 1. Forward pass: compute predictions
/// 2. Loss calculation: measure prediction error
/// 3. Backward pass: compute gradients via automatic differentiation
/// 4. Parameter update: optimizer adjusts weights to reduce loss
///
/// The `Optimizer` protocol defines the interface for step 4, with concrete implementations
/// like ``SGD`` and ``Adam`` providing different update strategies.
///
/// ## Basic Training Loop
///
/// ```swift
/// // Define model and optimizer
/// var model = Sequential {
///     Dense(inputSize: 784, outputSize: 128)
///     ReLU()
///     Dense(inputSize: 128, outputSize: 10)
/// }
/// var optimizer = SGD(for: model, learningRate: 0.01, momentum: 0.9)
///
/// // Training loop
/// for epoch in 1...10 {
///     for (x, y) in trainLoader {
///         // Forward pass
///         let predictions = model(x)
///         let loss = softmaxCrossEntropy(logits: predictions, labels: y)
///
///         // Backward pass - compute gradients
///         let (_, gradients) = valueWithGradient(at: model) { model -> Tensor in
///             let pred = model(x)
///             return softmaxCrossEntropy(logits: pred, labels: y)
///         }
///
///         // Update parameters
///         optimizer.update(&model, along: gradients)
///     }
///     print("Epoch \(epoch) complete")
/// }
/// ```
///
/// ## Available Optimizers
///
/// TaylorTorch provides two main optimizer implementations:
///
/// | Optimizer | Best For | Key Features |
/// |-----------|----------|--------------|
/// | ``SGD`` | CNNs, when batch size is large | Momentum, Nesterov, simple and stable |
/// | ``Adam`` | Transformers, RNNs, small batches | Adaptive learning rates, AdamW variant |
///
/// ### Choosing an Optimizer
///
/// **Use SGD when:**
/// - Training CNNs (ResNet, VGG, etc.)
/// - Large batch sizes (≥128)
/// - You want stable, predictable convergence
/// - Fine-tuning pretrained models
///
/// **Use Adam when:**
/// - Training Transformers (BERT, GPT)
/// - Training RNNs/LSTMs
/// - Small batch sizes (<32)
/// - Sparse gradients
/// - You need fast initial convergence
///
/// ## Complete MNIST Example
///
/// ```swift
/// import Torch
///
/// // 1. Define model
/// struct MNISTClassifier: Layer {
///     var conv1: Conv2D
///     var conv2: Conv2D
///     var flatten: Flatten
///     var fc1: Dense
///     var fc2: Dense
///
///     init() {
///         conv1 = Conv2D(inChannels: 1, outChannels: 32, kernelSize: (3, 3))
///         conv2 = Conv2D(inChannels: 32, outChannels: 64, kernelSize: (3, 3))
///         flatten = Flatten()
///         fc1 = Dense(inputSize: 9216, outputSize: 128)  // 64 * 12 * 12
///         fc2 = Dense(inputSize: 128, outputSize: 10)
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         var x = conv1(input).relu()  // [batch, 32, 26, 26]
///         x = conv2(x).relu()          // [batch, 64, 24, 24]
///         x = x.maxPool2d(kernelSize: (2, 2))  // [batch, 64, 12, 12]
///         x = flatten(x)               // [batch, 9216]
///         x = fc1(x).relu()            // [batch, 128]
///         return fc2(x)                // [batch, 10]
///     }
/// }
///
/// // 2. Initialize model and optimizer
/// var model = MNISTClassifier()
/// var optimizer = SGD(
///     for: model,
///     learningRate: 0.1,
///     momentum: 0.9,
///     nesterov: true
/// )
///
/// // 3. Training loop
/// let epochs = 10
/// let batchSize = 128
///
/// for epoch in 1...epochs {
///     var totalLoss: Float = 0
///     var correct: Int = 0
///     var total: Int = 0
///
///     for (images, labels) in trainLoader.batched(batchSize) {
///         // Forward + backward + optimize
///         let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///             let logits = model(images)
///             return softmaxCrossEntropy(logits: logits, labels: labels)
///         }
///
///         optimizer.update(&model, along: gradients)
///
///         // Track metrics
///         totalLoss += loss.item()
///         let predictions = model(images).argmax(dim: -1)
///         correct += (predictions == labels).sum().item()
///         total += batchSize
///     }
///
///     let accuracy = Float(correct) / Float(total) * 100
///     print("Epoch \(epoch): Loss = \(totalLoss), Accuracy = \(accuracy)%")
/// }
/// ```
///
/// ## Learning Rate Scheduling
///
/// Many optimizers support learning rate decay for better convergence:
///
/// ```swift
/// // SGD with step decay
/// var optimizer = SGD(
///     for: model,
///     learningRate: 0.1,
///     momentum: 0.9,
///     decay: 1e-6  // lr_t = lr / (1 + decay * t)
/// )
///
/// // Manual learning rate scheduling
/// for epoch in 1...100 {
///     // Reduce learning rate by 10x every 30 epochs
///     if epoch % 30 == 0 {
///         optimizer.learningRate *= 0.1
///     }
///
///     // ... training code ...
/// }
/// ```
///
/// ## Gradient Clipping
///
/// Prevent exploding gradients by clipping before the optimizer step:
///
/// ```swift
/// let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///     let predictions = model(input)
///     return loss(predictions, target)
/// }
///
/// // Clip gradients by norm
/// let clippedGradients = clipGradientsByNorm(gradients, maxNorm: 1.0)
/// optimizer.update(&model, along: clippedGradients)
/// ```
///
/// ## Multi-Device Training
///
/// Optimizers can be copied to different devices:
///
/// ```swift
/// let model = MyModel()
/// let optimizer = Adam(for: model, learningRate: 1e-3)
///
/// // Move to GPU
/// let gpuOptimizer = optimizer.copy(to: .gpu(0))
/// ```
///
/// ## See Also
///
/// - ``SGD`` - Stochastic Gradient Descent with momentum
/// - ``Adam`` - Adaptive Moment Estimation optimizer
public protocol Optimizer: CopyableToDevice {
  /// The type of the model to optimize.
  associatedtype Model: Differentiable
  /// The scalar parameter type.
  associatedtype Scalar: FloatingPoint
  /// The learning rate.
  var learningRate: Scalar { get set }
  /// Updates the given model along the given direction.
  mutating func update(_ model: inout Model, along direction: Model.TangentVector)
}
