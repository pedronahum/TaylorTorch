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

#if TENSORFLOW_USE_STANDARD_TOOLCHAIN
  import Numerics
#endif

/// Stochastic Gradient Descent (SGD) optimizer with momentum and Nesterov acceleration.
///
/// SGD is the workhorse optimizer for training deep learning models, especially CNNs.
/// It updates parameters by moving in the direction opposite to the gradient, with optional
/// momentum to accelerate convergence and dampen oscillations.
///
/// ## Basic Usage
///
/// ```swift
/// // Create model
/// var model = Sequential {
///     Conv2D(inChannels: 3, outChannels: 64, kernelSize: (3, 3))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))
///     Flatten()
///     Dense(inputSize: 64 * 14 * 14, outputSize: 10)
/// }
///
/// // Initialize SGD optimizer
/// var optimizer = SGD(
///     for: model,
///     learningRate: 0.01,
///     momentum: 0.9
/// )
///
/// // Training step
/// let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///     let predictions = model(images)
///     return softmaxCrossEntropy(logits: predictions, labels: labels)
/// }
/// optimizer.update(&model, along: gradients)
/// ```
///
/// ## Algorithm
///
/// SGD updates parameters using the following rules:
///
/// ### Standard SGD (momentum = 0)
/// ```
/// θ_t+1 = θ_t - α * ∇L(θ_t)
/// ```
///
/// ### SGD with Momentum
/// ```
/// v_t+1 = μ * v_t - α * ∇L(θ_t)
/// θ_t+1 = θ_t + v_t+1
/// ```
///
/// ### SGD with Nesterov Momentum
/// ```
/// v_t+1 = μ * v_t - α * ∇L(θ_t)
/// θ_t+1 = θ_t + μ * v_t+1 - α * ∇L(θ_t)
/// ```
///
/// Where:
/// - `θ` = model parameters
/// - `α` = learning rate
/// - `μ` = momentum coefficient
/// - `v` = velocity (accumulated gradient)
/// - `∇L` = gradient of loss
///
/// ## Momentum
///
/// Momentum accelerates SGD by accumulating a velocity vector in directions of persistent
/// gradient reduction. This helps overcome local minima and speeds up convergence.
///
/// ```swift
/// // Without momentum - may oscillate
/// var sgd = SGD(for: model, learningRate: 0.01)
///
/// // With momentum - smoother convergence
/// var sgdMomentum = SGD(
///     for: model,
///     learningRate: 0.01,
///     momentum: 0.9  // Typical value: 0.9 or 0.95
/// )
/// ```
///
/// **Momentum benefits:**
/// - Faster convergence (fewer training steps)
/// - Less oscillation in parameter space
/// - Better handling of noisy gradients
/// - Can escape shallow local minima
///
/// ## Nesterov Momentum
///
/// Nesterov Accelerated Gradient (NAG) is a smarter version of momentum that "looks ahead"
/// before computing the gradient. It often converges faster than standard momentum.
///
/// ```swift
/// var optimizer = SGD(
///     for: model,
///     learningRate: 0.1,
///     momentum: 0.9,
///     nesterov: true  // Enable Nesterov acceleration
/// )
/// ```
///
/// **When to use Nesterov:**
/// - Training CNNs (ResNet, VGG)
/// - When you want the fastest convergence with SGD
/// - Large batch training (batch size ≥ 128)
///
/// ## Learning Rate Decay
///
/// Gradually reducing the learning rate helps achieve better final performance:
///
/// ```swift
/// // Time-based decay: lr_t = lr / (1 + decay * t)
/// var optimizer = SGD(
///     for: model,
///     learningRate: 0.1,
///     momentum: 0.9,
///     decay: 1e-6
/// )
///
/// // After 100k steps:
/// // effective_lr = 0.1 / (1 + 1e-6 * 100000) = 0.0909
/// ```
///
/// You can also manually adjust learning rate:
///
/// ```swift
/// // Step decay: reduce by 10x every 30 epochs
/// for epoch in 1...100 {
///     if epoch % 30 == 0 {
///         optimizer.learningRate *= 0.1
///     }
///     // ... training code ...
/// }
/// ```
///
/// ## Complete ResNet Training Example
///
/// ```swift
/// // 1. Define ResNet model
/// struct ResNet18: Layer {
///     var conv1: Conv2D
///     var bn1: BatchNorm
///     var layer1: Sequential<...>
///     var layer2: Sequential<...>
///     var avgPool: AvgPool2D
///     var fc: Dense
///
///     init(numClasses: Int = 1000) {
///         conv1 = Conv2D(inChannels: 3, outChannels: 64, kernelSize: (7, 7), stride: (2, 2))
///         bn1 = BatchNorm(numFeatures: 64)
///         // ... initialize all layers ...
///         fc = Dense(inputSize: 512, outputSize: numClasses)
///         avgPool = AvgPool2D(kernelSize: (7, 7))
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         var x = conv1(input)
///         x = bn1(x).relu()
///         x = x.maxPool2d(kernelSize: (3, 3), stride: (2, 2), padding: (1, 1))
///         x = layer1(x)
///         x = layer2(x)
///         x = avgPool(x)
///         x = x.flatten(startDim: 1)
///         return fc(x)
///     }
/// }
///
/// // 2. Initialize with proper hyperparameters for ImageNet training
/// var model = ResNet18(numClasses: 1000)
/// var optimizer = SGD(
///     for: model,
///     learningRate: 0.1,      // Start high for large batch
///     momentum: 0.9,          // Standard for CNNs
///     decay: 1e-4,            // Weight decay for regularization
///     nesterov: true          // Better convergence
/// )
///
/// // 3. Training loop with learning rate schedule
/// let epochs = 90
/// let batchSize = 256
///
/// for epoch in 1...epochs {
///     // Learning rate schedule: reduce at epochs 30, 60, 80
///     if epoch == 30 || epoch == 60 || epoch == 80 {
///         optimizer.learningRate *= 0.1
///     }
///
///     for (images, labels) in trainLoader.batched(batchSize) {
///         // Forward + backward
///         let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///             let logits = model(images)
///             return softmaxCrossEntropy(logits: logits, labels: labels)
///         }
///
///         // Update parameters
///         optimizer.update(&model, along: gradients)
///     }
///
///     print("Epoch \(epoch)/\(epochs), LR: \(optimizer.learningRate)")
/// }
/// ```
///
/// ## Hyperparameter Guidelines
///
/// | Model Type | Learning Rate | Momentum | Nesterov | Batch Size |
/// |------------|---------------|----------|----------|------------|
/// | Small CNN (MNIST) | 0.01-0.1 | 0.9 | true | 32-128 |
/// | ResNet (ImageNet) | 0.1 | 0.9 | true | 256 |
/// | VGG | 0.01 | 0.9 | true | 128-256 |
/// | Fine-tuning | 0.001-0.01 | 0.9 | false | 16-64 |
///
/// **General rules:**
/// - Larger batch size → higher learning rate
/// - Deeper networks → lower learning rate
/// - Always use momentum (0.9 is standard)
/// - Enable Nesterov for CNNs
///
/// ## SGD vs Adam
///
/// | Aspect | SGD | Adam |
/// |--------|-----|------|
/// | Convergence Speed | Slower initially | Faster initially |
/// | Final Performance | Often better | May be worse |
/// | Hyperparameter Sensitivity | High (LR crucial) | Low (robust defaults) |
/// | Memory | Low | High (stores momentum and variance) |
/// | Best For | CNNs, large batch | Transformers, small batch |
///
/// **Use SGD when:**
/// - Training CNNs (ResNet, VGG, EfficientNet)
/// - You have large batches (≥128)
/// - You want the best possible final accuracy
/// - Fine-tuning pretrained models
///
/// ## Tips for Success
///
/// 1. **Learning rate is crucial**: Too high causes divergence, too low is slow
/// 2. **Use learning rate warmup**: Start low (0.01), increase to target over 5-10 epochs
/// 3. **Always use momentum**: 0.9 is a good default
/// 4. **Batch size matters**: Larger batches → higher learning rate
/// 5. **Learning rate schedule**: Reduce by 10x when validation loss plateaus
///
/// ## References
///
/// - ["A Stochastic Approximation Method"](
/// https://projecteuclid.org/euclid.aoms/1177729586) (Robbins and Monro, 1951)
/// - ["Some methods of speeding up the convergence of iteration method"](
/// https://vsokolov.org/courses/750/2018/files/polyak64.pdf) (Polyak, 1964)
/// - ["A method for unconstrained convex minimization problem with the rate of
/// convergence"](http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf)
/// (Nesterov, 1983)
/// - ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385)
/// (He et al., 2015) - Uses SGD with momentum
///
/// ## See Also
///
/// - ``Optimizer`` - Base protocol for all optimizers
/// - ``Adam`` - Adaptive learning rate optimizer
public class SGD<Model: Differentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model
  /// The learning rate.
  public var learningRate: Float
  /// The momentum factor. It accelerates stochastic gradient descent in the relevant direction and
  /// dampens oscillations.
  public var momentum: Float
  /// The learning rate decay.
  public var decay: Float
  /// Use Nesterov momentum if true.
  public var nesterov: Bool
  /// The velocity state of the model.
  public var velocity: Model.TangentVector = .zero
  /// The set of steps taken.
  public var step: Int = 0

  /// Creates an instance for `model`.
  ///
  /// - Parameters:
  ///   - learningRate: The learning rate. The default value is `0.01`.
  ///   - momentum: The momentum factor that accelerates stochastic gradient descent in the relevant
  ///     direction and dampens oscillations. The default value is `0`.
  ///   - decay: The learning rate decay. The default value is `0`.
  ///   - nesterov: Use Nesterov momentum iff `true`. The default value is `true`.
  public init(
    for model: __shared Model,
    learningRate: Float = 0.01,
    momentum: Float = 0,
    decay: Float = 0,
    nesterov: Bool = false
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(momentum >= 0, "Momentum must be non-negative")
    precondition(decay >= 0, "Learning rate decay must be non-negative")

    self.learningRate = learningRate
    self.momentum = momentum
    self.decay = decay
    self.nesterov = nesterov
  }

  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
    velocity = velocity.scaled(by: momentum) - direction.scaled(by: learningRate)
    if nesterov {
      model.move(by: velocity.scaled(by: momentum) - direction.scaled(by: learningRate))
    } else {
      model.move(by: velocity)
    }
  }

  public required init(copying other: SGD, to device: Device) {
    learningRate = other.learningRate
    momentum = other.momentum
    decay = other.decay
    nesterov = other.nesterov
    var newVelocity = other.velocity
    for kp in newVelocity.recursivelyAllWritableKeyPaths(to: Tensor.self) {
      newVelocity[keyPath: kp] = newVelocity[keyPath: kp].to(device: device)
    }
    for kp in newVelocity.recursivelyAllWritableKeyPaths(to: Tensor?.self) {
      if let tensor = newVelocity[keyPath: kp] {
        newVelocity[keyPath: kp] = tensor.to(device: device)
      }
    }
    for kp in newVelocity.recursivelyAllWritableKeyPaths(to: [Tensor].self) {
      newVelocity[keyPath: kp] = newVelocity[keyPath: kp].map { $0.to(device: device) }
    }
    velocity = newVelocity
    step = other.step
  }
}
