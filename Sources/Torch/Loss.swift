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

// This file has been adapted from the original at
// https://github.com/tensorflow/swift-apis/blob/main/Sources/TensorFlow/Loss.swift
// to reflect the ATen-based implementation of Tensor in Swift.

import _Differentiation

// MARK: - Basic Loss Functions

/// Computes the L1 loss (Mean Absolute Error) between predicted and expected values.
///
/// L1 loss measures the mean absolute difference between predictions and targets.
/// It's more robust to outliers than L2 loss because it penalizes errors linearly
/// rather than quadratically.
///
/// ## Formula
///
/// ```
/// L1(y, ŷ) = reduction(|y - ŷ|)
/// ```
///
/// Where:
/// - `y` = expected (ground truth)
/// - `ŷ` = predicted values
/// - `reduction` = aggregation function (sum, mean, etc.)
///
/// ## Basic Usage
///
/// ```swift
/// let predictions = model(input)  // [batch, features]
/// let targets = labels            // [batch, features]
///
/// // Mean absolute error (default)
/// let loss = l1Loss(predicted: predictions, expected: targets, reduction: mean)
///
/// // Sum of absolute errors
/// let totalError = l1Loss(predicted: predictions, expected: targets, reduction: sum)
/// ```
///
/// ## When to Use L1 Loss
///
/// **Use L1 loss when:**
/// - You have outliers in your data (L1 is less sensitive)
/// - You want sparse predictions (L1 encourages sparsity)
/// - Doing regression on noisy data
/// - You care about median accuracy more than mean accuracy
///
/// **Don't use L1 loss when:**
/// - You need smooth gradients (L1 has non-smooth gradient at 0)
/// - Small errors are as important as large errors (use L2 instead)
///
/// ## Complete Regression Example
///
/// ```swift
/// // Simple regression model
/// var model = Sequential {
///     Dense(inputSize: 10, outputSize: 64)
///     ReLU()
///     Dense(inputSize: 64, outputSize: 1)
/// }
///
/// var optimizer = Adam(for: model, learningRate: 1e-3)
///
/// // Training loop
/// for epoch in 1...100 {
///     for (x, y) in trainLoader {
///         let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///             let predictions = model(x)
///             return l1Loss(predicted: predictions, expected: y, reduction: mean)
///         }
///
///         optimizer.update(&model, along: gradients)
///     }
/// }
/// ```
///
/// ## L1 vs L2 Loss
///
/// | Aspect | L1 Loss | L2 Loss |
/// |--------|---------|---------|
/// | Formula | \|y - ŷ\| | (y - ŷ)² |
/// | Gradient | Constant magnitude | Proportional to error |
/// | Outlier Sensitivity | Low (robust) | High (sensitive) |
/// | Sparsity | Encourages | Doesn't encourage |
/// | Smoothness | Non-smooth at 0 | Smooth everywhere |
/// | Best for | Noisy data, outliers | Clean data, no outliers |
///
/// ## Parameters
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network
///   - expected: Expected values (ground truth targets)
///   - reduction: Aggregation function to apply (default: `sum`)
///
/// ## See Also
///
/// - ``l2Loss(predicted:expected:reduction:)`` - L2 loss (mean squared error)
/// - ``meanAbsoluteError(predicted:expected:)`` - Convenience function for mean L1 loss
/// - ``huberLoss(predicted:expected:delta:reduction:)`` - Hybrid of L1 and L2
@differentiable(reverse)
public func l1Loss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = sum
) -> Tensor {
  reduction(abs(expected - predicted))
}

/// Computes the L2 loss (Mean Squared Error) between predicted and expected values.
///
/// L2 loss measures the mean squared difference between predictions and targets.
/// It heavily penalizes large errors (quadratically), making it sensitive to outliers
/// but providing smooth gradients for optimization.
///
/// ## Formula
///
/// ```
/// L2(y, ŷ) = reduction((y - ŷ)²)
/// ```
///
/// Where:
/// - `y` = expected (ground truth)
/// - `ŷ` = predicted values
/// - `reduction` = aggregation function (sum, mean, etc.)
///
/// ## Basic Usage
///
/// ```swift
/// let predictions = model(input)  // [batch, features]
/// let targets = labels            // [batch, features]
///
/// // Mean squared error
/// let loss = l2Loss(predicted: predictions, expected: targets, reduction: mean)
///
/// // Sum of squared errors
/// let totalError = l2Loss(predicted: predictions, expected: targets, reduction: sum)
/// ```
///
/// ## When to Use L2 Loss
///
/// **Use L2 loss when:**
/// - Your data has minimal outliers
/// - You want large errors to be heavily penalized
/// - You need smooth gradients everywhere
/// - Doing standard regression tasks
/// - Training neural networks for continuous outputs
///
/// **Don't use L2 loss when:**
/// - Your data has many outliers (use L1 or Huber instead)
/// - Extreme errors would dominate training
/// - You want robust prediction to noise
///
/// ## Complete Regression Example
///
/// ```swift
/// // Neural network for house price prediction
/// struct HousePriceModel: Layer {
///     var fc1: Dense
///     var fc2: Dense
///     var fc3: Dense
///
///     init(numFeatures: Int) {
///         fc1 = Dense(inputSize: numFeatures, outputSize: 128)
///         fc2 = Dense(inputSize: 128, outputSize: 64)
///         fc3 = Dense(inputSize: 64, outputSize: 1)
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         var x = fc1(input).relu()
///         x = fc2(x).relu()
///         return fc3(x)  // No activation for regression
///     }
/// }
///
/// var model = HousePriceModel(numFeatures: 10)
/// var optimizer = Adam(for: model, learningRate: 1e-3)
///
/// // Training loop
/// for epoch in 1...200 {
///     for (features, prices) in trainLoader {
///         let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///             let predictions = model(features)
///             // Use mean squared error for regression
///             return l2Loss(predicted: predictions, expected: prices, reduction: mean)
///         }
///
///         optimizer.update(&model, along: gradients)
///     }
///
///     // Validation
///     let valPredictions = model(valFeatures)
///     let valLoss = l2Loss(predicted: valPredictions, expected: valPrices, reduction: mean)
///     let rmse = sqrt(valLoss)  // Root Mean Squared Error
///     print("Epoch \(epoch): RMSE = \(rmse.item())")
/// }
/// ```
///
/// ## L2 vs L1 Loss
///
/// | Aspect | L2 Loss | L1 Loss |
/// |--------|---------|---------|
/// | Formula | (y - ŷ)² | \|y - ŷ\| |
/// | Gradient | 2(y - ŷ) | sign(y - ŷ) |
/// | Outlier Sensitivity | High (sensitive) | Low (robust) |
/// | Optimization | Smooth, easy | Non-smooth at 0 |
/// | Error Penalty | Quadratic (severe for large errors) | Linear |
/// | Best for | Clean data, standard regression | Noisy data, outliers |
/// | Common Use | Default for regression | Robust regression |
///
/// ## Metrics Derived from L2 Loss
///
/// ```swift
/// // Mean Squared Error (MSE)
/// let mse = l2Loss(predicted: predictions, expected: targets, reduction: mean)
///
/// // Root Mean Squared Error (RMSE)
/// let rmse = sqrt(mse)
///
/// // Sum of Squared Errors (SSE)
/// let sse = l2Loss(predicted: predictions, expected: targets, reduction: sum)
/// ```
///
/// ## Image Reconstruction Example
///
/// L2 loss is commonly used for autoencoders and image reconstruction:
///
/// ```swift
/// struct Autoencoder: Layer {
///     var encoder: Sequential<...>
///     var decoder: Sequential<...>
///
///     init() {
///         encoder = Sequential {
///             Conv2D(inChannels: 3, outChannels: 32, kernelSize: (3, 3))
///             ReLU()
///             MaxPool2D(kernelSize: (2, 2))
///             // ... more layers ...
///         }
///         decoder = Sequential {
///             // ... decoder layers ...
///         }
///     }
///
///     @differentiable
///     func callAsFunction(_ images: Tensor) -> Tensor {
///         let encoded = encoder(images)
///         return decoder(encoded)
///     }
/// }
///
/// var model = Autoencoder()
/// var optimizer = Adam(for: model, learningRate: 1e-3)
///
/// // Training: minimize reconstruction error
/// let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///     let reconstructed = model(images)
///     // Pixel-wise L2 loss
///     return l2Loss(predicted: reconstructed, expected: images, reduction: mean)
/// }
/// ```
///
/// ## Parameters
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network
///   - expected: Expected values (ground truth targets)
///   - reduction: Aggregation function to apply (default: `sum`)
///
/// ## See Also
///
/// - ``l1Loss(predicted:expected:reduction:)`` - L1 loss (mean absolute error)
/// - ``meanSquaredError(predicted:expected:)`` - Convenience function for mean L2 loss
/// - ``huberLoss(predicted:expected:delta:reduction:)`` - Hybrid of L1 and L2 for robustness
@differentiable(reverse)
public func l2Loss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = sum
) -> Tensor {
  let difference = expected - predicted
  return reduction(difference * difference)
}

/// Computes the mean of absolute difference between labels and predictions.
/// `loss = mean(abs(expected - predicted))`
@differentiable(reverse)
public func meanAbsoluteError(
  predicted: Tensor,
  expected: Tensor
) -> Tensor {
  l1Loss(predicted: predicted, expected: expected, reduction: mean)
}

/// Computes the mean of squares of errors between labels and predictions.
/// `loss = mean(square(expected - predicted))`
@differentiable(reverse)
public func meanSquaredError(
  predicted: Tensor,
  expected: Tensor
) -> Tensor {
  l2Loss(predicted: predicted, expected: expected, reduction: mean)
}

/// Computes the mean squared logarithmic error between `predicted` and `expected`.
/// `loss = square(log(expected + 1) - log(predicted + 1))`
///
/// - Note: Negative tensor entries will be clamped at `0` to avoid undefined logarithmic behavior.
@differentiable(reverse)
public func meanSquaredLogarithmicError(
  predicted: Tensor,
  expected: Tensor
) -> Tensor {
  let logPredicted = log(maximum(predicted, Tensor(0)) + Tensor(1))
  let logExpected = log(maximum(expected, Tensor(0)) + Tensor(1))
  return l2Loss(predicted: logPredicted, expected: logExpected, reduction: mean)
}

/// Computes the mean absolute percentage error between `predicted` and `expected`.
/// `loss = 100 * mean(abs((expected - predicted) / abs(expected)))`
@differentiable(reverse)
public func meanAbsolutePercentageError(
  predicted: Tensor,
  expected: Tensor
) -> Tensor {
  Tensor(100) * abs((expected - predicted) / abs(expected)).mean()
}

// MARK: - Hinge Losses

/// Computes the hinge loss between `predicted` and `expected`.
/// `loss = reduction(max(0, 1 - predicted * expected))`
/// `expected` values are expected to be -1 or 1.
@differentiable(reverse)
public func hingeLoss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  reduction(relu(Tensor(1) - expected * predicted))
}

/// Computes the squared hinge loss between `predicted` and `expected`.
/// `loss = reduction(square(max(0, 1 - predicted * expected)))`
/// `expected` values are expected to be -1 or 1.
@differentiable(reverse)
public func squaredHingeLoss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  let loss = hingeLoss(predicted: predicted, expected: expected, reduction: { $0 })
  let squared = loss * loss
  return reduction(squared)
}

/// Computes the categorical hinge loss between `predicted` and `expected`.
/// `loss = maximum(negative - positive + 1, 0)`
/// where `negative = max((1 - expected) * predicted)` and `positive = sum(predicted * expected)`
@differentiable(reverse)
public func categoricalHingeLoss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  let positive = (expected * predicted).sum(dim: -1)

  // ✅ WORKAROUND: Use .topk(1, ...) instead of the buggy .max(...)
  // This is mathematically equivalent to finding the maximum value.
  let negativeScores = (1 - expected) * predicted
  let negative = negativeScores.topk(1, dim: -1).values.squeezed(dim: Int(-1))

  // The rest of the function remains the same.
  let margin = negative - positive + 1
  let loss = maximum(margin, Tensor(0))

  return reduction(loss)
}

// MARK: - Advanced Loss Functions

/// Differentiable `softplus` utility function.
@differentiable(reverse)
public func softplus(_ x: Tensor) -> Tensor {
  return log(Tensor(1) + exp(x))
}

/// Computes the logarithm of the hyperbolic cosine of the prediction error.
/// `logcosh = log((exp(x) + exp(-x))/2)`, where x is the error `predicted - expected`.
@differentiable(reverse)
public func logCoshLoss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  let x = predicted - expected
  // A numerically stable implementation: x + softplus(-2x) - log(2)
  return reduction(x + softplus(Tensor(-2) * x) - log(Tensor(2)))
}

/// Computes the Poisson loss between predicted and expected.
/// The Poisson loss is `predicted - expected * log(predicted)`.
@differentiable(reverse)
public func poissonLoss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  reduction(predicted - expected * log(predicted))
}

/// Computes Kullback-Leibler divergence loss between `expected` and `predicted`.
/// `loss = reduction(expected * log(expected / predicted))`
@differentiable(reverse)
public func kullbackLeiblerDivergence(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = sum
) -> Tensor {
  reduction(expected * log(expected / predicted))
}

/*
/// Computes the Huber loss between `predicted` and `expected`.
/// For each value `x` in `error = expected - predicted`:
/// - `0.5 * x^2` if `|x| <= δ`.
/// - `0.5 * δ^2 + δ * (|x| - δ)` otherwise.
@differentiable(reverse)
public func huberLoss(
  predicted: Tensor,
  expected: Tensor,
  delta: Float,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = sum
) -> Tensor {
  let error = expected - predicted
  let absError = abs(error)
  let deltaTensor = withoutDerivative(at: Tensor(delta))
  let quadratic = relu(deltaTensor - absError)  // |x| <= delta part
  let linear = absError - quadratic  // |x| > delta part
  let loss = (Tensor(0.5) * quadratic * quadratic) + (deltaTensor * linear)
  return reduction(loss)
}*/

/// Computes the Huber loss between `predicted` and `expected`.
/// For each value `x` in `error = expected - predicted`:
/// - `0.5 * x^2` if `|x| <= δ`.
/// - `δ * (|x| - 0.5 * δ)` otherwise.
@differentiable(reverse)
public func huberLoss(
  predicted: Tensor,
  expected: Tensor,
  delta: Float,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = sum
) -> Tensor {
  // 1. Calculate the error. `deltaTensor` is already marked as a constant.
  let error = expected - predicted
  let absError = abs(error)
  let deltaTensor = withoutDerivative(at: Tensor(delta, device: predicted.device))

  // 2. Create the boolean mask, stopping the gradient from flowing through the non-differentiable comparison.
  let quadraticMask = withoutDerivative(at: absError .<= deltaTensor)

  // 3. Calculate the loss for both cases across the entire tensor.
  let quadraticLoss = 0.5 * error * error
  let linearLoss = deltaTensor * (absError - 0.5 * deltaTensor)

  // 4. Convert the mask to float, also marking this non-differentiable conversion as a constant.
  let maskAsFloat = withoutDerivative(at: quadraticMask.to(dtype: predicted.dtype ?? .float32))

  // Use the non-differentiable mask to select between the two differentiable loss terms.
  let linearMaskAsFloat = 1.0 - maskAsFloat
  let loss = (quadraticLoss * maskAsFloat) + (linearLoss * linearMaskAsFloat)

  // 5. Apply the final reduction (e.g., sum).
  return reduction(loss)
}

// MARK: - Cross Entropy Losses

/// Computes the sigmoid cross entropy (binary cross entropy) between logits and labels.
/// Use this loss when there are only two label classes (assumed to be 0 and 1).
@differentiable(reverse)
public func sigmoidCrossEntropy(
  logits: Tensor,
  labels: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  // Numerically stable implementation: max(logits, 0) - logits * labels + log(1 + exp(-abs(logits)))
  let maxLogitsWithZero = maximum(logits, Tensor(0))
  let negAbsLogits = abs(logits).negated()
  let logExp = log(Tensor(1) + exp(negAbsLogits))
  return reduction(maxLogitsWithZero - logits * labels + logExp)
}

/// Computes the softmax cross-entropy loss between logits and labels for multi-class classification.
///
/// Softmax cross-entropy is the standard loss function for multi-class classification problems.
/// It combines softmax activation with cross-entropy loss in a numerically stable way,
/// and is used for training all modern classification networks (ResNet, BERT, GPT, etc.).
///
/// ## Formula
///
/// ```
/// L(y, ŷ) = -log(exp(ŷ_y) / Σ_j exp(ŷ_j))
///         = -ŷ_y + log(Σ_j exp(ŷ_j))
/// ```
///
/// Where:
/// - `y` = true class index
/// - `ŷ` = logits (unscaled predictions) from the model
/// - The formula computes the negative log probability of the correct class
///
/// ## Basic Usage
///
/// ```swift
/// // Image classification model
/// var model = Sequential {
///     Conv2D(inChannels: 3, outChannels: 64, kernelSize: (3, 3))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))
///     Flatten()
///     Dense(inputSize: 64 * 14 * 14, outputSize: 10)  // 10 classes
/// }
///
/// let images = Tensor.randn([32, 3, 28, 28])  // Batch of 32 images
/// let labels = Tensor([0, 1, 2, ...])         // True class indices [32]
///
/// // Forward pass
/// let logits = model(images)  // [32, 10] - raw scores (no softmax!)
///
/// // Compute loss
/// let loss = softmaxCrossEntropy(logits: logits, labels: labels)
/// ```
///
/// **Important**: Do NOT apply softmax to your model's output! This function expects raw logits.
///
/// ## Complete CIFAR-10 Training Example
///
/// ```swift
/// // CNN for CIFAR-10 classification
/// struct CIFAR10Classifier: Layer {
///     var conv1: Conv2D
///     var conv2: Conv2D
///     var conv3: Conv2D
///     var fc1: Dense
///     var fc2: Dense
///
///     init() {
///         conv1 = Conv2D(inChannels: 3, outChannels: 32, kernelSize: (3, 3))
///         conv2 = Conv2D(inChannels: 32, outChannels: 64, kernelSize: (3, 3))
///         conv3 = Conv2D(inChannels: 64, outChannels: 128, kernelSize: (3, 3))
///         fc1 = Dense(inputSize: 128 * 2 * 2, outputSize: 256)
///         fc2 = Dense(inputSize: 256, outputSize: 10)  // 10 classes
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         var x = conv1(input).relu()
///         x = x.maxPool2d(kernelSize: (2, 2))
///         x = conv2(x).relu()
///         x = x.maxPool2d(kernelSize: (2, 2))
///         x = conv3(x).relu()
///         x = x.maxPool2d(kernelSize: (2, 2))
///         x = x.flatten(startDim: 1)
///         x = fc1(x).relu()
///         return fc2(x)  // Return logits (NO softmax)
///     }
/// }
///
/// var model = CIFAR10Classifier()
/// var optimizer = SGD(for: model, learningRate: 0.01, momentum: 0.9)
///
/// // Training loop
/// for epoch in 1...100 {
///     var totalLoss: Float = 0
///     var correct: Int = 0
///     var total: Int = 0
///
///     for (images, labels) in trainLoader {
///         // Forward + backward
///         let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///             let logits = model(images)
///             return softmaxCrossEntropy(logits: logits, labels: labels)
///         }
///
///         optimizer.update(&model, along: gradients)
///
///         // Compute accuracy
///         let predictions = logits.argmax(dim: -1)
///         correct += (predictions == labels).sum().item()
///         total += labels.shape[0]
///         totalLoss += loss.item()
///     }
///
///     let accuracy = Float(correct) / Float(total) * 100
///     print("Epoch \(epoch): Loss = \(totalLoss), Accuracy = \(accuracy)%")
/// }
/// ```
///
/// ## When to Use Softmax Cross-Entropy
///
/// **Use softmax cross-entropy when:**
/// - Multi-class classification (3+ classes)
/// - Each sample belongs to exactly ONE class (mutually exclusive)
/// - Classes are nominal (no ordering: cat, dog, car, etc.)
/// - Training CNNs, Transformers, or any classifier
///
/// **Don't use softmax cross-entropy when:**
/// - Binary classification (use ``sigmoidCrossEntropy`` instead)
/// - Multi-label classification (use ``sigmoidCrossEntropy`` per label)
/// - Regression tasks (use ``l2Loss`` or ``l1Loss``)
///
/// ## Inference: Getting Class Probabilities and Predictions
///
/// ```swift
/// // During training: use logits directly with softmaxCrossEntropy
/// let loss = softmaxCrossEntropy(logits: logits, labels: labels)
///
/// // During inference: get probabilities or predictions
/// let logits = model(images)  // [batch, numClasses]
///
/// // Get class probabilities
/// let probabilities = logits.softmax(dim: -1)  // [batch, numClasses]
/// // probabilities[i] sums to 1.0
///
/// // Get predicted class
/// let predictions = logits.argmax(dim: -1)  // [batch]
/// // predictions[i] in 0...(numClasses-1)
///
/// // Get confidence for top prediction
/// let confidences = probabilities.max(dim: -1).values  // [batch]
/// ```
///
/// ## Transformer Text Classification Example
///
/// ```swift
/// // BERT-style text classifier
/// struct TextClassifier: Layer {
///     var embedding: Embedding
///     var transformer: TransformerEncoderLayer
///     var classifier: Dense
///
///     init(vocabSize: Int, numClasses: Int) {
///         embedding = Embedding(vocabularySize: vocabSize, embeddingSize: 768)
///         transformer = TransformerEncoderLayer(modelDim: 768, numHeads: 12)
///         classifier = Dense(inputSize: 768, outputSize: numClasses)
///     }
///
///     @differentiable
///     func callAsFunction(_ tokens: Tensor) -> Tensor {
///         var x = embedding(tokens)        // [batch, seqLen, 768]
///         x = transformer(x)               // [batch, seqLen, 768]
///         let cls = x[:, 0, :]             // [batch, 768] - Use [CLS] token
///         return classifier(cls)           // [batch, numClasses]
///     }
/// }
///
/// var model = TextClassifier(vocabSize: 30000, numClasses: 2)
/// var optimizer = Adam(for: model, learningRate: 2e-5)
///
/// // Fine-tuning loop
/// for (tokens, labels) in trainLoader {
///     let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
///         let logits = model(tokens)
///         return softmaxCrossEntropy(logits: logits, labels: labels)
///     }
///     optimizer.update(&model, along: gradients)
/// }
/// ```
///
/// ## Common Loss Functions Comparison
///
/// | Loss Function | Use Case | Input Format | Output Type |
/// |---------------|----------|--------------|-------------|
/// | ``softmaxCrossEntropy`` | Multi-class (>2 classes) | Logits + class indices | Classification |
/// | ``sigmoidCrossEntropy`` | Binary or multi-label | Logits + binary labels | Classification |
/// | ``l2Loss`` | Regression | Predictions + continuous targets | Regression |
/// | ``l1Loss`` | Robust regression | Predictions + continuous targets | Regression |
///
/// ## Numerical Stability
///
/// This implementation uses the log-sum-exp trick for numerical stability:
/// ```
/// log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))
/// ```
///
/// This prevents overflow/underflow when exponentiating large logits.
///
/// ## Label Smoothing (Advanced)
///
/// For better generalization, you can implement label smoothing:
///
/// ```swift
/// func labelSmoothingCrossEntropy(
///     logits: Tensor,
///     labels: Tensor,
///     smoothing: Float = 0.1
/// ) -> Tensor {
///     let numClasses = logits.shape.last!
///     let logProbs = logSoftmax(logits, dim: -1)
///
///     // Soft targets: (1 - smoothing) * one_hot + smoothing / numClasses
///     let oneHot = oneHot(indices: labels, depth: numClasses)
///     let smoothedLabels = oneHot * (1.0 - smoothing) + smoothing / Float(numClasses)
///
///     return -(logProbs * smoothedLabels).sum(dim: -1).mean()
/// }
/// ```
///
/// ## Parameters
///
/// - Parameters:
///   - logits: Unscaled log probabilities of shape `[batch, numClasses]`
///   - labels: Integer class indices of shape `[batch]`, values in `0...(numClasses-1)`
///   - reduction: Aggregation function (default: `mean`)
///
/// ## See Also
///
/// - ``sigmoidCrossEntropy(logits:labels:reduction:)`` - Binary cross-entropy
/// - ``Tensor/softmax(dim:)`` - Softmax activation for inference
/// - ``Tensor/argmax(dim:)`` - Get predicted class from logits
@differentiable(reverse)
public func softmaxCrossEntropy(
  logits: Tensor,
  labels: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  // 1. Get the number of classes from the last dimension of the logits.
  let numClasses = logits.shape.last!

  // 2. Compute the log-probabilities in a numerically stable way.
  let logProbs = logSoftmax(logits, dim: -1)

  // 3. Convert the integer labels to a one-hot representation.
  let labelsOneHot = withoutDerivative(
    at: oneHot(
      indices: labels,
      depth: numClasses
    ))

  // 4. Select the correct log-probabilities by multiplying with the one-hot tensor
  //    and sum to get the loss for each example. This is fully differentiable.
  let perExampleLoss = (logProbs * labelsOneHot).sum(dim: -1).negated()

  // 5. Apply the final reduction (e.g., mean) over the batch.
  return reduction(perExampleLoss)
}

/// Differentiable logSoftmax function.
@differentiable(reverse)
private func logSoftmax(_ tensor: Tensor, dim: Int) -> Tensor {
  let maxAlongDim = tensor.max(dim: dim, keepdim: true).values
  let x = tensor - maxAlongDim
  let logSumExp = log(sum(exp(x), dim: dim, keepdim: true))
  return x - logSumExp
}

/// Differentiable Negative Log Likelihood Loss.
@differentiable(reverse)
private func nllLoss(
  predicted: Tensor,  // Should be log-probabilities
  expected: Tensor,  // Should be integer class indices
  reduction: @differentiable(reverse) (Tensor) -> Tensor = mean
) -> Tensor {
  // This is a simplified implementation. A production-ready version
  // would use `gather` or `index_select` for efficiency.
  let numClasses = withoutDerivative(at: predicted.shape.last!)
  let oneHot = oneHot(indices: expected, depth: numClasses)
  let loss = (predicted * oneHot).sum(dim: -1).negated()
  return reduction(loss)
}

/// Creates a one-hot tensor from a tensor of indices.
/// - Parameters:
///   - indices: A tensor of integer indices.
///   - depth: The number of classes (the depth of the one-hot dimension).
///   - axis: The axis to insert the one-hot dimension.
///   - onValue: The value for the 'on' state (typically 1.0).
///   - offValue: The value for the 'off' state (typically 0.0).
/// - Returns: A one-hot encoded tensor.
//@differentiable(reverse)
private func oneHot(
  indices: Tensor,
  depth: Int,
  onValue: Float = 1.0,
  offValue: Float = 0.0
) -> Tensor {
  let device = withoutDerivative(at: indices.device)
  let intIndices = withoutDerivative(at: indices.to(dtype: .int64))
  let rank = withoutDerivative(at: intIndices.rank)

  var depthShape = [Int](repeating: 1, count: rank)
  depthShape.append(depth)

  let classIndices = withoutDerivative(
    at: Tensor.arange(Int64(0), to: Int64(depth), step: Int64(1), dtype: .int64, device: device)
      .reshaped(depthShape))
  let expandedIndices = withoutDerivative(at: intIndices.unsqueezed(dim: rank))
  let mask = expandedIndices.eq(classIndices)

  let onTensor = Tensor(onValue, dtype: .float32, device: device)
  let offTensor = Tensor(offValue, dtype: .float32, device: device)
  return TorchWhere.select(condition: mask, onTensor, offTensor)
}
