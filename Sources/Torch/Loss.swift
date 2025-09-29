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

// MARK: - Reduction Functions

/*
/// A differentiable version of the sum reduction, for use as a default parameter.
@differentiable(reverse)
public func sum(_ tensor: Tensor) -> Tensor {
  return tensor.sum()
}

/// A differentiable version of the mean reduction, for use as a default parameter.
@differentiable(reverse)
public func mean(_ tensor: Tensor) -> Tensor {
  return tensor.mean()
}
*/

// MARK: - Basic Loss Functions

/// Computes the L1 loss between `expected` and `predicted`.
/// `loss = reduction(abs(expected - predicted))`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
///   - reduction: A differentiable function to apply on the computed element-wise loss values.
@differentiable(reverse)
public func l1Loss(
  predicted: Tensor,
  expected: Tensor,
  reduction: @differentiable(reverse) (Tensor) -> Tensor = sum
) -> Tensor {
  reduction(abs(expected - predicted))
}

/// Computes the L2 loss between `expected` and `predicted`.
/// `loss = reduction(square(expected - predicted))`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
///   - reduction: A differentiable function to apply on the computed element-wise loss values.
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

/// Computes the softmax cross entropy between logits and labels.
/// Use this cross-entropy loss function when there are two or more label classes.
///
/// - Parameters:
///   - logits: Unscaled log probabilities from a neural network of shape `[batchSize, numClasses]`.
///   - labels: Integer values (class indices) of shape `[batchSize]`.
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
  // This implementation assumes your library has these standard functions.
  let shape = withoutDerivative(at: indices.shape)
  let rank = withoutDerivative(at: indices.rank)

  // Create the base tensor with the 'off' value.
  var oneHotShape = shape
  oneHotShape.insert(depth, at: rank)
  let base = Tensor(array: [offValue], shape: oneHotShape)

  // Create the update values.
  let updates = Tensor(array: [onValue], shape: shape)

  // Scatter the 'on' values into the correct locations.

  let oneHotIndices = withoutDerivative(at: indices.unsqueezed(dim: rank).to(dtype: .int64))
  return base.scatterAdd(dim: rank, index: oneHotIndices, source: updates)
}
