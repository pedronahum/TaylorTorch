import Foundation
import _Differentiation

/// A regularization layer that randomly zeros elements during training to prevent overfitting.
///
/// `Dropout` is a powerful regularization technique that randomly sets a fraction of input elements
/// to zero during training. This prevents neurons from co-adapting too much and helps the model
/// generalize better to unseen data.
///
/// ## Overview
///
/// During training, Dropout:
/// 1. Randomly zeros out elements with probability `p`
/// 2. Scales remaining elements by `1/(1-p)` (inverted dropout)
/// 3. Forces the network to learn robust features
///
/// During inference (evaluation), Dropout acts as an identity function, passing inputs unchanged.
///
/// ## Creating and Using Dropout
///
/// ```swift
/// // Standard dropout with 50% drop rate
/// let model = Sequential {
///     Linear(inputSize: 784, outputSize: 512)
///     ReLU()
///     Dropout(probability: 0.5)
///     Linear(inputSize: 512, outputSize: 10)
/// }
///
/// // Training mode - dropout active
/// let trainOutput = model(input)
///
/// // Inference mode - dropout disabled
/// let testOutput = model.inferring(from: input)
/// ```
///
/// ## Topics
///
/// ### Creating Dropout
///
/// - ``init(probability:)``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``Layer/inferring(from:)`` - Run layer in inference mode
/// - ``Linear`` - Fully connected layers often used with dropout
/// - ``BatchNorm`` - Alternative regularization technique
public struct Dropout: ParameterlessLayer {
  /// The probability of dropping each element during training.
  ///
  /// Must be in the range [0, 1). Common values are 0.5 for hidden layers and 0.1-0.2 for
  /// convolutional layers.
  @noDerivative public let probability: Float

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  /// Creates a dropout layer with the specified drop probability.
  ///
  /// - Parameter probability: The probability of dropping each element during training.
  ///                         Must be in the range [0, 1). Defaults to 0.5 (50% dropout).
  ///
  /// ```swift
  /// // Standard dropout (50%)
  /// let dropout = Dropout(probability: 0.5)
  ///
  /// // Light regularization (20%)
  /// let lightDropout = Dropout(probability: 0.2)
  ///
  /// // No dropout (identity)
  /// let noDropout = Dropout(probability: 0.0)
  /// ```
  ///
  /// - Precondition: `probability` must be in [0, 1).
  public init(probability p: Float = 0.5) {
    precondition(p >= 0 && p < 1, "Dropout probability must be in [0, 1).")
    self.probability = p
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Identity at inference or p == 0
    let phase = withoutDerivative(at: Context.local.learningPhase)
    let p = withoutDerivative(at: probability)
    if phase == .inference || p == 0 { return x }

    // Generate a Bernoulli(keepProb) mask via uniform thresholding.
    let keepProb = 1.0 - p
    let dtype = withoutDerivative(at: x.dtype ?? .float32)
    let dev = withoutDerivative(at: x.device)
    let u = Tensor.uniform(low: 0.0, high: 1.0, shape: x.shape, dtype: dtype, device: dev)
    let keepMask = withoutDerivative(at: u.gt(p))  // boolean

    // Inverted dropout: y = keep ? x / keepProb : 0
    let pos = x.dividing(keepProb)
    let zero = Tensor.zeros(shape: [], dtype: dtype, device: dev)
    return TorchWhere.select(condition: keepMask, pos, zero)
  }
}

// Manual VJPs that do not re-enter `callAsFunction`.
extension Dropout {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (EmptyTangentVector, Tensor))
  {
    let phase = withoutDerivative(at: Context.local.learningPhase)
    let p = withoutDerivative(at: probability)

    // Inference or p == 0 => identity and identity gradient.
    if phase == .inference || p == 0 {
      return (x, { upstream in (EmptyTangentVector(), upstream) })
    }

    let keepProb = 1.0 - p
    let dtype = withoutDerivative(at: x.dtype ?? .float32)
    let dev = withoutDerivative(at: x.device)

    // Make the mask here and capture it for the pullback.
    let u = Tensor.uniform(low: 0.0, high: 1.0, shape: x.shape, dtype: dtype, device: dev)
    let keepMask = withoutDerivative(at: u.gt(p))  // Bool tensor

    let pos = x.dividing(keepProb)
    let zero = Tensor.zeros(shape: [], dtype: dtype, device: dev)
    let y = TorchWhere.select(condition: keepMask, pos, zero)

    // Pullback: ∂L/∂x = upstream * keepMask / keepProb
    let scale = Tensor(1.0 / keepProb, dtype: dtype, device: dev)
    let maskF = keepMask.to(dtype: dtype)

    let pb: (Tensor) -> (EmptyTangentVector, Tensor) = { upstream in
      let dx = upstream.multiplying(maskF).multiplying(scale)
      return (EmptyTangentVector(), dx)
    }
    return (y, pb)
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> EmptyTangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}
