import _Differentiation

// Dropout (inverted): randomly zeroes a fraction `rate` of inputs during training,
// scales survivors by 1/(1 - rate). Inference path is a no-op.
//
// WHY:
// - Regularization that preserves expected activation magnitude at inference.
// - Uses ForwardContext(training:) instead of a global learning phase,
//   aligning with your Layer protocol’s contextful call.  See Layer.swift + ForwardContext.swift.
//
// References: S4TF Dropout design (inverted scaling; learning-phase gating).
// https://github.com/tensorflow/swift-apis/blob/f51ee4618d652a2419e998bf9418ad80bda67454/Sources/TensorFlow/Layers/Dropout.swift
// Layer protocol & ForwardContext in this repo.
// (Citations placed in the PR description / discussion.)

/// Inverted dropout layer that zeroes activations with probability `rate` during training.
public struct Dropout: Layer {
  /// Probability of dropping (zeroing) a unit. `rate ∈ [0, 1]`.
  @noDerivative public var rate: Double

  /// Optional factory to produce a deterministic mask for testing.
  /// Must return a Bool tensor with the same shape as the input.
  @noDerivative public var maskFactory: ((Tensor) -> Tensor)?

  /// Creates an inverted-dropout layer.
  /// - Parameters:
  ///   - rate: Probability of setting each activation to zero.
  ///   - maskFactory: Optional closure that produces a deterministic mask for tests.
  public init(rate: Double, maskFactory: ((Tensor) -> Tensor)? = nil) {
    precondition(rate >= 0 && rate <= 1, "Dropout rate must be in [0, 1].")
    self.rate = rate
    self.maskFactory = maskFactory
  }

  // Inference: default to identity so `model(x)` is eval‑safe.
  /// Returns `x` unchanged when invoked without a training context.
  /// - Parameter x: Input activations.
  /// - Returns: The unmodified input tensor.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x }

  // Training/inference behavior controlled by ForwardContext.
  /// Applies dropout when `context.training` is `true`.
  /// - Parameters:
  ///   - x: Input activations.
  ///   - context: Forward context that gates training mode.
  /// - Returns: Activations with elements zeroed according to the dropout mask.
  @differentiable(reverse,wrt: x)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    guard context.training else { return x }
    if rate <= 0 { return x }
    if rate >= 1 { return x - x }

    let keep = 1.0 - rate

    // Build a boolean mask of the same shape as x. RNG is fenced off from AD.
    let detached = withoutDerivative(at: x)
    let maskBool: Tensor
    if let make = maskFactory {
      let produced = make(detached)
      precondition(
        produced.dtype == .bool && produced.shape == detached.shape,
        "maskFactory must return Bool tensor with same shape as input")
      maskBool = withoutDerivative(at: produced)
    } else {
      // Sample U[0,1); keep if u < keep
      let uniform = Tensor.uniform(
        low: 0.0,
        high: 1.0,
        shape: detached.shape,
        dtype: .float32,
        device: detached.device)
      let threshold = Tensor(keep).to(dtype: .float32).to(device: detached.device)
      maskBool = withoutDerivative(at: uniform .< threshold)
    }

    // Inverted dropout: E[out] == x
    let mask = withoutDerivative(at: maskBool.to(dtype: x.dtype!))
    return x.multiplying(mask).dividing(keep)
  }

  // No trainable parameters.
  /// Dropout has no trainable parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Dropout exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<Dropout, Tensor>] { [] }

  /// Tangent representation for `Dropout`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because there are no parameters.
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because there are no parameters.
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
