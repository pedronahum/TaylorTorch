// Sources/Torch/Modules/Layers/Dropout.swift
//
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

import _Differentiation

public struct Dropout: Layer {
  /// Probability of dropping (zeroing) a unit. `rate ∈ [0, 1]`.
  @noDerivative public var rate: Double

  /// Optional factory to produce a deterministic mask for testing.
  /// Must return a Bool tensor with the same shape as the input.
  @noDerivative public var maskFactory: ((Tensor) -> Tensor)?

  public init(rate: Double, maskFactory: ((Tensor) -> Tensor)? = nil) {
    precondition(rate >= 0 && rate <= 1, "Dropout rate must be in [0, 1].")
    self.rate = rate
    self.maskFactory = maskFactory
  }

  // Inference: default to identity so `model(x)` is eval‑safe.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x }

  // Training/inference behavior controlled by ForwardContext.
  @differentiable(reverse, wrt: x)
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
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<Dropout, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector { .init() }
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
