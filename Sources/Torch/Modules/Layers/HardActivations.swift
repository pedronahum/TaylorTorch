// Sources/Torch/Modules/Layers/HardActivations.swift
//
// WHY
// - Fast, piece‑wise linear activations common in mobile/edge models (HardTanh,
//   HardSigmoid, HardSwish), plus ELU.
// - Stateless Layer wrappers compose cleanly with Sequential/Builder and training
//   context without introducing parameters.
//
// Notes
// - Uses clamp/comparison‑based masking whose pullbacks you already test
//   (e.g., clamp masks gradients outside the active region; where/mask splits
//   upstream flow). See TorchTests for clamp/where/comparison derivatives.

import _Differentiation

// MARK: - Functional APIs (Activations namespace)

extension Activations {
  /// y = clamp(x, minVal, maxVal)
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - minVal: Lower bound of the hard clip.
  ///   - maxVal: Upper bound of the hard clip.
  /// - Returns: Tensor with values clamped to `[minVal, maxVal]`.
  public static func hardtanh(_ x: Tensor, minVal: Double = -1.0, maxVal: Double = 1.0) -> Tensor {
    x.clamp(min: minVal, max: maxVal)
  }

  /// PyTorch-style: y = clip(x + 3, 0, 6) / 6
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor transformed by the hard-sigmoid function.
  public static func hardsigmoid(_ x: Tensor) -> Tensor {
    x.adding(3.0).clamp(min: 0.0, max: 6.0).dividing(6.0)
  }

  /// y = x * hardsigmoid(x)
  ///
  /// Piece-wise derivative:
  ///   x <= -3 -> 0
  ///   -3 < x < 3 -> (2x + 3)/6
  ///   x >= 3 -> 1
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor transformed by the hard-swish function.
  public static func hardswish(_ x: Tensor) -> Tensor {
    x.multiplying(hardsigmoid(x))
  }

  /// ELU: y = { x                          if x > 0
  ///          { α * (exp(x) - 1)           otherwise
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - alpha: Slope for the negative region.
  /// - Returns: Tensor transformed by the ELU function.
  public static func elu(_ x: Tensor, alpha: Double = 1.0) -> Tensor {
    let a = withoutDerivative(at: Tensor(alpha).to(dtype: x.dtype ?? .float32).to(device: x.device))
    // Masked piece‑wise; comparisons have zero pullback, so only chosen branch gets gradient.
    let posMask = withoutDerivative(at: x.gt(0.0)).to(dtype: a.dtype!)
    let negMask = withoutDerivative(at: Tensor(1.0, dtype: a.dtype!, device: x.device))
      .subtracting(posMask)
    let neg = a.multiplying(x.exp().subtracting(1.0))
    return posMask.multiplying(x).adding(negMask.multiplying(neg))
  }
}

// MARK: - Layer wrappers (stateless)

/// Stateless hard-tanh activation layer.
public struct HardTanh: Layer {
  @noDerivative public var minVal: Double
  @noDerivative public var maxVal: Double

  /// Creates a hard-tanh activation layer.
  /// - Parameters:
  ///   - minVal: Lower clamp bound.
  ///   - maxVal: Upper clamp bound.
  public init(minVal: Double = -1.0, maxVal: Double = 1.0) {
    precondition(minVal < maxVal, "minVal must be < maxVal")
    self.minVal = minVal
    self.maxVal = maxVal
  }

  /// Applies the hard-tanh transformation.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor with values clamped to `[minVal, maxVal]`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    Activations.hardtanh(x, minVal: minVal, maxVal: maxVal)
  }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor with values clamped to `[minVal, maxVal]`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  /// HardTanh has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// HardTanh exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<HardTanh, Tensor>] { [] }

  /// Tangent representation for `HardTanh`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors; returns an empty tangent.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors; returns an empty tangent.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

/// Stateless hard-sigmoid activation layer.
public struct HardSigmoid: Layer {
  /// Creates a hard-sigmoid activation layer.
  public init() {}

  /// Applies the hard-sigmoid transformation.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor transformed by the hard-sigmoid function.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { Activations.hardsigmoid(x) }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor transformed by the hard-sigmoid function.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  /// HardSigmoid has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// HardSigmoid exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<HardSigmoid, Tensor>] { [] }

  /// Tangent representation for `HardSigmoid`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors; returns an empty tangent.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors; returns an empty tangent.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

/// Stateless hard-swish activation layer.
public struct HardSwish: Layer {
  /// Creates a hard-swish activation layer.
  public init() {}

  /// Applies the hard-swish transformation.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor transformed by the hard-swish function.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { Activations.hardswish(x) }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor transformed by the hard-swish function.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  /// HardSwish has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// HardSwish exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<HardSwish, Tensor>] { [] }

  /// Tangent representation for `HardSwish`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors; returns an empty tangent.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors; returns an empty tangent.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

/// Exponential linear unit activation layer.
public struct ELU: Layer {
  @noDerivative public var alpha: Double
  /// Creates an ELU activation layer.
  /// - Parameter alpha: Slope applied to the negative region.
  public init(alpha: Double = 1.0) { self.alpha = alpha }

  /// Applies the ELU transformation.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor transformed by the ELU function.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { Activations.elu(x, alpha: alpha) }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor transformed by the ELU function.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  /// ELU has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// ELU exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<ELU, Tensor>] { [] }

  /// Tangent representation for `ELU`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors; returns an empty tangent.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors; returns an empty tangent.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
