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
  public static func hardtanh(_ x: Tensor, minVal: Double = -1.0, maxVal: Double = 1.0) -> Tensor {
    x.clamp(min: minVal, max: maxVal)
  }

  /// PyTorch-style: y = clip(x + 3, 0, 6) / 6
  public static func hardsigmoid(_ x: Tensor) -> Tensor {
    x.adding(3.0).clamp(min: 0.0, max: 6.0).dividing(6.0)
  }

  /// y = x * hardsigmoid(x)
  ///
  /// Piece-wise derivative:
  ///   x <= -3 -> 0
  ///   -3 < x < 3 -> (2x + 3)/6
  ///   x >= 3 -> 1
  public static func hardswish(_ x: Tensor) -> Tensor {
    x.multiplying(hardsigmoid(x))
  }

  /// ELU: y = { x                          if x > 0
  ///          { α * (exp(x) - 1)           otherwise
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

public struct HardTanh: Layer {
  @noDerivative public var minVal: Double
  @noDerivative public var maxVal: Double

  public init(minVal: Double = -1.0, maxVal: Double = 1.0) {
    precondition(minVal < maxVal, "minVal must be < maxVal")
    self.minVal = minVal
    self.maxVal = maxVal
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    Activations.hardtanh(x, minVal: minVal, maxVal: maxVal)
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<HardTanh, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

public struct HardSigmoid: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { Activations.hardsigmoid(x) }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<HardSigmoid, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

public struct HardSwish: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { Activations.hardswish(x) }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<HardSwish, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

public struct ELU: Layer {
  @noDerivative public var alpha: Double
  public init(alpha: Double = 1.0) { self.alpha = alpha }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { Activations.elu(x, alpha: alpha) }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // No parameters
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<ELU, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
