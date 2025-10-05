// Sources/Torch/Modules/Layers/Activations.swift
//
// WHY
// - Typed, parameter‑free nonlinearities that compose cleanly with Sequential/Builder.
// - Zero-cost wrappers over the differentiable Tensor ops (relu/tanh/sigmoid/erf/...).
// - No trainable parameters → empty ParameterIterable & TangentVector.
// - `Softplus` delegates to your existing numerically-stable helper.
//
// References in this repo: Layer.swift, Sequential.swift, LayerBuilder.swift, Loss.swift (softplus).

import Foundation
import _Differentiation

// MARK: - ReLU

public struct ReLU: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.relu() }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { x.relu() }

  // --- Boilerplate (no parameters) ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<ReLU, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - LeakyReLU

public struct LeakyReLU: Layer {
  /// Negative slope (PyTorch default often 0.01).
  @noDerivative public var alpha: Double

  public init(alpha: Double = 0.01) { self.alpha = alpha }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // y = relu(x) - α * relu(-x)
    x.relu().adding(x.negated().relu().multiplying(-alpha))
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate (no parameters) ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<LeakyReLU, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Sigmoid

public struct Sigmoid: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.sigmoid() }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { x.sigmoid() }

  // --- Boilerplate ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<Sigmoid, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Tanh

public struct Tanh: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.tanh() }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { x.tanh() }

  // --- Boilerplate ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<Tanh, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - GELU (exact / tanh approximation)

public struct GELU: Layer {
  /// When `true`, uses Hendrycks & Gimpel tanh approximation.
  @noDerivative public var approximate: Bool

  public init(approximate: Bool = false) { self.approximate = approximate }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    if approximate {
      // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
      let kappa = Tensor(Foundation.sqrt(2.0 / Double.pi))
      let inner = x.adding(0.044715 * x.multiplying(x).multiplying(x))
      return 0.5 * x.multiplying((1.0 + (kappa.multiplying(inner)).tanh()))
    } else {
      // 0.5 * x * (1 + erf(x / √2))
      let invSqrt2 = 1.0 / Foundation.sqrt(2.0)
      return 0.5 * x.multiplying(1.0 + (x.multiplying(invSqrt2)).erf())
    }
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<GELU, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - SiLU / Swish (x * sigmoid(x))

public struct SiLU: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.multiplying(x.sigmoid()) }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<SiLU, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Softplus

public struct Softplus: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Delegate to your numerically-stable helper.
    softplus(x)
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<Softplus, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
