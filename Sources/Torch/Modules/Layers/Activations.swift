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

/// A parameter-free rectified linear unit activation layer.
public struct ReLU: Layer {
  /// Creates a ReLU activation layer.
  public init() {}

  /// Applies the rectified linear unit element-wise to `x`.
  /// - Parameter x: Input tensor received from the previous layer.
  /// - Returns: A tensor with negative values clamped to zero.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.relu() }

  /// Applies the rectified linear unit while capturing the forward context.
  /// - Parameters:
  ///   - x: Input tensor received from the previous layer.
  ///   - context: Forward-context handle that records intermediate values.
  /// - Returns: A tensor with negative values clamped to zero.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { x.relu() }

  // --- Boilerplate (no parameters) ---
  /// Moves the layer's parameters by `offset`. ReLU has no parameters, so this is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Returns the writable key paths for trainable parameters. ReLU has none.
  public static var parameterKeyPaths: [WritableKeyPath<ReLU, Tensor>] { [] }

  /// Tangent representation for `ReLU`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// The additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because the tangent vector is empty.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because the tangent vector is empty.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// ReLU exposes no parameter key paths.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - LeakyReLU

/// A rectified linear unit variant that preserves a small slope for negative inputs.
public struct LeakyReLU: Layer {
  /// Negative slope (PyTorch default often 0.01).
  @noDerivative public var alpha: Double

  /// Creates a leaky-ReLU activation.
  /// - Parameter alpha: Negative slope applied to inputs less than zero.
  public init(alpha: Double = 0.01) { self.alpha = alpha }

  /// Applies the leaky-ReLU nonlinearity element-wise to `x`.
  /// - Parameter x: Input tensor received from the previous layer.
  /// - Returns: A tensor where negative elements are scaled by `alpha`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // y = relu(x) - α * relu(-x)
    x.relu().adding(x.negated().relu().multiplying(-alpha))
  }

  /// Applies the leaky-ReLU nonlinearity while capturing the forward context.
  /// - Parameters:
  ///   - x: Input tensor received from the previous layer.
  ///   - context: Forward-context handle that records intermediate values.
  /// - Returns: A tensor where negative elements are scaled by `alpha`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate (no parameters) ---
  /// Moves the layer's parameters by `offset`. Leaky-ReLU has no parameters, so this is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Returns the writable key paths for trainable parameters. Leaky-ReLU has none.
  public static var parameterKeyPaths: [WritableKeyPath<LeakyReLU, Tensor>] { [] }
  /// Tangent representation for `LeakyReLU`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// The additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because the tangent vector is empty.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because the tangent vector is empty.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Leaky-ReLU exposes no parameter key paths.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Sigmoid

/// A logistic sigmoid activation layer.
public struct Sigmoid: Layer {
  /// Creates a sigmoid activation layer.
  public init() {}

  /// Applies the logistic sigmoid element-wise to `x`.
  /// - Parameter x: Input tensor received from the previous layer.
  /// - Returns: A tensor with values squeezed to `(0, 1)`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.sigmoid() }

  /// Applies the logistic sigmoid while capturing the forward context.
  /// - Parameters:
  ///   - x: Input tensor received from the previous layer.
  ///   - context: Forward-context handle that records intermediate values.
  /// - Returns: A tensor with values squeezed to `(0, 1)`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { x.sigmoid() }

  // --- Boilerplate ---
  /// Moves the layer's parameters by `offset`. Sigmoid has no parameters, so this is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Returns the writable key paths for trainable parameters. Sigmoid has none.
  public static var parameterKeyPaths: [WritableKeyPath<Sigmoid, Tensor>] { [] }
  /// Tangent representation for `Sigmoid`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// The additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because the tangent vector is empty.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because the tangent vector is empty.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Sigmoid exposes no parameter key paths.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Tanh

/// A hyperbolic tangent activation layer.
public struct Tanh: Layer {
  /// Creates a hyperbolic tangent activation layer.
  public init() {}

  /// Applies the hyperbolic tangent element-wise to `x`.
  /// - Parameter x: Input tensor received from the previous layer.
  /// - Returns: A tensor with values compressed to `[-1, 1]`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.tanh() }

  /// Applies the hyperbolic tangent while capturing the forward context.
  /// - Parameters:
  ///   - x: Input tensor received from the previous layer.
  ///   - context: Forward-context handle that records intermediate values.
  /// - Returns: A tensor with values compressed to `[-1, 1]`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { x.tanh() }

  // --- Boilerplate ---
  /// Moves the layer's parameters by `offset`. Tanh has no parameters, so this is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Returns the writable key paths for trainable parameters. Tanh has none.
  public static var parameterKeyPaths: [WritableKeyPath<Tanh, Tensor>] { [] }
  /// Tangent representation for `Tanh`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// The additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because the tangent vector is empty.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because the tangent vector is empty.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Tanh exposes no parameter key paths.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - GELU (exact / tanh approximation)

/// A Gaussian error linear unit activation layer.
public struct GELU: Layer {
  /// When `true`, uses Hendrycks & Gimpel tanh approximation.
  @noDerivative public var approximate: Bool

  /// Creates a GELU activation layer.
  /// - Parameter approximate: Whether to use the tanh approximation for improved throughput.
  public init(approximate: Bool = false) { self.approximate = approximate }

  /// Applies the GELU nonlinearity to `x`.
  /// - Parameter x: Input tensor received from the previous layer.
  /// - Returns: A tensor with values transformed according to the GELU formula.
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

  /// Applies the GELU nonlinearity while capturing the forward context.
  /// - Parameters:
  ///   - x: Input tensor received from the previous layer.
  ///   - context: Forward-context handle that records intermediate values.
  /// - Returns: A tensor with values transformed according to the GELU formula.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate ---
  /// Moves the layer's parameters by `offset`. GELU has no parameters, so this is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Returns the writable key paths for trainable parameters. GELU has none.
  public static var parameterKeyPaths: [WritableKeyPath<GELU, Tensor>] { [] }
  /// Tangent representation for `GELU`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// The additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because the tangent vector is empty.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because the tangent vector is empty.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// GELU exposes no parameter key paths.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - SiLU / Swish (x * sigmoid(x))

/// A sigmoid-weighted linear unit (Swish) activation layer.
public struct SiLU: Layer {
  /// Creates a SiLU activation layer.
  public init() {}

  /// Applies the SiLU activation element-wise to `x`.
  /// - Parameter x: Input tensor received from the previous layer.
  /// - Returns: A tensor with values transformed according to `x * sigmoid(x)`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.multiplying(x.sigmoid()) }

  /// Applies the SiLU activation while capturing the forward context.
  /// - Parameters:
  ///   - x: Input tensor received from the previous layer.
  ///   - context: Forward-context handle that records intermediate values.
  /// - Returns: A tensor with values transformed according to `x * sigmoid(x)`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate ---
  /// Moves the layer's parameters by `offset`. SiLU has no parameters, so this is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Returns the writable key paths for trainable parameters. SiLU has none.
  public static var parameterKeyPaths: [WritableKeyPath<SiLU, Tensor>] { [] }
  /// Tangent representation for `SiLU`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// The additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because the tangent vector is empty.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because the tangent vector is empty.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// SiLU exposes no parameter key paths.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Softplus

/// A softplus activation layer that smooths its ReLU counterpart.
public struct Softplus: Layer {
  /// Creates a softplus activation layer.
  public init() {}

  /// Applies the softplus activation element-wise to `x`.
  /// - Parameter x: Input tensor received from the previous layer.
  /// - Returns: A tensor with values transformed according to `log(1 + exp(x))`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Delegate to your numerically-stable helper.
    softplus(x)
  }

  /// Applies the softplus activation while capturing the forward context.
  /// - Parameters:
  ///   - x: Input tensor received from the previous layer.
  ///   - context: Forward-context handle that records intermediate values.
  /// - Returns: A tensor with values transformed according to `log(1 + exp(x))`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // --- Boilerplate ---
  /// Moves the layer's parameters by `offset`. Softplus has no parameters, so this is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Returns the writable key paths for trainable parameters. Softplus has none.
  public static var parameterKeyPaths: [WritableKeyPath<Softplus, Tensor>] { [] }
  /// Tangent representation for `Softplus`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// The additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// Adds two tangent vectors. No-op because the tangent vector is empty.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Subtracts two tangent vectors. No-op because the tangent vector is empty.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector { .init() }
    /// Softplus exposes no parameter key paths.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
