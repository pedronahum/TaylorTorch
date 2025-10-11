// Sources/Torch/Modules/Dense.swift
import Foundation
import _Differentiation

/// Public identity activation that’s safe to use across module boundaries.
@differentiable(reverse)
public func identityActivation(_ x: Tensor) -> Tensor { x }

@derivative(of: identityActivation)
public func _vjpIdentityActivation(_ x: Tensor)
  -> (value: Tensor, pullback: (Tensor) -> Tensor)
{
  (x, { v in v })
}

/// Dense = Linear + activation
///
/// Stores the activation as a non-differentiable closure. Gradients flow through
/// the closure as long as it is `@differentiable(reverse)`.
public struct Dense: Layer {
  public var linear: Linear

  /// Activation closure. Not a parameter; exclude from tangent.
  @noDerivative public var activation: @differentiable(reverse) (Tensor) -> Tensor

  public typealias Input = Tensor
  public typealias Output = Tensor

  // MARK: - Manual TangentVector
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var linear: Linear.TangentVector

    public init(linear: Linear.TangentVector = .zero) {
      self.linear = linear
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(linear: lhs.linear + rhs.linear)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(linear: lhs.linear - rhs.linear)
    }

    // VectorProtocol (scalar ops) – delegate to the nested vector
    public func adding(_ x: Float) -> Self { .init(linear: linear.adding(x)) }
    public func subtracting(_ x: Float) -> Self { .init(linear: linear.subtracting(x)) }
    public func scaled(by s: Float) -> Self { .init(linear: linear.scaled(by: s)) }

    // PointwiseMultiplicative (.* / one / reciprocal) – delegate
    public static var one: Self { .init(linear: .one) }
    public var reciprocal: Self { .init(linear: linear.reciprocal) }
    public static func .* (lhs: Self, rhs: Self) -> Self { .init(linear: lhs.linear .* rhs.linear) }
  }

  public mutating func move(by d: TangentVector) {
    linear.move(by: d.linear)
  }

  // MARK: - Initializers

  /// Glorot/Xavier uniform on `Linear` + configurable activation (defaults to identity).
  public init(
    inputSize inFeatures: Int,
    outputSize outFeatures: Int,
    activation: @escaping @differentiable(reverse) (Tensor) -> Tensor = identityActivation,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.linear = Linear(
      inputSize: inFeatures, outputSize: outFeatures, dtype: dtype, device: device)
    self.activation = activation
  }

  // Convenience factories for common activations you already use.
  public static func tanh(
    inputSize: Int, outputSize: Int, dtype: DType = .float32, device: Device = .cpu
  ) -> Dense {
    Dense(
      inputSize: inputSize, outputSize: outputSize, activation: { $0.tanh() }, dtype: dtype,
      device: device)
  }

  public static func sigmoid(
    inputSize: Int, outputSize: Int, dtype: DType = .float32, device: Device = .cpu
  ) -> Dense {
    Dense(
      inputSize: inputSize, outputSize: outputSize, activation: { $0.sigmoid() }, dtype: dtype,
      device: device)
  }

  /// If your `Tensor` exposes `relu()`, you can use:
  public static func relu(
    inputSize: Int, outputSize: Int, dtype: DType = .float32, device: Device = .cpu
  ) -> Dense {
    Dense(
      inputSize: inputSize, outputSize: outputSize, activation: { $0.relu() }, dtype: dtype,
      device: device)
  }

  // MARK: - Forward

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    activation(linear(x))
  }
}

// MARK: - Manual derivatives (avoid curried-self path)
extension Dense {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: Dense, _ i: Tensor) -> Tensor {
      s.activation(s.linear(i))
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (y, pb)
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> TangentVector)
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
