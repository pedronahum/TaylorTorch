// Sources/Torch/Modules/Linear.swift
import Foundation
import _Differentiation

/// Dense affine transformation: y = x.matmul(weight) + bias
/// - Shapes:
///   - input  x: [batch, in]
///   - weight : [in, out]
///   - bias   : [out]
public struct Linear: Layer {
  // Parameters
  public var weight: Tensor  // [in, out]
  public var bias: Tensor  // [out]

  // Layer signatures
  public typealias Input = Tensor
  public typealias Output = Tensor

  // MARK: - Manual TangentVector (avoid synthesis pitfalls)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // explicit zero/+/-
    KeyPathIterable,  // required by Module
    VectorProtocol,  // required by Module
    PointwiseMultiplicative  // required by Module
  {
    public typealias VectorSpaceScalar = Float
    public var weight: Tensor
    public var bias: Tensor

    public init(weight: Tensor = Tensor(0), bias: Tensor = Tensor(0)) {
      self.weight = weight
      self.bias = bias
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight + rhs.weight, bias: lhs.bias + rhs.bias)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight - rhs.weight, bias: lhs.bias - rhs.bias)
    }
  }

  // Required when manually defining TangentVector
  public mutating func move(by d: TangentVector) {
    weight += d.weight
    bias += d.bias
  }

  // MARK: - Initializers
  /// Glorot/Xavier uniform init (no transpose needed; weight is [in, out]).
  public init(
    inputSize inFeatures: Int,
    outputSize outFeatures: Int,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    let a = Foundation.sqrt(6.0 / Double(inFeatures + outFeatures))  // Glorot
    self.weight = Tensor.uniform(
      low: -a, high: a, shape: [inFeatures, outFeatures], dtype: dtype, device: device)
    self.bias = Tensor.zeros(shape: [outFeatures], dtype: dtype, device: device)
  }

  // MARK: - Forward
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    x.matmul(weight).adding(bias)
  }
}

// MARK: - Manual derivatives (avoid curried-self path)
extension Linear {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: Linear, _ i: Tensor) -> Tensor {
      i.matmul(s.weight).adding(s.bias)
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
