// Sources/Torch/Modules/Combinators/Residual.swift
//
// WHY: Residual connections (x + f(x)) are everywhere (ResNets, Transformers).
// This typed wrapper composes parameter traversal and optimizers.
// NOTE: `f(x)` must match the shape of `x` for the `+` to be valid.

import _Differentiation

public struct Residual<L: Layer>: Layer {
  public var layer: L

  public init(_ layer: L) { self.layer = layer }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.adding(layer(x)) }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    x.adding(layer.call(x, context: context))
  }

  public mutating func move(by offset: TangentVector) { layer.move(by: offset.layer) }

  // Rewritten with a `for` loop to avoid the compiler bug.
  public static var parameterKeyPaths: [WritableKeyPath<Residual, Tensor>] {
    var paths: [WritableKeyPath<Residual, Tensor>] = []
    for kp in L.parameterKeyPaths {
      paths.append((\Residual.layer).appending(kp))
    }
    return paths
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var layer: L.TangentVector
    public static var zero: TangentVector { .init(layer: .zero) }
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(layer: lhs.layer + rhs.layer)
    }
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(layer: lhs.layer - rhs.layer)
    }

    // Rewritten with a `for` loop to avoid the compiler bug.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      var paths: [WritableKeyPath<TangentVector, Tensor>] = []
      for kp in L.TangentVector.parameterKeyPaths {
        paths.append((\TangentVector.layer).appending(kp))
      }
      return paths
    }
  }
}
