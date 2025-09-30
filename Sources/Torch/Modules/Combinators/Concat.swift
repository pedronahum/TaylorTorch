// Sources/Torch/Modules/Combinators/Concat.swift
//
// WHY: Feature concatenation along a chosen axis (e.g., U‑Net skips,
// multi-branch blocks). TaylorTorch supports differentiable `cat`
// and tests its pullback behavior; this wrapper makes it a reusable module.

// Sources/Torch/Modules/Combinators/Concat.swift
import _Differentiation

public struct Concat<L1: Layer, L2: Layer>: Layer {
  public var l1: L1
  public var l2: L2
  // Mark this property as a non-differentiable constant.
  @noDerivative public var dim: Int

  public init(_ l1: L1, _ l2: L2, dim: Int) {
    self.l1 = l1
    self.l2 = l2
    self.dim = dim
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // The `withoutDerivative(at:)` is no longer needed here.
    Tensor.cat([l1(x), l2(x)], dim: dim)
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    // The `withoutDerivative(at:)` is no longer needed here either.
    Tensor.cat([l1.call(x, context: context), l2.call(x, context: context)], dim: dim)
  }

  public mutating func move(by offset: TangentVector) {
    l1.move(by: offset.l1)
    l2.move(by: offset.l2)
  }

  public static var parameterKeyPaths: [WritableKeyPath<Concat, Tensor>] {
    var paths: [WritableKeyPath<Concat, Tensor>] = []
    for kp in L1.parameterKeyPaths {
      paths.append((\Concat.l1).appending(kp))
    }
    for kp in L2.parameterKeyPaths {
      paths.append((\Concat.l2).appending(kp))
    }
    return paths
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var l1: L1.TangentVector
    public var l2: L2.TangentVector

    public static var zero: TangentVector { .init(l1: .zero, l2: .zero) }
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(l1: lhs.l1 + rhs.l1, l2: lhs.l2 + rhs.l2)
    }
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(l1: lhs.l1 - rhs.l1, l2: lhs.l2 - rhs.l2)
    }

    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      var paths: [WritableKeyPath<TangentVector, Tensor>] = []
      for kp in L1.TangentVector.parameterKeyPaths {
        paths.append((\TangentVector.l1).appending(kp))
      }
      for kp in L2.TangentVector.parameterKeyPaths {
        paths.append((\TangentVector.l2).appending(kp))
      }
      return paths
    }
  }
}
