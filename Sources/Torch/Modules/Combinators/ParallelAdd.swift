// Sources/Torch/Modules/Combinators/ParallelAdd.swift
//
// WHY: A common merge pattern: compute two branches and add them.
// Useful for “bottleneck” blocks, attention output merges, etc.

import _Differentiation

public struct ParallelAdd<L1: Layer, L2: Layer>: Layer {
  public var l1: L1
  public var l2: L2

  public init(_ l1: L1, _ l2: L2) {
    self.l1 = l1
    self.l2 = l2
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { l1(x).adding(l2(x)) }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    l1.call(x, context: context).adding(l2.call(x, context: context))
  }

  public mutating func move(by offset: TangentVector) {
    l1.move(by: offset.l1)
    l2.move(by: offset.l2)
  }

  public static var parameterKeyPaths: [WritableKeyPath<ParallelAdd, Tensor>] {
    var paths: [WritableKeyPath<ParallelAdd, Tensor>] = []
    for kp in L1.parameterKeyPaths {
      paths.append((\ParallelAdd.l1).appending(kp))
    }
    for kp in L2.parameterKeyPaths {
      paths.append((\ParallelAdd.l2).appending(kp))
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
