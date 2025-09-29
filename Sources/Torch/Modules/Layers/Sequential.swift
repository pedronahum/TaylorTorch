// Sources/Torch/Modules/Layers/Sequential.swift
import _Differentiation

public struct Sequential<L1: Layer, L2: Layer>: Layer {
  public var l1: L1
  public var l2: L2

  public init(_ l1: L1, _ l2: L2) {
    self.l1 = l1
    self.l2 = l2
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    l2(l1(x))
  }

  public mutating func move(by offset: TangentVector) {
    self.l1.move(by: offset.l1)
    self.l2.move(by: offset.l2)
  }

  // Corrected syntax with parentheses to disambiguate the method call.
  public static var parameterKeyPaths: [WritableKeyPath<Sequential, Tensor>] {
    var paths: [WritableKeyPath<Sequential, Tensor>] = []
    for kp in L1.parameterKeyPaths {
      paths.append((\Sequential.l1).appending(kp))
    }
    for kp in L2.parameterKeyPaths {
      paths.append((\Sequential.l2).appending(kp))
    }
    return paths
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var l1: L1.TangentVector
    public var l2: L2.TangentVector

    public static var zero: TangentVector {
      TangentVector(l1: .zero, l2: .zero)
    }

    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(l1: lhs.l1 + rhs.l1, l2: lhs.l2 + rhs.l2)
    }

    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(l1: lhs.l1 - rhs.l1, l2: lhs.l2 - rhs.l2)
    }

    // Corrected syntax with parentheses to disambiguate the method call.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      var paths: [WritableKeyPath<TangentVector, Tensor>] = []
      for kp in L1.TangentVector.parameterKeyPaths {
        paths.append((\Self.l1).appending(kp))
      }
      for kp in L2.TangentVector.parameterKeyPaths {
        paths.append((\Self.l2).appending(kp))
      }
      return paths
    }
  }
}

// Convenience: chain operator for readability.
infix operator >>> : AdditionPrecedence
@inlinable
public func >>> <A: Layer, B: Layer>(lhs: A, rhs: B) -> Sequential<A, B> {
  .init(lhs, rhs)
}
