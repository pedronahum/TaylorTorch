// Sources/Torch/Modules/Layers/Linear.swift
import _Differentiation

public struct Linear: Layer {
  public var weight: Tensor  // [out, in]
  public var bias: Tensor  // [out]

  public init(weight: Tensor, bias: Tensor) {
    self.weight = weight
    self.bias = bias
  }

  /// y = x W^T + b
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    x.matmul(weight.transposed(-1, -2)).adding(bias)
  }

  public mutating func move(by offset: TangentVector) {
    self.weight.move(by: offset.weight)
    self.bias.move(by: offset.bias)
  }

  // Parameter traversal
  public static var parameterKeyPaths: [WritableKeyPath<Linear, Tensor>] {
    [\Linear.weight, \Linear.bias]
  }

  // Manually implement AdditiveArithmetic to work around a compiler bug
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var weight: Tensor
    public var bias: Tensor

    public static var zero: TangentVector {
      // Assuming Tensor conforms to AdditiveArithmetic and has a .zero
      TangentVector(weight: .zero, bias: .zero)
    }

    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(
        weight: lhs.weight.adding(rhs.weight),
        bias: lhs.bias.adding(rhs.bias)
      )
    }

    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(
        weight: lhs.weight.adding(rhs.weight.multiplying(-1)),
        bias: lhs.bias.adding(rhs.bias.multiplying(-1))
      )
    }

    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\.weight, \.bias]
    }
  }
}
