// Sources/Torch/Modules/Layers/Linear.swift
import _Differentiation

/// Fully connected affine transform `y = x Wᵀ + b`.
public struct Linear: Layer {
  /// Learnable weight matrix with shape `[outFeatures, inFeatures]`.
  public var weight: Tensor  // [out, in]
  /// Learnable bias vector with shape `[outFeatures]`.
  public var bias: Tensor  // [out]

  /// Creates a linear layer with explicit parameters.
  /// - Parameters:
  ///   - weight: Weight matrix shaped `[outFeatures, inFeatures]`.
  ///   - bias: Bias vector shaped `[outFeatures]`.
  public init(weight: Tensor, bias: Tensor) {
    self.weight = weight
    self.bias = bias
  }

  /// y = x W^T + b
  /// - Parameter x: Input activations.
  /// - Returns: Affine transformation applied to `x`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    x.matmul(weight.transposed(-1, -2)).adding(bias)
  }

  /// Applies the tangent `offset` to the layer's parameters.
  /// - Parameter offset: Tangent vector from differentiation.
  public mutating func move(by offset: TangentVector) {
    self.weight.move(by: offset.weight)
    self.bias.move(by: offset.bias)
  }

  // Parameter traversal
  /// Writable key paths for trainable parameters.
  public static var parameterKeyPaths: [WritableKeyPath<Linear, Tensor>] {
    [\Linear.weight, \Linear.bias]
  }

  // Manually implement AdditiveArithmetic to work around a compiler bug
  /// Tangent representation for `Linear`.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Tangent for the weight matrix.
    public var weight: Tensor
    /// Tangent for the bias vector.
    public var bias: Tensor

    /// Additive identity for the tangent vector.
    public static var zero: TangentVector {
      // Assuming Tensor conforms to AdditiveArithmetic and has a .zero
      TangentVector(weight: .zero, bias: .zero)
    }

    /// Adds two tangent vectors element-wise.
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(
        weight: lhs.weight.adding(rhs.weight),
        bias: lhs.bias.adding(rhs.bias)
      )
    }

    /// Subtracts two tangent vectors element-wise.
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(
        weight: lhs.weight.adding(rhs.weight.multiplying(-1)),
        bias: lhs.bias.adding(rhs.bias.multiplying(-1))
      )
    }

    /// Writable key paths for the tangent components.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\.weight, \.bias]
    }
  }
}
