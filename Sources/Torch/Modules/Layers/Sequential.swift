// Sources/Torch/Modules/Layers/Sequential.swift
import _Differentiation

/// Chains two layers such that the output of `L1` feeds into `L2`.
public struct Sequential<L1: Layer, L2: Layer>: Layer {
  /// First layer in the composition.
  public var l1: L1
  /// Second layer in the composition.
  public var l2: L2

  /// Creates a sequential composition of `l1` followed by `l2`.
  /// - Parameters:
  ///   - l1: Leading layer.
  ///   - l2: Trailing layer that consumes the output of `l1`.
  public init(_ l1: L1, _ l2: L2) {
    self.l1 = l1
    self.l2 = l2
  }

  /// Applies the composed layers to `x`.
  /// - Parameter x: Input tensor.
  /// - Returns: Output of `l2(l1(x))`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    l2(l1(x))
  }

  /// Applies the tangent `offset` to both layers.
  /// - Parameter offset: Tangent vector produced by differentiation.
  public mutating func move(by offset: TangentVector) {
    self.l1.move(by: offset.l1)
    self.l2.move(by: offset.l2)
  }

  // Corrected syntax with parentheses to disambiguate the method call.
  /// Writable key paths for the composed layers' parameters.
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

  /// Tangent representation for `Sequential`.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Tangent for the leading layer.
    public var l1: L1.TangentVector
    /// Tangent for the trailing layer.
    public var l2: L2.TangentVector

    /// Additive identity for the tangent vector.
    public static var zero: TangentVector {
      TangentVector(l1: .zero, l2: .zero)
    }

    /// Adds two tangent vectors element-wise.
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(l1: lhs.l1 + rhs.l1, l2: lhs.l2 + rhs.l2)
    }

    /// Subtracts two tangent vectors element-wise.
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      TangentVector(l1: lhs.l1 - rhs.l1, l2: lhs.l2 - rhs.l2)
    }

    // Corrected syntax with parentheses to disambiguate the method call.
    /// Writable key paths for the tangent components.
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
/// Returns a two-layer sequential composition using the chaining operator.
/// - Parameters:
///   - lhs: Leading layer.
///   - rhs: Trailing layer.
/// - Returns: `Sequential(lhs, rhs)`.
public func >>> <A: Layer, B: Layer>(lhs: A, rhs: B) -> Sequential<A, B> {
  .init(lhs, rhs)
}
