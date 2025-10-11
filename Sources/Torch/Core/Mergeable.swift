import _Differentiation

/// A type with values that support differentiable binary operations.
///
/// Used by `BidirectionalRecurrentLayer` as a generic requirement for merge functions.
public protocol Mergeable: Differentiable, AdditiveArithmetic {
  /// Concatenates two values.
  @differentiable(reverse)
  static func concatenate(_ lhs: Self, _ rhs: Self) -> Self

  /// Adds two values and produces their sum.
  ///
  /// - Note: renaming `sum` to `+` results in a compiler crash when conforming `Tensor` to
  /// `Mergeable` (SR-13229).
  @differentiable(reverse)
  static func sum(_ lhs: Self, _ rhs: Self) -> Self

  /// Averages two values.
  @differentiable(reverse)
  static func average(_ lhs: Self, _ rhs: Self) -> Self

  /// Multiplies two values.
  @differentiable(reverse)
  static func multiply(_ lhs: Self, _ rhs: Self) -> Self

  /// Stack two values.
  @differentiable(reverse)
  static func stack(_ lhs: Self, _ rhs: Self) -> Self
}

extension Tensor: Mergeable {
  /// Concatenates two tensors along last axis.
  @differentiable(reverse)
  public static func concatenate(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    let lhsRank = withoutDerivative(at: lhs.rank)
    let rhsRank = withoutDerivative(at: rhs.rank)
    precondition(lhsRank == rhsRank, "concatenate: rank mismatch")
    precondition(lhsRank > 0, "concatenate: tensors must have rank ≥ 1")
    let axis = withoutDerivative(at: _normalizeDimension(-1, rank: lhsRank))
    return Tensor.cat([lhs, rhs], dim: axis)
  }

  /// Adds two values and produces their sum.
  @differentiable(reverse)
  public static func sum(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    lhs.adding(rhs)
  }

  /// Averages two values.
  @differentiable(reverse)
  public static func average(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    let half: Float = 0.5
    return lhs.adding(rhs).multiplying(half)
  }

  /// Multiplies two values.
  @differentiable(reverse)
  public static func multiply(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    lhs.multiplying(rhs)
  }

  /// Stack two values.
  @differentiable(reverse)
  public static func stack(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    Tensor.stack([lhs, rhs])
  }
}

/// Concatenates two values.
@differentiable(reverse)
public func concatenate<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.concatenate(first, second)
}

/// Adds two values and produces their sum.
@differentiable(reverse)
public func sum<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.sum(first, second)
}

/// Averages two values.
@differentiable(reverse)
public func average<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.average(first, second)
}

/// Multiplies two values.
@differentiable(reverse)
public func multiply<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.multiply(first, second)
}

/// Stack two values.
@differentiable(reverse)
public func stack<T: Mergeable>(
  _ first: T,
  _ second: T
) -> T {
  T.stack(first, second)
}
