import _Differentiation

/// A differentiable version of the sum reduction.
@differentiable(reverse)
public func sum(_ tensor: Tensor) -> Tensor {
  return tensor.sum()
}

/// A differentiable version of the mean reduction.
@differentiable(reverse)
public func mean(_ tensor: Tensor) -> Tensor {
  return tensor.mean()
}

/// A differentiable version of the abs function
@differentiable(reverse)
public func abs(_ tensor: Tensor) -> Tensor {
  return tensor.abs()
}

/// A differentiable version of the negation operation.
@differentiable(reverse)
public func negated(_ tensor: Tensor) -> Tensor {
  return tensor.negated()
}

/// A differentiable version of the ReLU operation.
@differentiable(reverse)
public func relu(_ tensor: Tensor) -> Tensor {
  return tensor.relu()
}

/// A differentiable version of the exponential function.
@differentiable(reverse)
public func exp(_ tensor: Tensor) -> Tensor {
  return tensor.exp()
}

/// A differentiable version of the natural logarithm.
@differentiable(reverse)
public func log(_ tensor: Tensor) -> Tensor {
  return tensor.log()
}

/// A differentiable version of the square root.
@differentiable(reverse)
public func sqrt(_ tensor: Tensor) -> Tensor {
  return tensor.sqrt()
}

/// A differentiable binary subtraction with scaling.
@differentiable(reverse)
public func subtract(
  _ lhs: Tensor,
  _ rhs: Tensor,
  alpha: Scalar = .int64(1)
) -> Tensor {
  return lhs.subtracting(rhs, alpha: alpha)
}

/// A differentiable binary multiplication.
@differentiable(reverse)
public func multiply(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.multiplying(rhs)
}

/// A differentiable binary division.
@differentiable(reverse)
public func divide(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.dividing(rhs)
}

/// A differentiable tensor-scalar subtraction.
@differentiable(reverse)
public func subtract<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.subtracting(scalar)
}

/// A differentiable tensor-scalar multiplication.
@differentiable(reverse)
public func multiply<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.multiplying(scalar)
}

/// A differentiable tensor-scalar division.
@differentiable(reverse)
public func divide<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.dividing(scalar)
}

/// A differentiable scalar power.
@differentiable(reverse)
public func pow<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ power: T
) -> Tensor {
  return tensor.pow(power)
}

/// A differentiable tensor power.
@differentiable(reverse)
public func pow(_ tensor: Tensor, _ other: Tensor) -> Tensor {
  return tensor.pow(other)
}

/// A differentiable clamp operation.
@differentiable(reverse)
public func clamp<T: TorchArithmetic>(
  _ tensor: Tensor,
  min: T,
  max: T
) -> Tensor {
  return tensor.clamp(min: min, max: max)
}

/// A differentiable dimensional sum reduction.
@differentiable(reverse)
public func sum(
  _ tensor: Tensor,
  dim: Int,
  keepdim: Bool = false
) -> Tensor {
  return tensor.sum(dim: dim, keepdim: keepdim)
}

/// A differentiable dimensional mean reduction.
@differentiable(reverse)
public func mean(
  _ tensor: Tensor,
  dim: Int,
  keepdim: Bool = false
) -> Tensor {
  return tensor.mean(dim: dim, keepdim: keepdim)
}

/// A differentiable matrix multiplication.
@differentiable(reverse)
public func matmul(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.matmul(rhs)
}

/// A differentiable dot product.
@differentiable(reverse)
public func dot(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.dot(rhs)
}

/// Element-wise equality comparison.
public func eq(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.eq(rhs)
}

/// Element-wise less-than comparison.
public func lt(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.lt(rhs)
}

/// Element-wise less-than-or-equal comparison.
public func le(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.le(rhs)
}

/// Element-wise greater-than comparison.
public func gt(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.gt(rhs)
}

/// Element-wise greater-than-or-equal comparison.
public func ge(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.ge(rhs)
}

/// Element-wise equality comparison with a scalar.
public func eq<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.eq(scalar)
}

/// Element-wise less-than comparison with a scalar.
public func lt<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.lt(scalar)
}

/// Element-wise less-than-or-equal comparison with a scalar.
public func le<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.le(scalar)
}

/// Element-wise greater-than comparison with a scalar.
public func gt<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.gt(scalar)
}

/// Element-wise greater-than-or-equal comparison with a scalar.
public func ge<T: TorchArithmetic>(
  _ tensor: Tensor,
  _ scalar: T
) -> Tensor {
  return tensor.ge(scalar)
}

/// Ternary where selection.
public func torchWhere(condition: Tensor, _ a: Tensor, _ b: Tensor) -> Tensor {
  return TorchWhere.select(condition: condition, a, b)
}
