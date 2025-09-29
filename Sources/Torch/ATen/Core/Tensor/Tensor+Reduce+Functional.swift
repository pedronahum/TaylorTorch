import _Differentiation

/// A differentiable version of the global minimum reduction.
@differentiable(reverse)
public func min(_ tensor: Tensor) -> Tensor {
  return tensor.min()
}

/// A differentiable version of the global maximum reduction.
@differentiable(reverse)
public func max(_ tensor: Tensor) -> Tensor {
  return tensor.max()
}

/// A differentiable dimensional minimum returning values and indices.
@differentiable(reverse)
public func min(
  _ tensor: Tensor,
  dim: Int,
  keepdim: Bool = false
) -> TensorPair {
  return tensor.min(dim: dim, keepdim: keepdim)
}

/// A differentiable dimensional maximum returning values and indices.
@differentiable(reverse)
public func max(
  _ tensor: Tensor,
  dim: Int,
  keepdim: Bool = false
) -> TensorPair {
  return tensor.max(dim: dim, keepdim: keepdim)
}

/// Dimensional argmin returning the indices tensor.
public func argmin(
  _ tensor: Tensor,
  dim: Int,
  keepdim: Bool = false
) -> Tensor {
  return tensor.argmin(dim: dim, keepdim: keepdim)
}

/// Dimensional argmax returning the indices tensor.
public func argmax(
  _ tensor: Tensor,
  dim: Int,
  keepdim: Bool = false
) -> Tensor {
  return tensor.argmax(dim: dim, keepdim: keepdim)
}

/// Differentiable top-k selection returning values and indices.
@differentiable(reverse)
public func topk(
  _ tensor: Tensor,
  k: Int,
  dim: Int = -1,
  largest: Bool = true,
  sorted: Bool = true
) -> TensorPair {
  return tensor.topk(k, dim: dim, largest: largest, sorted: sorted)
}

/// Differentiable sort returning sorted values and indices.
@differentiable(reverse)
public func sort(
  _ tensor: Tensor,
  dim: Int = -1,
  descending: Bool = false
) -> TensorPair {
  return tensor.sort(dim: dim, descending: descending)
}

/// Differentiable element-wise minimum between two tensors.
@differentiable(reverse)
public func minimum(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.minimum(rhs)
}

/// Differentiable element-wise maximum between two tensors.
@differentiable(reverse)
public func maximum(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  return lhs.maximum(rhs)
}
