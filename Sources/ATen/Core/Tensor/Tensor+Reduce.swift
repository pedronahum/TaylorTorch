@preconcurrency import ATenCXX

/// Convenience wrapper for returning both values and indices from reduction APIs.
public struct TensorPair: Sendable {
  /// Tensor containing the reduction values.
  public let values: Tensor
  /// Tensor containing the indices associated with each value.
  public let indices: Tensor
  @inlinable public init(values: Tensor, indices: Tensor) {
    self.values = values
    self.indices = indices
  }
}

public extension Tensor {
  // ---- Full scalar reductions (rank-0)
  /// Returns the minimum value across all elements.
  @inlinable func min() -> Tensor { Tensor(_impl.minAll()) }
  /// Returns the maximum value across all elements.
  @inlinable func max() -> Tensor { Tensor(_impl.maxAll()) }

  // ---- Reductions along a dimension (values + indices)
  /// Returns the minimum value and its index along `dim`.
  /// - Parameters:
  ///   - dim: Dimension to reduce.
  ///   - keepdim: When `true`, retains reduced dimensions with length `1`.
  @inlinable func min(dim: Int, keepdim: Bool = false) -> TensorPair {
    // ✅ Call the separate C++ functions for values and indices
    let values = Tensor(_impl.minDim(Int64(dim), keepdim))
    let indices = Tensor(_impl.argminDim(Int64(dim), keepdim))
    return TensorPair(values: values, indices: indices)
  }
  /// Returns the maximum value and its index along `dim`.
  /// - Parameters:
  ///   - dim: Dimension to reduce.
  ///   - keepdim: When `true`, retains reduced dimensions with length `1`.
  @inlinable func max(dim: Int, keepdim: Bool = false) -> TensorPair {
    // ✅ Call the separate C++ functions for values and indices
    let values = Tensor(_impl.maxDim(Int64(dim), keepdim))
    let indices = Tensor(_impl.argmaxDim(Int64(dim), keepdim))
    return TensorPair(values: values, indices: indices)
  }

  // ---- Argmin/Argmax along a dimension (indices only)
  // These are already correct as they only return one tensor.
  /// Returns the indices of the minimum values along `dim`.
  @inlinable func argmin(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.argminDim(Int64(dim), keepdim))
  }
  /// Returns the indices of the maximum values along `dim`.
  @inlinable func argmax(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.argmaxDim(Int64(dim), keepdim))
  }

  // ---- topk / sort (values + indices)
  /// Returns the top-`k` values and their indices along `dim`.
  /// - Parameters:
  ///   - k: Number of elements to select (must be > 0).
  ///   - dim: Dimension along which to select (defaults to last dimension).
  ///   - largest: When `true`, returns the largest values; otherwise the smallest.
  ///   - sorted: Controls whether the resulting values are sorted.
  @inlinable func topk(_ k: Int, dim: Int = -1, largest: Bool = true, sorted: Bool = true) -> TensorPair {
    // ✅ Call the separate C++ functions for values and indices
    let values = Tensor(_impl.topk_values(Int64(k), Int64(dim), largest, sorted))
    let indices = Tensor(_impl.topk_indices(Int64(k), Int64(dim), largest, sorted))
    return TensorPair(values: values, indices: indices)
  }
  /// Returns the sorted values and their indices along `dim`.
  /// - Parameters:
  ///   - dim: Dimension along which to sort.
  ///   - descending: When `true`, sorts in descending order.
  @inlinable func sort(dim: Int = -1, descending: Bool = false) -> TensorPair {
    // ✅ Call the separate C++ functions for values and indices
    let values = Tensor(_impl.sortDim_values(Int64(dim), descending))
    let indices = Tensor(_impl.sortDim_indices(Int64(dim), descending))
    return TensorPair(values: values, indices: indices)
  }

  // ---- Element-wise minimum / maximum (tensor ⊗ tensor)
  /// Returns the element-wise minimum of two tensors.
  @inlinable func minimum(_ other: Tensor) -> Tensor { Tensor(_impl.minimum(other._impl)) }
  /// Returns the element-wise maximum of two tensors.
  @inlinable func maximum(_ other: Tensor) -> Tensor { Tensor(_impl.maximum(other._impl)) }
}
