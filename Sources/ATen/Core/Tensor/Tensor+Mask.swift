@preconcurrency import ATenCXX

public extension Tensor {
  /// Return a tensor where positions selected by `mask` are replaced by `value` and others unchanged.
  /// - Parameters:
  ///   - mask: Boolean tensor broadcastable to `self`.
  ///   - value: Scalar broadcast across positions where the mask is `true`.
  @inlinable
  func maskedFill<T: TorchArithmetic>(where mask: Tensor, with value: T) -> Tensor {
    Tensor(_impl.maskedFillScalar(mask._impl, value._cxxScalar))
  }

  /// Return a tensor where positions selected by `mask` are replaced by `values` and others unchanged.
  /// - Parameters:
  ///   - mask: Boolean tensor broadcastable to both operands.
  ///   - values: Tensor providing replacement elements.
  /// out = mask ? values[i] : self[i]
  @inlinable
  func maskedFill(where mask: Tensor, with values: Tensor) -> Tensor {
    // ✅ Call the new global C++ helper function
    Tensor(masked_fill_tensor_helper(_impl, mask._impl, values._impl))
  }

  /// Return a 1-D tensor containing the elements of `self` for which `mask` evaluates to `true`.
  /// - Parameter mask: Boolean tensor broadcastable to `self`.
  @inlinable
  func maskedSelect(where mask: Tensor) -> Tensor {
    Tensor(_impl.maskedSelect(mask._impl))
  }

  // Boolean reductions
  /// Returns a rank-0 boolean tensor that is `true` when any element evaluates to non-zero.
  @inlinable func any() -> Tensor { Tensor(_impl.anyAll()) }
  /// Returns a rank-0 boolean tensor that is `true` when every element evaluates to non-zero.
  @inlinable func all() -> Tensor { Tensor(_impl.allAll()) }

  /// Reduce along `dim` with a logical OR, optionally keeping reduced dimensions.
  @inlinable func any(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.anyDim(Int64(dim), keepdim))
  }
  /// Reduce along `dim` with a logical AND, optionally keeping reduced dimensions.
  @inlinable func all(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.allDim(Int64(dim), keepdim))
  }

  /// Return the indices of elements that evaluate to non-zero, with shape `[N, rank]`.
  @inlinable
  func nonzero() -> Tensor { Tensor(_impl.nonzero()) }
}
