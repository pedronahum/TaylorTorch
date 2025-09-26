@preconcurrency import ATenCXX

extension Tensor {
  /// indexSelect along `dim` with host indices (Int/Int64 accepted).
  @inlinable
  public func indexSelect<T: TorchSliceIndex & FixedWidthInteger>(dim: Int, indices: [T]) -> Tensor
  {
    // Promote indices to Int64 to match c10::Long
    var longs = indices.map { Int64($0) }
    return Tensor(_impl.indexSelect(Int64(dim), &longs, longs.count))
  }
}

public extension Tensor {
  /// Scatter `values` into positions described by `indices`, returning a fresh tensor.
  /// - Parameters:
  ///   - indices: Per-dimension index tensors (Int/Long dtypes recommended).
  ///   - values: Source tensor providing replacement values.
  ///   - accumulate: When `true`, add into existing elements instead of overwriting.
  @inlinable
  func indexPut(indices: [Tensor], values: Tensor, accumulate: Bool = false) -> Tensor {
    precondition(!indices.isEmpty, "indexPut: indices cannot be empty")
    let impls = indices.map { $0._impl }
    return impls.withUnsafeBufferPointer { buffer in
      precondition(buffer.baseAddress != nil)
      return Tensor(_impl.indexPut(buffer.baseAddress!, buffer.count, values._impl, accumulate))
    }
  }

  /// Add `source` into `self` along `dim` at locations specified by `index`.
  /// - Parameters:
  ///   - dim: Dimension along which to scatter.
  ///   - index: 1-D tensor containing the indices to update.
  ///   - source: Tensor to be added at the indexed positions.
  ///   - alpha: Optional scaling applied to `source` before accumulation.
  @inlinable
  func indexAdd(dim: Int, index: Tensor, source: Tensor, alpha: Scalar = .int64(1)) -> Tensor {
    Tensor(_impl.indexAdd(Int64(dim), index._impl, source._impl, alpha._cxxScalar))
  }

  /// Copy `source` into `self` along `dim` at the indices provided by `index`.
  /// - Parameters:
  ///   - dim: Dimension whose positions are replaced.
  ///   - index: 1-D tensor of indices selecting rows to copy into.
  ///   - source: Tensor providing replacement rows.
  @inlinable
  func indexCopy(dim: Int, index: Tensor, source: Tensor) -> Tensor {
    Tensor(_impl.indexCopy(Int64(dim), index._impl, source._impl))
  }

  /// Return the elements of `self` for which `mask` evaluates to `true`.
  /// - Parameter mask: Boolean tensor broadcastable to `self`.
  @inlinable
  func maskedSelect(_ mask: Tensor) -> Tensor {
    maskedSelect(where: mask)
  }
}

/*
// 2-D sugar: t[row, col] ≡ t.select(dim: 0, index: row).select(dim: 0, index: col)
public extension Tensor {
  /// Convenience 2-D accessor equivalent to consecutive `select` calls on the leading dimension.
  /// - Parameters:
  ///   - row: Row index evaluated against dimension 0 (negative indices allowed).
  ///   - col: Column index evaluated against dimension 1 (negative indices allowed).
  @inlinable
  // 'TorchSliceIndex' constraint to both Row and Col
  subscript<Row: TorchSliceIndex & FixedWidthInteger, Col: TorchSliceIndex & FixedWidthInteger>(
    _ row: Row, _ col: Col
  ) -> Tensor {
    self.select(dim: 0, index: row).select(dim: 0, index: col)
  }
}*/
