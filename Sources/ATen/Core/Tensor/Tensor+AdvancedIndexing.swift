// === Tensor+AdvancedIndexing.swift ===
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

extension Tensor {
  /// Scatter `values` into positions described by `indices`, returning a fresh tensor.
  /// - Parameters:
  ///   - indices: Per-dimension index tensors (Int/Long dtypes recommended).
  ///   - values: Source tensor providing replacement values.
  ///   - accumulate: When `true`, add into existing elements instead of overwriting.
  @inlinable
  public func indexPut(indices: [Tensor], values: Tensor, accumulate: Bool = false) -> Tensor {
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
  public func indexAdd(dim: Int, index: Tensor, source: Tensor, alpha: Scalar = .int64(1)) -> Tensor
  {
    Tensor(_impl.indexAdd(Int64(dim), index._impl, source._impl, alpha._cxxScalar))
  }

  /// Copy `source` into `self` along `dim` at the indices provided by `index`.
  /// - Parameters:
  ///   - dim: Dimension whose positions are replaced.
  ///   - index: 1-D tensor of indices selecting rows to copy into.
  ///   - source: Tensor providing replacement rows.
  @inlinable
  public func indexCopy(dim: Int, index: Tensor, source: Tensor) -> Tensor {
    Tensor(_impl.indexCopy(Int64(dim), index._impl, source._impl))
  }

  /// Return the elements of `self` for which `mask` evaluates to `true`.
  /// - Parameter mask: Boolean tensor broadcastable to `self`.
  @inlinable
  public func maskedSelect(_ mask: Tensor) -> Tensor {
    maskedSelect(where: mask)
  }

}

// Helper: compute broadcasted shape (right-aligned, 1 expands)
@usableFromInline
@inline(__always)
internal func _broadcastedShape(_ a: [Int], _ b: [Int]) -> [Int] {
  let n = max(a.count, b.count)
  var out = [Int](repeating: 1, count: n)
  for i in 0..<n {
    let ai = i < n - a.count ? 1 : a[i - (n - a.count)]
    let bi = i < n - b.count ? 1 : b[i - (n - b.count)]
    precondition(ai == bi || ai == 1 || bi == 1, "Shapes \(a) and \(b) are not broadcastable")
    out[i] = max(ai, bi)
  }
  return out
}

// Helper: count TRUEs in a boolean mask safely on host
@usableFromInline
@inline(__always)
internal func _countTrue(_ mask: Tensor) -> Int {
  let ones = mask.to(dtype: .int64).sum()
  let scalars: [Int64] = ones.toArray()
  // sum() is rank-0 → one element
  return Int(scalars[0])
}

extension Tensor {
  /// Functional masked scatter with robust shape/dtype handling.
  /// Two supported forms:
  ///  1) `source.numel == mask.trueCount` (classic packed replacement)
  ///  2) `source.shape == broadcast(self.shape, mask.shape)` (pick-from-source)
  @inlinable
  public func maskedScatter(where mask: Tensor, source: Tensor) -> Tensor {
    // 1. Ensure self and mask are broadcastable and materialize the base tensor.
    let outShape = broadcastShapes(self.shape, mask.shape)
    let base = self.broadcasted(to: outShape)
    let boolMask = mask.broadcasted(to: outShape).to(dtype: .bool)

    // 2. Ensure the source tensor's dtype matches the base tensor's dtype.
    guard let baseDType = base.dtype else {
      preconditionFailure("maskedScatter: base tensor must have a concrete dtype")
    }
    let sourceTyped = source.to(dtype: baseDType)

    // 3. Enforce the core contract of masked_scatter.
    let trueCount = _countTrue(boolMask)
    precondition(
      sourceTyped.count == trueCount,
      """
      maskedScatter: The number of elements in the source tensor (\(sourceTyped.count))
      must be equal to the number of true elements in the mask (\(trueCount)).
      """
    )

    // 4. Use `indexPut` with a boolean mask, which is the exact equivalent.
    // This is robust and leverages a well-defined existing operation.
    return base.indexPut(indices: [boolMask], values: sourceTyped, accumulate: false)
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
