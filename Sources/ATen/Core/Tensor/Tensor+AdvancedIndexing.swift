@preconcurrency import ATenCXX

public extension Tensor {
  /// indexSelect along `dim` with host indices (Int/Int64 accepted).
  @inlinable
  func indexSelect<T: TorchSliceIndex & FixedWidthInteger>(dim: Int, indices: [T]) -> Tensor {
    // Promote indices to Int64 to match c10::Long
    var longs = indices.map { Int64($0) }
    return Tensor(_impl.indexSelect(Int64(dim), &longs, longs.count))
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



