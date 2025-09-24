@preconcurrency import ATenCXX

public extension Tensor {
  // MARK: Transpose & permute

  /// Returns a view with dimensions `dim0` and `dim1` swapped.
  /// - Parameters:
  ///   - dim0: First dimension to transpose (negatives allowed).
  ///   - dim1: Second dimension to transpose (negatives allowed).
  @inlinable
  func transposed(_ dim0: Int, _ dim1: Int) -> Tensor {
    Tensor(_impl.transpose(Int64(dim0), Int64(dim1)))
  }

  /// Permute axes to `order` (length must equal `rank`). Negatives allowed.
  @inlinable
  func permuted(_ order: [Int]) -> Tensor {
    precondition(order.count == rank, "permute: order.count must equal rank")
    var v = order.map { Int64($0) }
    return Tensor(_impl.permute(&v, v.count))
  }

  // MARK: Reshape & flatten

  /// Reshape to `shape` (uses `at::reshape`: view if possible, else copy).
  @inlinable
  func reshaped(_ shape: [Int]) -> Tensor {
    var s64 = shape.map { Int64($0) }
    return Tensor(_impl.reshape(&s64, s64.count))
  }

  /// Flattens the inclusive range of dimensions from `startDim` to `endDim` (defaults to all dimensions).
  @inlinable
  func flattened(startDim: Int = 0, endDim: Int = -1) -> Tensor {
    Tensor(_impl.flatten(Int64(startDim), Int64(endDim)))
  }

  // MARK: Squeeze / unsqueeze

  /// Removes every dimension whose extent is 1.
  @inlinable
  func squeezed() -> Tensor {
    Tensor(_impl.squeezeAll())
  }

  /// Removes the specified dimension when its extent is 1 (negative dims allowed).
  @inlinable
  func squeezed(dim: Int) -> Tensor {
    Tensor(_impl.squeezeDim(Int64(dim)))
  }

  /// Inserts a singleton dimension at `dim`, accepting negative indices.
  @inlinable
  func unsqueezed(dim: Int) -> Tensor {
    Tensor(_impl.unsqueeze(Int64(dim)))
  }
}
