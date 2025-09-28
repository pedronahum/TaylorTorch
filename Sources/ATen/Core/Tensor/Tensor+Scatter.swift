import ATenCXX

extension Tensor {
  /// Adds all values from the `source` tensor into this tensor at the indices
  /// specified in the `index` tensor along a given `dim`. This is an
  /// out-of-place operation.
  /// - Parameters:
  ///   - dim: The axis along which to index.
  ///   - index: The indices of elements to scatter.
  ///   - source: The source tensor to scatter into `self`.
  /// - Returns: A new tensor with the source values added at the specified indices.
  @inlinable
  public func scatterAdd(dim: Int, index: Tensor, source: Tensor) -> Tensor {
    Tensor(_impl.scatterAdd(Int64(dim), index._impl, source._impl))
  }
}
