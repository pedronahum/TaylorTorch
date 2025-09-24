// Sources/ATen/Tensor+AxisSugar.swift
public extension Tensor {
  /// Returns a slice that selects the element at `index` along the given logical `axis`.
  @inlinable func select<T: FixedWidthInteger & TorchSliceIndex>(dim axis: Axis, index: T) -> Tensor {
    select(dim: axis.resolve(forRank: rank), index: index)
  }

  /// Narrows `self` along the specified logical `axis`, starting at `start` and keeping `length` elements.
  @inlinable func narrow<T: FixedWidthInteger & TorchSliceIndex>(
    dim axis: Axis, start: T, length: T
  ) -> Tensor {
    narrow(dim: axis.resolve(forRank: rank), start: start, length: length)
  }

  /// Produces a strided slice along the logical `axis`, matching PyTorch slicing semantics.
  @inlinable func slice(dim axis: Axis, start: Int = 0, end: Int? = nil, step: Int = 1) -> Tensor {
    slice(dim: axis.resolve(forRank: rank), start: start, end: end, step: step)
  }

  /// Returns a tensor with the provided axes swapped.
  @inlinable func transposed(_ a: Axis, _ b: Axis) -> Tensor {
    transposed(a.resolve(forRank: rank), b.resolve(forRank: rank))
  }

  /// Reduces the tensor by summing along `axes`, optionally keeping reduced dimensions.
  @inlinable func sum(along axes: [Axis], keepdim: Bool = false) -> Tensor {
    // Reduce one axis at a time; simple 80/20.
    axes.reduce(self) { $0.sum(dim: $1.resolve(forRank: $0.rank), keepdim: keepdim) }
  }

  /// Returns the mean computed along `axes`, optionally preserving reduced dimensions.
  @inlinable func mean(along axes: [Axis], keepdim: Bool = false) -> Tensor {
    axes.reduce(self) { $0.mean(dim: $1.resolve(forRank: $0.rank), keepdim: keepdim) }
  }
}
