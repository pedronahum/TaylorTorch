@preconcurrency import ATenCXX

// Prefer canonical end-exclusive semantics in Swift.
internal extension Tensor {
  // This combination is now valid: an internal, inlinable helper.
  @usableFromInline
  @inline(__always)
  /// Returns the length of the specified logical dimension.
  func _dimSize(_ dim: Int) -> Int {
    Int(_impl.sizeAt(Int64(dim)))
  }
}

// MARK: - Core ops
public extension Tensor {
  /// Select a single index along the specified dimension (negative dims/indices allowed).
  /// - Parameters:
  ///   - dim: Dimension to index.
  ///   - index: Element position, evaluated relative to `dim`.
  @inlinable
  func select<T: TorchSliceIndex & FixedWidthInteger>(dim: Int, index: T) -> Tensor {
    Tensor(_impl.select(Int64(dim), Int64(index)))
  }

  /// Return a narrowed view along `dim`, starting at `start` and spanning `length` elements.
  /// - Parameters:
  ///   - dim: Dimension to narrow.
  ///   - start: Starting index for the slice.
  ///   - length: Number of elements to include.
  @inlinable
  func narrow<T: TorchSliceIndex & FixedWidthInteger>(
    dim: Int, start: T, length: T
  ) -> Tensor {
    Tensor(_impl.narrow(Int64(dim), Int64(start), Int64(length)))
  }

  /// Return an end-exclusive slice along `dim`. Only positive `step` values are supported.
  /// - Parameters:
  ///   - dim: Dimension to slice.
  ///   - start: Inclusive lower bound (defaults to `0`).
  ///   - end: Exclusive upper bound. Pass `nil` to extend to the dimension's end.
  ///   - step: Positive stride between elements (defaults to `1`).
  @inlinable
  func slice(
    dim: Int,
    start: Int = 0,
    end: Int? = nil,
    step: Int = 1
  ) -> Tensor {
    let e = end ?? _dimSize(dim)
    return Tensor(_impl.slice(Int64(dim), Int64(start), Int64(e), Int64(step)))
  }
}

// MARK: - Subscripts
public extension Tensor {
  /// Returns the element at `index` from the leading dimension (`dim == 0`).
  @inlinable
  subscript<T: TorchSliceIndex & FixedWidthInteger>(_ index: T) -> Tensor {
    select(dim: 0, index: index)
  }

  /// Returns the element at `index` along the specified dimension.
  @inlinable
  subscript<T: TorchSliceIndex & FixedWidthInteger>(dim dim: Int, _ index: T) -> Tensor {
    select(dim: dim, index: index)
  }

  /// Returns an end-exclusive range slice along the specified dimension.
  @inlinable
  subscript(dim dim: Int, _ range: Range<Int>) -> Tensor {
    slice(dim: dim, start: range.lowerBound, end: range.upperBound, step: 1)
  }

  /// Returns an end-inclusive slice along the specified dimension.
  @inlinable
  subscript(dim dim: Int, _ range: ClosedRange<Int>) -> Tensor {
    slice(dim: dim, start: range.lowerBound, end: range.upperBound &+ 1, step: 1)
  }

  /// Returns a slice from `range.lowerBound` to the end of the dimension.
  @inlinable
  subscript(dim dim: Int, _ range: PartialRangeFrom<Int>) -> Tensor {
    slice(dim: dim, start: range.lowerBound, end: nil, step: 1)
  }

  /// Returns a slice from the start of the dimension up to (but excluding) `range.upperBound`.
  @inlinable
  subscript(dim dim: Int, _ range: PartialRangeUpTo<Int>) -> Tensor {
    slice(dim: dim, start: 0, end: range.upperBound, step: 1)
  }

  /// Returns a slice from the start of the dimension through `range.upperBound` inclusive.
  @inlinable
  subscript(dim dim: Int, _ range: PartialRangeThrough<Int>) -> Tensor {
    slice(dim: dim, start: 0, end: range.upperBound &+ 1, step: 1)
  }
}


public extension Tensor {
  /// Split a tensor into chunks containing at most `size` elements along `dim`.
  /// The final chunk may be smaller than `size`. Results are views into the original tensor.
  /// - Parameters:
  ///   - size: Maximum chunk size (must be > 0).
  ///   - dim: Dimension to split (defaults to `0`).
  func split(size: Int, dim: Int = 0) -> [Tensor] {
    precondition(size > 0, "split size must be > 0")
    let d = dim >= 0 ? dim : dim + rank
    precondition(d >= 0 && d < rank, "dim out of range")
    let total = Int(_impl.sizeAt(Int64(d)))
    var out: [Tensor] = []
    var start = 0
    while start < total {
      let len = Swift.min(size, total - start)
      out.append(self.narrow(dim: d, start: Int64(start), length: Int64(len)))
      start += len
    }
    return out
  }

  /// Split a tensor into a fixed number of equally sized chunks (within one element) along `dim`.
  /// Results are views into the original tensor.
  /// - Parameters:
  ///   - chunks: Target number of chunks (must be > 0).
  ///   - dim: Dimension to split (defaults to `0`).
  func chunk(_ chunks: Int, dim: Int = 0) -> [Tensor] {
    precondition(chunks > 0, "chunks must be > 0")
    let d = dim >= 0 ? dim : dim + rank
    precondition(d >= 0 && d < rank, "dim out of range")
    let total = Int(_impl.sizeAt(Int64(d)))
    let base = total / chunks
    let rem  = total % chunks
    var out: [Tensor] = []
    var start = 0
    for i in 0..<chunks {
      let len = base + (i < rem ? 1 : 0)
      if len == 0 { continue }
      out.append(self.narrow(dim: d, start: Int64(start), length: Int64(len)))
      start += len
    }
    return out
  }
}
