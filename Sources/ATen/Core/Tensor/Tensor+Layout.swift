@preconcurrency import ATenCXX

public extension Tensor {
  /// Total number of elements contained in the tensor.
  var count: Int { Int(_impl.numel()) }

  /// Physical strides (in elements) for each dimension of the tensor.
  var strides: [Int] {
    (0..<rank).map { Int(_impl.strideAt(Int64($0))) }
  }

  /// Indicates whether the tensor's storage is physically contiguous in memory.
  var isContiguous: Bool { _impl.isContiguous() }

  /// Return a contiguous copy or view of the tensor.
  func contiguous() -> Tensor { Tensor(_impl.contiguous()) }
}
