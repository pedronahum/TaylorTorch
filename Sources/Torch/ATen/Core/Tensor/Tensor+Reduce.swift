@preconcurrency import ATenCXX
import _Differentiation

/// Convenience wrapper for returning both values and indices from reduction APIs.
public struct TensorPair: Sendable, Differentiable {
  /// Tensor containing the reduction values.
  public var values: Tensor
  /// Tensor containing the indices associated with each value.
  @noDerivative public let indices: Tensor

  @inlinable
  public init(values: Tensor, indices: Tensor) {
    self.values = values
    self.indices = indices
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic {
    public var values: Tensor.TangentVector

    public static var zero: TangentVector {
      .init(values: .zero)
    }

    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(values: lhs.values + rhs.values)
    }

    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(values: lhs.values - rhs.values)
    }
  }

  public mutating func move(by offset: TangentVector) {
    self.values.move(by: offset.values)
  }
}

extension Tensor {
  // ---- Full scalar reductions (rank-0)
  /// Returns the minimum value across all elements.
  @inlinable public func min() -> Tensor { Tensor(_impl.minAll()) }
  /// Returns the maximum value across all elements.
  @inlinable public func max() -> Tensor { Tensor(_impl.maxAll()) }

  // ---- Reductions along a dimension (values + indices)
  /// Returns the minimum value and its index along `dim`.
  @inlinable public func min(dim: Int, keepdim: Bool = false) -> TensorPair {
    let result = _impl.minDimWithIndices(Int64(dim), keepdim)
    // ✅ FIX: Wrap the low-level TTSTensor results in the Swift Tensor type.
    return TensorPair(values: Tensor(result.first), indices: Tensor(result.second))
  }
  /// Returns the maximum value and its index along `dim`.
  @inlinable public func max(dim: Int, keepdim: Bool = false) -> TensorPair {
    let result = _impl.maxDimWithIndices(Int64(dim), keepdim)
    // ✅ FIX: Wrap the low-level TTSTensor results in the Swift Tensor type.
    return TensorPair(values: Tensor(result.first), indices: Tensor(result.second))
  }

  // ---- Argmin/Argmax along a dimension (indices only)
  /// Returns the indices of the minimum values along `dim`.
  @inlinable public func argmin(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.argminDim(Int64(dim), keepdim))
  }
  /// Returns the indices of the maximum values along `dim`.
  @inlinable public func argmax(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.argmaxDim(Int64(dim), keepdim))
  }

  // ---- topk / sort (values + indices)
  /// Returns the top-`k` values and their indices along `dim`.
  @inlinable public func topk(_ k: Int, dim: Int = -1, largest: Bool = true, sorted: Bool = true)
    -> TensorPair
  {
    let result = _impl.topkWithIndices(Int64(k), Int64(dim), largest, sorted)
    // ✅ FIX: Wrap the low-level TTSTensor results in the Swift Tensor type.
    return TensorPair(values: Tensor(result.first), indices: Tensor(result.second))
  }
  /// Returns the sorted values and their indices along `dim`.
  @inlinable public func sort(dim: Int = -1, descending: Bool = false) -> TensorPair {
    let result = _impl.sortDimWithIndices(Int64(dim), descending)
    // ✅ FIX: Wrap the low-level TTSTensor results in the Swift Tensor type.
    return TensorPair(values: Tensor(result.first), indices: Tensor(result.second))
  }

  // ---- Element-wise minimum / maximum (tensor ⊗ tensor)
  /// Returns the element-wise minimum of two tensors.
  @inlinable public func minimum(_ other: Tensor) -> Tensor { Tensor(_impl.minimum(other._impl)) }
  /// Returns the element-wise maximum of two tensors.
  @inlinable public func maximum(_ other: Tensor) -> Tensor { Tensor(_impl.maximum(other._impl)) }
}
