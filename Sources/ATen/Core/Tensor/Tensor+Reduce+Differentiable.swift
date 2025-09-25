import _Differentiation

@inlinable
internal func _resolveFloatingDType(_ candidates: DType?...) -> DType {
  for candidate in candidates {
    if let dtype = candidate, dtype.isFloating { return dtype }
  }
  return .float32
}

extension Tensor {
  /// Reverse-mode derivative for the global `min`, sharing gradients equally across tied minima.
  @derivative(of: min)
  @inlinable
  internal func _vjpMinReduce() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.min()
    return (
      result,
      { v in
        let dtype = _resolveFloatingDType(v.dtype, self.dtype)
        let gradOut = v.to(dtype: dtype)
        let mask = (self .== result).to(dtype: dtype)
        let scaledMask = mask.dividing(mask.sum())
        return gradOut.multiplying(scaledMask)
      }
    )
  }

  /// Reverse-mode derivative for the global `max`, sharing gradients equally across tied maxima.
  @derivative(of: max)
  @inlinable
  internal func _vjpMaxReduce() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.max()
    return (
      result,
      { v in
        let dtype = _resolveFloatingDType(v.dtype, self.dtype)
        let gradOut = v.to(dtype: dtype)
        let mask = (self .== result).to(dtype: dtype)
        let scaledMask = mask.dividing(mask.sum())
        return gradOut.multiplying(scaledMask)
      }
    )
  }

  /// Reverse-mode derivative for element-wise `minimum`, routing gradients to the winning operand (splitting ties).
  @derivative(of: minimum)
  @inlinable
  internal func _vjpMinimum(_ other: Tensor) -> (
    value: Tensor,
    pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.minimum(other)
    return (
      result,
      { v in
        let dtype = _resolveFloatingDType(v.dtype, self.dtype, other.dtype)
        let gradOut = v.to(dtype: dtype)
        let lhsStrict = (self .< other).to(dtype: dtype)
        let rhsStrict = (self .> other).to(dtype: dtype)
        let equalMask = (self .== other).to(dtype: dtype)
        let shared = equalMask.dividing(2)
        let lhsMask = lhsStrict.adding(shared)
        let rhsMask = rhsStrict.adding(shared)
        let lhsGrad = _reduceLike(gradOut.multiplying(lhsMask), targetShape: self.shape)
        let rhsGrad = _reduceLike(gradOut.multiplying(rhsMask), targetShape: other.shape)
        return (lhsGrad, rhsGrad)
      }
    )
  }

  /// Reverse-mode derivative for element-wise `maximum`, routing gradients to the dominant operand (splitting ties).
  @derivative(of: maximum)
  @inlinable
  internal func _vjpMaximum(_ other: Tensor) -> (
    value: Tensor,
    pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.maximum(other)
    return (
      result,
      { v in
        let dtype = _resolveFloatingDType(v.dtype, self.dtype, other.dtype)
        let gradOut = v.to(dtype: dtype)
        let lhsStrict = (self .> other).to(dtype: dtype)
        let rhsStrict = (self .< other).to(dtype: dtype)
        let equalMask = (self .== other).to(dtype: dtype)
        let shared = equalMask.dividing(2)
        let lhsMask = lhsStrict.adding(shared)
        let rhsMask = rhsStrict.adding(shared)
        let lhsGrad = _reduceLike(gradOut.multiplying(lhsMask), targetShape: self.shape)
        let rhsGrad = _reduceLike(gradOut.multiplying(rhsMask), targetShape: other.shape)
        return (lhsGrad, rhsGrad)
      }
    )
  }
}
