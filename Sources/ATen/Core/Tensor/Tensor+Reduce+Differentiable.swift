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

@inlinable
internal func _argExtremePullback(
  input: Tensor,
  upstream: Tensor,
  indices: Tensor,
  dim: Int,
  keepdim: Bool
) -> Tensor {
  let axis = _normalizeDimension(dim, rank: input.rank)
  let axisSize = input.shape[axis]
  let dtype = _resolveFloatingDType(upstream.dtype, input.dtype)
  let gradOut = upstream.to(dtype: dtype)
  let gradExpanded = keepdim ? gradOut : gradOut.unsqueezed(dim: axis)
  let indicesExpanded = (keepdim ? indices : indices.unsqueezed(dim: axis)).to(dtype: .int64)

  let axisIndices = Tensor.arange(
    0,
    to: axisSize,
    step: 1,
    dtype: .int64,
    device: input.device
  )

  var axisShape = [Int](repeating: 1, count: input.rank)
  axisShape[axis] = axisSize
  let axisIndexTensor = axisIndices.reshaped(axisShape)

  let mask = indicesExpanded.eq(axisIndexTensor).to(dtype: dtype)
  return gradExpanded.multiplying(mask)
}

@inlinable
internal func _scatterSelectedGradient(
  input: Tensor,
  upstream: Tensor,
  indices: Tensor,
  dim: Int,
  axisSize: Int
) -> Tensor {
  let dtype = _resolveFloatingDType(upstream.dtype, input.dtype)
  if axisSize == 0 || input.count == 0 {
    return Tensor.zeros(shape: input.shape, dtype: dtype, device: input.device)
  }

  let axis = _normalizeDimension(dim, rank: input.rank)
  let gradValues = upstream.to(dtype: dtype)
  let perm = _moveDimToEnd(rank: input.rank, dim: axis)
  let inversePerm = _inversePermutation(perm)

  let gradPermuted = gradValues.permuted(perm)
  let indicesPermuted = indices.to(dtype: .int64).permuted(perm)

  let selectedCount = gradPermuted.shape.last ?? 1
  if selectedCount == 0 {
    return Tensor.zeros(shape: input.shape, dtype: dtype, device: input.device)
  }

  let batch = gradPermuted.shape.dropLast().reduce(1, *)
  let grad2D = gradPermuted.reshaped([batch, selectedCount])
  let indices2D = indicesPermuted.reshaped([batch, selectedCount])

  let axisIndices = Tensor.arange(
    0,
    to: axisSize,
    step: 1,
    dtype: .int64,
    device: input.device
  )

  let mask =
    indices2D
    .unsqueezed(dim: 2)
    .eq(axisIndices.reshaped([1, 1, axisSize]))
    .to(dtype: dtype)

  let scattered2D =
    grad2D
    .unsqueezed(dim: 2)
    .to(dtype: dtype)
    .multiplying(mask)
    .sum(dim: 1)

  var restoredShape = gradPermuted.shape
  restoredShape[restoredShape.count - 1] = axisSize
  let restored = scattered2D.reshaped(restoredShape)
  return restored.permuted(inversePerm)
}

extension Tensor {
  /// Reverse-mode derivative for `argmin(dim:keepdim:)`, scattering gradients to
  /// the minima selected along the reduction axis.
  @derivative(of: argmin(dim:keepdim:), wrt: self)
  @inlinable
  internal func _vjpArgmin(dim: Int, keepdim: Bool) -> (
    value: Tensor,
    pullback: (Tensor) -> Tensor
  ) {
    let result = self.argmin(dim: dim, keepdim: keepdim)
    return (
      result,
      { v in
        _argExtremePullback(
          input: self,
          upstream: v,
          indices: result,
          dim: dim,
          keepdim: keepdim
        )
      }
    )
  }

  /// Reverse-mode derivative for `argmax(dim:keepdim:)`, routing gradients to
  /// the maxima selected along the reduction axis.
  @derivative(of: argmax(dim:keepdim:), wrt: self)
  @inlinable
  internal func _vjpArgmax(dim: Int, keepdim: Bool) -> (
    value: Tensor,
    pullback: (Tensor) -> Tensor
  ) {
    let result = self.argmax(dim: dim, keepdim: keepdim)
    return (
      result,
      { v in
        _argExtremePullback(
          input: self,
          upstream: v,
          indices: result,
          dim: dim,
          keepdim: keepdim
        )
      }
    )
  }

  /// Reverse-mode derivative for `sort(dim:descending:)`, undoing the sort by
  /// scattering the upstream value gradients back into their original positions.
  @derivative(of: sort(dim:descending:), wrt: self)
  @inlinable
  internal func _vjpSort(dim: Int, descending: Bool) -> (
    value: TensorPair,
    pullback: (TensorPair.TangentVector) -> Tensor  // This is now valid and means (Tensor) -> Tensor
  ) {
    let pair = self.sort(dim: dim, descending: descending)
    return (
      pair,
      { tangent in  // 'tangent' is now a Tensor
        let axis = _normalizeDimension(dim, rank: self.rank)
        let axisSize = self.shape[axis]
        let gradient = _scatterSelectedGradient(
          input: self,
          upstream: tangent,  // 👈 Use tangent directly
          indices: pair.indices,
          dim: dim,
          axisSize: axisSize
        )
        return gradient
      }
    )
  }

  /// Reverse-mode derivative for `topk(_:dim:largest:sorted:)`, scattering the
  /// upstream value gradients onto the elements that were chosen during ranking.
  @derivative(of: topk(_:dim:largest:sorted:), wrt: self)
  @inlinable
  internal func _vjpTopk(
    _ k: Int,
    dim: Int,
    largest: Bool,
    sorted: Bool
  ) -> (
    value: TensorPair,
    pullback: (TensorPair.TangentVector) -> Tensor  // Also valid here
  ) {
    let pair = self.topk(k, dim: dim, largest: largest, sorted: sorted)
    return (
      pair,
      { tangent in  // 'tangent' is a Tensor
        let axis = _normalizeDimension(dim, rank: self.rank)
        let axisSize = self.shape[axis]
        let gradient = _scatterSelectedGradient(
          input: self,
          upstream: tangent,  // 👈 Use tangent directly
          indices: pair.indices,
          dim: dim,
          axisSize: axisSize
        )
        return gradient
      }
    )
  }

}
