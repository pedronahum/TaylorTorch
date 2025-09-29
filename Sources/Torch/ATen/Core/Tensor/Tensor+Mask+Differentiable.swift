// === Tensor+Mask+Differentiable.swift ===
import _Differentiation

@usableFromInline
@inline(__always)
internal func _materializeMask(_ mask: Tensor, targetShape: [Int]) -> Tensor {
  if mask.shape == targetShape { return mask }
  return mask.broadcasted(to: targetShape)
}

@usableFromInline
@inline(__always)
internal func _maskNumericAndComplement(
  mask: Tensor,
  targetShape: [Int],
  dtype: DType
) -> (Tensor, Tensor) {
  let materialized = _materializeMask(mask, targetShape: targetShape)
  let numericMask = materialized.to(dtype: dtype)
  let complement = numericMask.negated().adding(1)
  return (numericMask, complement)
}

extension Tensor {
  /// Reverse-mode derivative for `maskedFill(where:with:)` (scalar variant),
  /// propagating gradients only through the elements where the mask is `false`.
  @derivative(of: maskedFill(where:with:), wrt: self)
  @inlinable
  internal func _vjpMaskedFillScalar<T: TorchArithmetic>(
    where mask: Tensor,
    with value: T
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = maskedFill(where: mask, with: value)
    return (
      result,
      { v in
        let dtype = _resolveFloatingDType(v.dtype, self.dtype)
        let gradOut = v.to(dtype: dtype)
        let (_, keepMask) = _maskNumericAndComplement(
          mask: mask, targetShape: self.shape, dtype: dtype)
        let gradSelf = gradOut.multiplying(keepMask)
        return _reduceLike(gradSelf, targetShape: self.shape)
      }
    )
  }

  /// Reverse-mode derivative for tensor `maskedFill(where:with:)`, splitting
  /// gradients between the original tensor (`mask == false`) and replacement
  /// values (`mask == true`).
  @derivative(of: maskedFill(where:with:), wrt: (self, values))
  @inlinable
  internal func _vjpMaskedFillTensor(
    where mask: Tensor,
    with values: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    let result = maskedFill(where: mask, with: values)
    return (
      result,
      { v in
        let dtype = _resolveFloatingDType(v.dtype, self.dtype, values.dtype)
        let gradOut = v.to(dtype: dtype)
        let (maskNumeric, keepMask) = _maskNumericAndComplement(
          mask: mask, targetShape: self.shape, dtype: dtype)
        let gradSelf = _reduceLike(gradOut.multiplying(keepMask), targetShape: self.shape)
        let gradValues = _reduceLike(gradOut.multiplying(maskNumeric), targetShape: values.shape)
        return (gradSelf, gradValues)
      }
    )
  }
}

extension Tensor {
  /// Reverse-mode derivative for `maskedScatter(where:source:)`.
  ///
  /// This pullback splits the incoming gradient `v` into two parts:
  /// 1. `gradSelf`: Gradients from `v` where the mask was `false`.
  /// 2. `gradSource`: Gradients from `v` gathered from where the mask was `true`.
  @derivative(of: maskedScatter(where:source:), wrt: (self, source))
  @inlinable
  internal func _vjpMaskedScatter(
    where mask: Tensor,
    source: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    // 1. Perform the forward pass to get the result.
    let result = maskedScatter(where: mask, source: source)

    // 2. Define the pullback function.
    return (
      result,
      { v in
        // Determine the broadcasted shape and the correct floating-point dtype.
        let outShape = broadcastShapes(self.shape, mask.shape)
        let dtype = _resolveFloatingDType(v.dtype, self.dtype, source.dtype)
        let gradOut = v.to(dtype: dtype)

        // Create numeric versions of the mask and its complement (keepMask).
        let (maskNumeric, keepMask) = _maskNumericAndComplement(
          mask: mask, targetShape: outShape, dtype: dtype)

        // ---- Gradient for `self` ----
        // The gradient flows back to `self` only where its original values were kept.
        // This is identical to the logic in `_vjpMaskedFillTensor`.
        let gradSelf = _reduceLike(gradOut.multiplying(keepMask), targetShape: self.shape)

        // ---- Gradient for `source` ----
        // The gradient for the source is formed by "gathering" the gradients from the
        // output where the mask was true. `maskedSelect` performs this gathering.
        let boolMask = maskNumeric.to(dtype: .bool)
        let gatheredGrads = gradOut.maskedSelect(where: boolMask)

        // The gathered gradients are a flat tensor, so we reshape them to match the
        // original source tensor's shape.
        let gradSource = _reduceLike(gatheredGrads, targetShape: source.shape)

        return (gradSelf, gradSource)
      }
    )
  }
}

extension Tensor {
  // Pullback for maskedSelect: scatter the upstream into the selected positions.
  @derivative(of: maskedSelect(_:), wrt: self)
  @inlinable
  public func _vjpMaskedSelect2(_ mask: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Tensor)
  {
    let y = maskedSelect(mask)
    return (
      y,
      { v in
        let dtype = _resolveFloatingDType(v.dtype, self.dtype)
        // Materialize the (possibly broadcastable) mask to self’s shape
        let matMask = _materializeMask(mask, targetShape: self.shape)
        let zeros = Tensor.zeros(shape: self.shape, dtype: dtype, device: self.device)
        return zeros.maskedScatter(where: matMask, source: v.to(dtype: dtype))
      }
    )
  }
}
