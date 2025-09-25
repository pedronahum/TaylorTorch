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
