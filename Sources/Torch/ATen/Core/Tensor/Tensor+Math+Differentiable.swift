import _Differentiation

/// Materialises an all-zero tensor matching the provided tensor's shape,
/// defaulting to the same dtype/device unless explicitly overridden.
@inlinable
internal func _zerosLike(_ tensor: Tensor, dtype: DType? = nil) -> Tensor {
  let resolvedDType = dtype ?? tensor.dtype ?? .float32
  return Tensor.zeros(shape: tensor.shape, dtype: resolvedDType, device: tensor.device)
}

/// Materialises an all-one tensor matching the provided tensor's shape,
/// defaulting to the same dtype/device unless explicitly overridden.
@inlinable
internal func _onesLike(_ tensor: Tensor, dtype: DType? = nil) -> Tensor {
  let resolvedDType = dtype ?? tensor.dtype ?? .float32
  return Tensor.ones(shape: tensor.shape, dtype: resolvedDType, device: tensor.device)
}

/// Normalises a dimension index, supporting negative offsets and validating that
/// the resulting value falls within `[0, rank)`.
@inlinable
internal func _normalizeDimension(_ dim: Int, rank: Int) -> Int {
  let normalized = dim >= 0 ? dim : dim + rank
  precondition(rank > 0, "Dimension out of range")
  precondition(normalized >= 0 && normalized < rank, "Dimension out of range")
  return normalized
}

/// Reduces a gradient tensor so its shape matches `targetShape`, summing across
/// extra or broadcasted axes when necessary.
@inlinable
internal func _reduceLike(_ gradient: Tensor, targetShape: [Int]) -> Tensor {
  if gradient.shape == targetShape { return gradient }

  var result = gradient
  if targetShape.isEmpty {
    while result.rank > 0 {
      result = result.sum(dim: 0)
    }
    return result
  }

  var shape = result.shape
  if shape.count < targetShape.count {
    let missing = targetShape.count - shape.count
    for _ in 0..<missing {
      result = result.unsqueezed(dim: 0)
    }
    shape = result.shape
  }
  let extraDims = max(0, shape.count - targetShape.count)
  if extraDims > 0 {
    for dim in 0..<extraDims {
      result = result.sum(dim: dim, keepdim: true)
    }
    shape = result.shape
  }

  if !targetShape.isEmpty {
    for index in targetShape.indices {
      let dim = index + extraDims
      if shape[dim] != 1 && targetShape[index] == 1 {
        result = result.sum(dim: dim, keepdim: true)
        shape = result.shape
      }
    }
  }

  return result.reshaped(targetShape)
}

// Wrap all derivative functions in an extension of Tensor
extension Tensor {
  /// Reverse-mode derivative for `negated`, flipping the sign of the upstream
  /// gradient to reflect the unary negation.
  @derivative(of: negated)
  @inlinable
  internal func _vjpNegated() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    // 'self' is now available implicitly as the base tensor
    return (self.negated(), { v in v.negated() })
  }

  /// Reverse-mode derivative for `abs`, propagating the upstream gradient
  /// multiplied by the sign of the original input.
  @derivative(of: abs)
  @inlinable
  internal func _vjpAbs() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.abs()
    return (
      result,
      { v in
        let sign = self.dividing(result.adding(1e-12))
        return v.multiplying(sign)
      }
    )
  }

  /// Reverse-mode derivative for `relu`, masking out negative inputs while
  /// forwarding positive gradients unchanged.
  @derivative(of: relu)
  @inlinable
  internal func _vjpRelu() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.relu()
    return (
      result,
      { v in
        let mask = self .> 0
        return v.multiplying(mask.to(dtype: v.dtype!))
      }
    )
  }

  /// Reverse-mode derivative for `exp`, scaling the gradient by the forward
  /// activation to mirror the analytic derivative.
  @derivative(of: exp)
  @inlinable
  internal func _vjpExp() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.exp()
    return (
      result,
      { v in
        return v.multiplying(result)
      }
    )
  }

  /// Reverse-mode derivative for `log`, dividing the upstream gradient by the
  /// original input.
  @derivative(of: log)
  @inlinable
  internal func _vjpLog() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (
      self.log(),
      { v in
        return v.dividing(self)
      }
    )
  }

  /// Reverse-mode derivative for `sqrt`, scaling the gradient by `1 / (2 * sqrt(x))`.
  @derivative(of: sqrt)
  @inlinable
  internal func _vjpSqrt() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.sqrt()
    return (
      result,
      { v in
        return v.dividing(result.multiplying(2))
      }
    )
  }

  /// Reverse-mode derivative for `tanh`, scaling by `1 - tanh(x)^2`.
  @derivative(of: tanh)
  @inlinable
  internal func _vjpTanh() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.tanh()
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let gradFactor = _onesLike(self, dtype: dtype).subtracting(result.multiplying(result))
        return v.multiplying(gradFactor)
      }
    )
  }

  /// Reverse-mode derivative for `sigmoid`, scaling by `sigmoid(x) * (1 - sigmoid(x))`.
  @derivative(of: sigmoid)
  @inlinable
  internal func _vjpSigmoid() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.sigmoid()
    return (
      result,
      { v in
        let complement = result.negated().adding(1)
        return v.multiplying(result).multiplying(complement)
      }
    )
  }

  /// Reverse-mode derivative for `sin`, multiplying by `cos(x)`.
  @derivative(of: sin)
  @inlinable
  internal func _vjpSin() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.sin()
    let cosValue = self.cos()
    return (
      result,
      { v in v.multiplying(cosValue) }
    )
  }

  /// Reverse-mode derivative for `cos`, multiplying by `-sin(x)`.
  @derivative(of: cos)
  @inlinable
  internal func _vjpCos() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.cos()
    let sinValue = self.sin()
    return (
      result,
      { v in v.multiplying(sinValue).negated() }
    )
  }

  /// Reverse-mode derivative for `tan`, scaling by `1 + tan(x)^2`.
  @derivative(of: tan)
  @inlinable
  internal func _vjpTan() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.tan()
    return (
      result,
      { v in
        let gradFactor = result.multiplying(result).adding(1)
        return v.multiplying(gradFactor)
      }
    )
  }

  /// Reverse-mode derivative for `asin`, dividing by `sqrt(1 - x^2)`.
  @derivative(of: asin)
  @inlinable
  internal func _vjpAsin() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.asin()
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let denom = _onesLike(self, dtype: dtype)
          .subtracting(self.multiplying(self))
          .sqrt()
        return v.dividing(denom)
      }
    )
  }

  /// Reverse-mode derivative for `acos`, dividing by `-sqrt(1 - x^2)`.
  @derivative(of: acos)
  @inlinable
  internal func _vjpAcos() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.acos()
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let denom = _onesLike(self, dtype: dtype)
          .subtracting(self.multiplying(self))
          .sqrt()
        return v.dividing(denom).negated()
      }
    )
  }

  /// Reverse-mode derivative for `atan`, dividing by `1 + x^2`.
  @derivative(of: atan)
  @inlinable
  internal func _vjpAtan() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.atan()
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let denom = _onesLike(self, dtype: dtype)
          .adding(self.multiplying(self))
        return v.dividing(denom)
      }
    )
  }

  /// Reverse-mode derivative for `sinh`, multiplying by `cosh(x)`.
  @derivative(of: sinh)
  @inlinable
  internal func _vjpSinh() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.sinh()
    let coshValue = self.cosh()
    return (
      result,
      { v in v.multiplying(coshValue) }
    )
  }

  /// Reverse-mode derivative for `cosh`, multiplying by `sinh(x)`.
  @derivative(of: cosh)
  @inlinable
  internal func _vjpCosh() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.cosh()
    let sinhValue = self.sinh()
    return (
      result,
      { v in v.multiplying(sinhValue) }
    )
  }

  /// Reverse-mode derivative for `asinh`, dividing by `sqrt(1 + x^2)`.
  @derivative(of: asinh)
  @inlinable
  internal func _vjpAsinh() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.asinh()
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let denom = _onesLike(self, dtype: dtype)
          .adding(self.multiplying(self))
          .sqrt()
        return v.dividing(denom)
      }
    )
  }

  /// Reverse-mode derivative for `acosh`, dividing by `sqrt(x^2 - 1)`.
  @derivative(of: acosh)
  @inlinable
  internal func _vjpAcosh() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.acosh()
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let denom = self.multiplying(self)
          .subtracting(_onesLike(self, dtype: dtype))
          .sqrt()
        return v.dividing(denom)
      }
    )
  }

  /// Reverse-mode derivative for `atanh`, dividing by `1 - x^2`.
  @derivative(of: atanh)
  @inlinable
  internal func _vjpAtanh() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.atanh()
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let denom = _onesLike(self, dtype: dtype)
          .subtracting(self.multiplying(self))
        return v.dividing(denom)
      }
    )
  }

  /// Reverse-mode derivative for `erf`, scaling by `(2/√π) * exp(-x^2)`.
  @derivative(of: erf)
  @inlinable
  internal func _vjpErf() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.erf()
    return (
      result,
      { v in
        let expTerm = self.multiplying(self).negated().exp()
        let factor = 2.0 / Double.pi.squareRoot()
        return v.multiplying(factor).multiplying(expTerm)
      }
    )
  }

  /// Reverse-mode derivative for `erfc`, scaling by `-(2/√π) * exp(-x^2)`.
  @derivative(of: erfc)
  @inlinable
  internal func _vjpErfc() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.erfc()
    return (
      result,
      { v in
        let expTerm = self.multiplying(self).negated().exp()
        let factor = -2.0 / Double.pi.squareRoot()
        return v.multiplying(factor).multiplying(expTerm)
      }
    )
  }
}

// MARK: - Binary (tensor ⊗ tensor) Differentiation
extension Tensor {

  /// Reverse-mode derivative for tensor–tensor `subtracting`, passing the
  /// upstream gradient to the minuend and its negation to the subtrahend.
  // subtracting (tensor ⊗ tensor)
  @derivative(of: subtracting)
  @inlinable
  internal func _vjpSubtracting(_ other: Tensor, alpha: Scalar = .int64(1)) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.subtracting(other, alpha: alpha)
    return (
      result,
      { v in
        let dSelf = _reduceLike(v, targetShape: self.shape)
        let dOther = _reduceLike(v.negated(), targetShape: other.shape)
        return (dSelf, dOther)
      }
    )
  }

  // multiplying (tensor ⊗ tensor)
  @derivative(of: multiplying)
  @inlinable
  internal func _vjpMultiplying(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.multiplying(other)
    return (
      result,
      { v in
        let dSelf = _reduceLike(v.multiplying(other), targetShape: self.shape)
        let dOther = _reduceLike(v.multiplying(self), targetShape: other.shape)
        return (dSelf, dOther)
      }
    )
  }

  // dividing (tensor ⊗ tensor)
  @derivative(of: dividing)
  @inlinable
  internal func _vjpDividing(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.dividing(other)
    return (
      result,
      { v in
        let dSelf = _reduceLike(v.dividing(other), targetShape: self.shape)
        let rhsSq = other.multiplying(other)
        let num = v.multiplying(self)
        let dOther = _reduceLike(num.dividing(rhsSq).negated(), targetShape: other.shape)
        return (dSelf, dOther)
      }
    )
  }
}

// MARK: - Binary (tensor ⊗ scalar) Differentiation
extension Tensor {

  /// Reverse-mode derivative for tensor–scalar `subtracting`, returning the
  /// upstream gradient unchanged for the tensor operand.
  @derivative(of: subtracting(_:), wrt: self)
  @inlinable
  internal func _vjpSubtractingScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.subtracting(scalar)
    return (result, { v in v })
  }

  /// Reverse-mode derivative for tensor–scalar `multiplying`, scaling the
  /// gradient by the scalar factor.
  @derivative(of: multiplying(_:), wrt: self)
  @inlinable
  internal func _vjpMultiplyingScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.multiplying(scalar)
    return (result, { v in v.multiplying(scalar) })
  }

  /// Reverse-mode derivative for tensor–scalar `dividing`, dividing the
  /// gradient by the scalar factor.
  @derivative(of: dividing(_:), wrt: self)
  @inlinable
  internal func _vjpDividingScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.dividing(scalar)
    return (result, { v in v.dividing(scalar) })
  }
}

// MARK: - Power Differentiation
extension Tensor {

  /// Reverse-mode derivative for tensor–scalar `pow`, returning `power * x^(power-1)`
  /// scaled by the upstream gradient (with zero when `power == 0`).
  @derivative(of: pow(_:), wrt: self)
  @inlinable
  internal func _vjpPowScalar<T: TorchArithmetic>(_ power: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.pow(power)
    return (
      result,
      { v in
        if power == 0 {
          return _zerosLike(self, dtype: v.dtype ?? self.dtype)
        }
        let baseFactor = self.pow(power - 1)
        return v.multiplying(power).multiplying(baseFactor)
      }
    )
  }

  /// Reverse-mode derivative for tensor–tensor `pow`, applying the chain rule
  /// to both base and exponent while reducing broadcasted gradients.
  @derivative(of: pow(_:))
  @inlinable
  internal func _vjpPowTensor(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.pow(other)
    return (
      result,
      { v in
        var gradSelf = v.multiplying(other).multiplying(self.pow(other.subtracting(1)))
        var gradOther = v.multiplying(result.multiplying(self.log()))
        gradSelf = _reduceLike(gradSelf, targetShape: self.shape)
        gradOther = _reduceLike(gradOther, targetShape: other.shape)
        return (gradSelf, gradOther)
      }
    )
  }
}

// MARK: - Clamp Differentiation
extension Tensor {

  /// Reverse-mode derivative for `clamp(min:max:)`, masking gradients to zero
  /// where the input is saturated at the bounds.
  @derivative(of: clamp(min:max:), wrt: self)
  @inlinable
  internal func _vjpClamp<T: TorchArithmetic>(min: T, max: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.clamp(min: min, max: max)
    return (
      result,
      { v in
        let dtype = v.dtype ?? self.dtype ?? .float32
        let lowerMask = (self .> min).to(dtype: dtype)
        let upperMask = (self .< max).to(dtype: dtype)
        let mask = lowerMask.multiplying(upperMask)
        return v.multiplying(mask)
      }
    )
  }
}

// MARK: - Reductions Differentiation
extension Tensor {

  /// Reverse-mode derivative for scalar `sum`, broadcasting the upstream value
  /// across all elements of the input tensor.
  @derivative(of: sum)
  @inlinable
  internal func _vjpSum() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.sum()
    return (
      result,
      { v in
        let ones = _onesLike(self, dtype: v.dtype ?? self.dtype)
        return v.multiplying(ones)
      }
    )
  }

  /// Reverse-mode derivative for dimensional `sum`, expanding (and unsqueezing
  /// when needed) the upstream gradient to match the input tensor's shape.
  @derivative(of: sum(dim:keepdim:), wrt: self)
  @inlinable
  internal func _vjpSum(dim: Int, keepdim: Bool) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.sum(dim: dim, keepdim: keepdim)
    return (
      result,
      { v in
        let normalizedDim = _normalizeDimension(dim, rank: self.rank)
        var grad = keepdim ? v : v.unsqueezed(dim: normalizedDim)
        grad = grad.expanded(to: self.shape)
        return grad
      }
    )
  }

  /// Reverse-mode derivative for scalar `mean`, distributing the gradient evenly
  /// across every element of the input tensor.
  @derivative(of: mean)
  @inlinable
  internal func _vjpMean() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = self.mean()
    return (
      result,
      { v in
        let ones = _onesLike(self, dtype: v.dtype ?? self.dtype)
        let count = Swift.max(1, self.shape.reduce(1, *))
        return v.multiplying(ones).dividing(Double(count))
      }
    )
  }

  /// Reverse-mode derivative for dimensional `mean`, expanding the upstream
  /// gradient and dividing by the number of elements reduced along the axis.
  @derivative(of: mean(dim:keepdim:), wrt: self)
  @inlinable
  internal func _vjpMean(dim: Int, keepdim: Bool) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.mean(dim: dim, keepdim: keepdim)
    return (
      result,
      { v in
        let normalizedDim = _normalizeDimension(dim, rank: self.rank)
        var grad = keepdim ? v : v.unsqueezed(dim: normalizedDim)
        grad = grad.expanded(to: self.shape)
        let divisor = Double(self.shape[normalizedDim])
        return grad.dividing(divisor)
      }
    )
  }
}

// MARK: - Linalg Differentiation
extension Tensor {

  /// Reverse-mode derivative for `matmul`, handling vector/matrix combinations
  /// and reducing broadcasted gradients back to each operand's shape.
  @derivative(of: matmul)
  @inlinable
  internal func _vjpMatmul(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let lhsShape = self.shape
    let rhsShape = other.shape
    let result = self.matmul(other)
    return (
      result,
      { v in
        if self.rank == 1 && other.rank == 1 {
          let gradSelf = _reduceLike(v.multiplying(other), targetShape: lhsShape)
          let gradOther = _reduceLike(v.multiplying(self), targetShape: rhsShape)
          return (gradSelf, gradOther)
        }

        var lhs = self
        var rhs = other
        var grad = v
        var lhsWasVector = false
        var rhsWasVector = false

        if lhs.rank == 1 {
          lhs = lhs.unsqueezed(dim: 0)
          lhsWasVector = true
        }
        if rhs.rank == 1 {
          rhs = rhs.unsqueezed(dim: -1)
          rhsWasVector = true
        }
        if grad.rank == 1 {
          if rhsWasVector {
            grad = grad.unsqueezed(dim: -1)
          } else if lhsWasVector {
            grad = grad.unsqueezed(dim: 0)
          }
        }

        var gradSelf = grad.matmul(rhs.transposed(-1, -2))
        var gradOther = lhs.transposed(-1, -2).matmul(grad)

        if lhsWasVector {
          gradSelf = gradSelf.squeezed(dim: 0)
        }
        if rhsWasVector {
          gradOther = gradOther.squeezed(dim: -1)
        }

        gradSelf = _reduceLike(gradSelf, targetShape: lhsShape)
        gradOther = _reduceLike(gradOther, targetShape: rhsShape)
        return (gradSelf, gradOther)
      }
    )
  }

  /// Reverse-mode derivative for `dot`, distributing the upstream scalar along
  /// each operand scaled by the other vector.
  @derivative(of: dot)
  @inlinable
  internal func _vjpDot(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.dot(other)
    return (
      result,
      { v in
        let gradSelf = _reduceLike(v.multiplying(other), targetShape: self.shape)
        let gradOther = _reduceLike(v.multiplying(self), targetShape: other.shape)
        return (gradSelf, gradOther)
      }
    )
  }
}

// MARK: - Comparisons (tensor ⊗ tensor) Differentiation
extension Tensor {

  /// Reverse-mode derivative for element-wise equality, which is zero-valued
  /// because the operation is non-differentiable.
  @derivative(of: eq)
  @inlinable
  internal func _vjpEq(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.eq(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

  /// Reverse-mode derivative for element-wise less-than, returning zeros since
  /// the comparison is non-differentiable.
  @derivative(of: lt)
  @inlinable
  internal func _vjpLt(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.lt(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

  /// Reverse-mode derivative for element-wise less-than-or-equal, returning
  /// zeros because the comparison is non-differentiable.
  @derivative(of: le)
  @inlinable
  internal func _vjpLe(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.le(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

  /// Reverse-mode derivative for element-wise greater-than, returning zeros
  /// since comparison results carry no gradient.
  @derivative(of: gt)
  @inlinable
  internal func _vjpGt(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.gt(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

  /// Reverse-mode derivative for element-wise greater-than-or-equal, returning
  /// zeros because the comparison is non-differentiable.
  @derivative(of: ge)
  @inlinable
  internal func _vjpGe(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.ge(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }
}

// MARK: - Comparisons (tensor ⊗ scalar) Differentiation
extension Tensor {

  /// Reverse-mode derivative for tensor–scalar equality, returning zeros since
  /// the predicate carries no gradient.
  @derivative(of: eq(_:), wrt: self)
  @inlinable
  internal func _vjpEqScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.eq(scalar)
    return (result, { _ in _zerosLike(self) })
  }

  /// Reverse-mode derivative for tensor–scalar less-than, returning zeros.
  @derivative(of: lt(_:), wrt: self)
  @inlinable
  internal func _vjpLtScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.lt(scalar)
    return (result, { _ in _zerosLike(self) })
  }

  /// Reverse-mode derivative for tensor–scalar less-than-or-equal, returning
  /// zeros.
  @derivative(of: le(_:), wrt: self)
  @inlinable
  internal func _vjpLeScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.le(scalar)
    return (result, { _ in _zerosLike(self) })
  }

  /// Reverse-mode derivative for tensor–scalar greater-than, returning zeros.
  @derivative(of: gt(_:), wrt: self)
  @inlinable
  internal func _vjpGtScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.gt(scalar)
    return (result, { _ in _zerosLike(self) })
  }

  /// Reverse-mode derivative for tensor–scalar greater-than-or-equal, returning
  /// zeros.
  @derivative(of: ge(_:), wrt: self)
  @inlinable
  internal func _vjpGeScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.ge(scalar)
    return (result, { _ in _zerosLike(self) })
  }
}

// MARK: - Where (ternary) Differentiation
extension TorchWhere {

  /// Reverse-mode derivative for `TorchWhere.select`, routing gradients through
  /// either branch based on the boolean condition while respecting broadcasting.
  @derivative(of: select(condition:_:_:), wrt: (a, b))
  @inlinable
  internal static func _vjpSelect(
    condition: Tensor,
    _ a: Tensor,
    _ b: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    let result = select(condition: condition, a, b)
    return (
      result,
      { v in
        let dtype = v.dtype ?? a.dtype ?? b.dtype ?? .float32
        let mask = condition.to(dtype: dtype)
        let ones = _onesLike(mask, dtype: dtype)
        let inverseMask = ones.subtracting(mask)
        let gradA = _reduceLike(v.multiplying(mask), targetShape: a.shape)
        let gradB = _reduceLike(v.multiplying(inverseMask), targetShape: b.shape)
        return (gradA, gradB)
      }
    )
  }
}
