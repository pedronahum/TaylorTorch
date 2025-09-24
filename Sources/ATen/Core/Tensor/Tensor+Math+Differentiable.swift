import _Differentiation

@inlinable
internal func _zerosLike(_ tensor: Tensor, dtype: DType? = nil) -> Tensor {
  let resolvedDType = dtype ?? tensor.dtype ?? .float32
  return Tensor.zeros(shape: tensor.shape, dtype: resolvedDType, device: tensor.device)
}

@inlinable
internal func _onesLike(_ tensor: Tensor, dtype: DType? = nil) -> Tensor {
  let resolvedDType = dtype ?? tensor.dtype ?? .float32
  return Tensor.ones(shape: tensor.shape, dtype: resolvedDType, device: tensor.device)
}

@inlinable
internal func _normalizeDimension(_ dim: Int, rank: Int) -> Int {
  let normalized = dim >= 0 ? dim : dim + rank
  precondition(rank > 0, "Dimension out of range")
  precondition(normalized >= 0 && normalized < rank, "Dimension out of range")
  return normalized
}

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
  @derivative(of: negated)
  @inlinable
  internal func _vjpNegated() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    // 'self' is now available implicitly as the base tensor
    return (self.negated(), { v in v.negated() })
  }

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
}

// MARK: - Binary (tensor ⊗ tensor) Differentiation
extension Tensor {

  @derivative(of: subtracting)
  @inlinable
  internal func _vjpSubtracting(_ other: Tensor, alpha: Scalar = .int64(1)) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.subtracting(other)
    return (
      result,
      { v in
        return (v, v.negated())
      }
    )
  }

  @derivative(of: multiplying)
  @inlinable
  internal func _vjpMultiplying(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.multiplying(other)
    return (
      result,
      { v in
        return (v.multiplying(other), v.multiplying(self))
      }
    )
  }
  @derivative(of: dividing)
  @inlinable
  internal func _vjpDividing(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.dividing(other)
    return (
      result,
      { v in
        return (v.dividing(other), v.multiplying(self).dividing(other.multiplying(other)).negated())
      }
    )
  }
}

// MARK: - Binary (tensor ⊗ scalar) Differentiation
extension Tensor {

  @derivative(of: subtracting(_:), wrt: self)
  @inlinable
  internal func _vjpSubtractingScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.subtracting(scalar)
    return (result, { v in v })
  }

  @derivative(of: multiplying(_:), wrt: self)
  @inlinable
  internal func _vjpMultiplyingScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.multiplying(scalar)
    return (result, { v in v.multiplying(scalar) })
  }

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

  @derivative(of: eq)
  @inlinable
  internal func _vjpEq(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.eq(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

  @derivative(of: lt)
  @inlinable
  internal func _vjpLt(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.lt(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

  @derivative(of: le)
  @inlinable
  internal func _vjpLe(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.le(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

  @derivative(of: gt)
  @inlinable
  internal func _vjpGt(_ other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = self.gt(other)
    return (result, { _ in (_zerosLike(self), _zerosLike(other)) })
  }

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

  @derivative(of: eq(_:), wrt: self)
  @inlinable
  internal func _vjpEqScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.eq(scalar)
    return (result, { _ in _zerosLike(self) })
  }

  @derivative(of: lt(_:), wrt: self)
  @inlinable
  internal func _vjpLtScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.lt(scalar)
    return (result, { _ in _zerosLike(self) })
  }

  @derivative(of: le(_:), wrt: self)
  @inlinable
  internal func _vjpLeScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.le(scalar)
    return (result, { _ in _zerosLike(self) })
  }

  @derivative(of: gt(_:), wrt: self)
  @inlinable
  internal func _vjpGtScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = self.gt(scalar)
    return (result, { _ in _zerosLike(self) })
  }

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
