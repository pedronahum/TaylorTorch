import _Differentiation

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
