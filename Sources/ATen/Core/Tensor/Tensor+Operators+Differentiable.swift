import _Differentiation

// MARK: - Tensor ⊗ Tensor Operators
@derivative(of: +)
@inlinable
internal func _vjpAddTensors(_ lhs: Tensor, _ rhs: Tensor) -> (
  value: Tensor,
  pullback: (Tensor) -> (Tensor, Tensor)
) {
  let result = lhs.adding(rhs)
  return (
    result,
    { v in
      let gradLhs = _reduceLike(v, targetShape: lhs.shape)
      let gradRhs = _reduceLike(v, targetShape: rhs.shape)
      return (gradLhs, gradRhs)
    }
  )
}

@derivative(of: -)
@inlinable
internal func _vjpSubtractTensors(_ lhs: Tensor, _ rhs: Tensor) -> (
  value: Tensor,
  pullback: (Tensor) -> (Tensor, Tensor)
) {
  let result = lhs.subtracting(rhs)
  return (
    result,
    { v in
      let gradLhs = _reduceLike(v, targetShape: lhs.shape)
      let gradRhs = _reduceLike(v.negated(), targetShape: rhs.shape)
      return (gradLhs, gradRhs)
    }
  )
}

@derivative(of: *)
@inlinable
internal func _vjpMultiplyTensors(_ lhs: Tensor, _ rhs: Tensor) -> (
  value: Tensor,
  pullback: (Tensor) -> (Tensor, Tensor)
) {
  let result = lhs.multiplying(rhs)
  return (
    result,
    { v in
      let gradLhs = _reduceLike(v.multiplying(rhs), targetShape: lhs.shape)
      let gradRhs = _reduceLike(v.multiplying(lhs), targetShape: rhs.shape)
      return (gradLhs, gradRhs)
    }
  )
}

@derivative(of: /)
@inlinable
internal func _vjpDivideTensors(_ lhs: Tensor, _ rhs: Tensor) -> (
  value: Tensor,
  pullback: (Tensor) -> (Tensor, Tensor)
) {
  let result = lhs.dividing(rhs)
  return (
    result,
    { v in
      let gradLhs = _reduceLike(v.dividing(rhs), targetShape: lhs.shape)
      let rhsSquared = rhs.multiplying(rhs)
      let gradRhsNumerator = v.multiplying(lhs)
      let gradRhs = _reduceLike(gradRhsNumerator.dividing(rhsSquared).negated(), targetShape: rhs.shape)
      return (gradLhs, gradRhs)
    }
  )
}

// MARK: - Tensor ⊗ Scalar Operators
@derivative(of: +, wrt: lhs)
@inlinable
internal func _vjpAddTensorScalar<T: TorchArithmetic>(
  _ lhs: Tensor,
  _ rhs: T
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let result = lhs.adding(rhs)
  return (
    result,
    { v in _reduceLike(v, targetShape: lhs.shape) }
  )
}

@derivative(of: -, wrt: lhs)
@inlinable
internal func _vjpSubtractTensorScalar<T: TorchArithmetic>(
  _ lhs: Tensor,
  _ rhs: T
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let result = lhs.subtracting(rhs)
  return (
    result,
    { v in _reduceLike(v, targetShape: lhs.shape) }
  )
}

@derivative(of: *, wrt: lhs)
@inlinable
internal func _vjpMultiplyTensorScalar<T: TorchArithmetic>(
  _ lhs: Tensor,
  _ rhs: T
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let result = lhs.multiplying(rhs)
  return (
    result,
    { v in _reduceLike(v.multiplying(rhs), targetShape: lhs.shape) }
  )
}

@derivative(of: /, wrt: lhs)
@inlinable
internal func _vjpDivideTensorScalar<T: TorchArithmetic>(
  _ lhs: Tensor,
  _ rhs: T
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let result = lhs.dividing(rhs)
  return (
    result,
    { v in _reduceLike(v.dividing(rhs), targetShape: lhs.shape) }
  )
}

// MARK: - Scalar ⊗ Tensor Operators
@derivative(of: +, wrt: rhs)
@inlinable
internal func _vjpAddScalarTensor<T: TorchArithmetic>(
  _ lhs: T,
  _ rhs: Tensor
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let result = rhs.adding(lhs)
  return (
    result,
    { v in _reduceLike(v, targetShape: rhs.shape) }
  )
}

@derivative(of: -, wrt: rhs)
@inlinable
internal func _vjpSubtractScalarTensor<T: TorchArithmetic>(
  _ lhs: T,
  _ rhs: Tensor
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let result = rhs.negated().adding(lhs)
  return (
    result,
    { v in _reduceLike(v.negated(), targetShape: rhs.shape) }
  )
}

@derivative(of: *, wrt: rhs)
@inlinable
internal func _vjpMultiplyScalarTensor<T: TorchArithmetic>(
  _ lhs: T,
  _ rhs: Tensor
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let result = rhs.multiplying(lhs)
  return (
    result,
    { v in _reduceLike(v.multiplying(lhs), targetShape: rhs.shape) }
  )
}

@derivative(of: /, wrt: rhs)
@inlinable
internal func _vjpDivideScalarTensor<T: TorchArithmetic>(
  _ lhs: T,
  _ rhs: Tensor
) -> (
  value: Tensor,
  pullback: (Tensor) -> Tensor
) {
  let lhsTensor = Tensor(lhs)
  let result = lhsTensor.dividing(rhs)
  return (
    result,
    { v in
      let rhsSquared = rhs.multiplying(rhs)
      let gradNumerator = v.multiplying(lhsTensor)
      let grad = gradNumerator.dividing(rhsSquared).negated()
      return _reduceLike(grad, targetShape: rhs.shape)
    }
  )
}
