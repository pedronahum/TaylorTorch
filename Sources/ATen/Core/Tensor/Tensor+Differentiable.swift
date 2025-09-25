import _Differentiation

/// Projects a heterogeneous `Scalar` value into a `Double` so broadcasted
/// gradients can be scaled using native arithmetic regardless of the payload.
@usableFromInline
internal func _scalarToDouble(_ scalar: Scalar) -> Double {
  switch scalar {
  case .int8(let value): return Double(value)
  case .int16(let value): return Double(value)
  case .int32(let value): return Double(value)
  case .int64(let value): return Double(value)
  case .float(let value): return Double(value)
  case .double(let value): return value
  case .bool(let value): return value ? 1.0 : 0.0
  }
}

extension Tensor {
  /// Reverse-mode derivative for the full `adding(_:alpha:)` overload, reducing
  /// broadcasted gradients to match each operand before returning the pair of
  /// tangents.
  @derivative(of: adding(_:alpha:))
  @inlinable
  internal func _vjpAdding(_ other: Tensor, alpha: Scalar) -> (
    value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)
  ) {
    let result = adding(other, alpha: alpha)
    return (
      result,
      { v in
        let scale = _scalarToDouble(alpha)
        let gradSelf = _reduceLike(v, targetShape: self.shape)
        let gradOther = _reduceLike(v.multiplying(scale), targetShape: other.shape)
        return (gradSelf, gradOther)
      }
    )
  }

  /// Reverse-mode derivative for tensor-scalar `adding`, ensuring the gradient
  /// collapses along broadcast dimensions while leaving the scalar untouched.
  @derivative(of: adding(_:), wrt: self)
  @inlinable
  internal func _vjpAddingScalar<T: TorchArithmetic>(_ scalar: T) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = adding(scalar)
    return (result, { v in _reduceLike(v, targetShape: self.shape) })
  }
}
