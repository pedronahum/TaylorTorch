import _Differentiation

extension Tensor {
  @derivative(of: expanded(to:implicit:), wrt: self)
  @inlinable
  internal func _vjpExpanded(to shape: [Int], implicit: Bool) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = expanded(to: shape, implicit: implicit)
    return (
      result,
      { v in _reduceLike(v, targetShape: self.shape) }
    )
  }

  @derivative(of: expanded(as:), wrt: self)
  @inlinable
  internal func _vjpExpanded(as other: Tensor) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = expanded(as: other)
    return (
      result,
      { v in _reduceLike(v, targetShape: self.shape) }
    )
  }

  @derivative(of: broadcasted(to:), wrt: self)
  @inlinable
  internal func _vjpBroadcasted(to shape: [Int]) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = broadcasted(to: shape)
    return (
      result,
      { v in _reduceLike(v, targetShape: self.shape) }
    )
  }
}

