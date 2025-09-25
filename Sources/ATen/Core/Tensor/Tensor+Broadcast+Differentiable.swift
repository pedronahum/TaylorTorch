import _Differentiation

extension Tensor {
  /// Reverse-mode derivative for `expanded(to:implicit:)`, summing over the
  /// dimensions that were broadcast when collapsing the gradient back to the
  /// source tensor.
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

  /// Reverse-mode derivative for `expanded(as:)`, reusing `_reduceLike` to fold
  /// gradients across axes introduced by the expansion.
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

  /// Reverse-mode derivative for `broadcasted(to:)`, summing along broadcasted
  /// axes so the returned gradient matches the source tensor's shape.
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
