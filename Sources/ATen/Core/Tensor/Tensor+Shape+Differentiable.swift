import _Differentiation

extension Tensor {
  @inlinable
  public func flattened() -> Tensor {
    return self.flattened(startDim: 0, endDim: self.rank - 1)
  }
  @derivative(of: transposed(_:_:), wrt: self)
  @inlinable
  internal func _vjpTransposed(_ dim0: Int, _ dim1: Int) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = transposed(dim0, dim1)
    return (
      result,
      { v in v.transposed(dim0, dim1) }
    )
  }

  @derivative(of: permuted(_:), wrt: self)
  @inlinable
  internal func _vjpPermuted(_ order: [Int]) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = permuted(order)
    return (
      result,
      { v in
        let rank = self.rank
        var normalized = [Int](repeating: 0, count: order.count)
        for (index, dim) in order.enumerated() {
          let positive = dim >= 0 ? dim : dim + rank
          normalized[index] = positive
        }
        var inverse = [Int](repeating: 0, count: normalized.count)
        for (index, dim) in normalized.enumerated() {
          inverse[dim] = index
        }
        return v.permuted(inverse)
      }
    )
  }

  @derivative(of: reshaped(_:), wrt: self)
  @inlinable
  internal func _vjpReshaped(_ shape: [Int]) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let originalShape = self.shape
    let result = reshaped(shape)
    return (
      result,
      { v in v.reshaped(originalShape) }
    )
  }

  @derivative(of: flattened(startDim:endDim:), wrt: self)
  @inlinable
  internal func _vjpFlattened(startDim: Int, endDim: Int) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let originalShape = self.shape
    let result = flattened(startDim: startDim, endDim: endDim)
    return (
      result,
      { v in v.reshaped(originalShape) }
    )
  }

  /*
  @derivative(of: flattened, wrt: self)
  @inlinable
  internal func _vjpFlattenedDefault() -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let originalShape = self.shape
    let result = flattened()
    return (
      result,
      { v in v.reshaped(originalShape) }
    )
  }
  */

  @derivative(of: squeezed, wrt: self)
  @inlinable
  internal func _vjpSqueezed() -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let originalShape = self.shape
    let result = squeezed()
    return (
      result,
      { v in
        var grad = v
        for (index, size) in originalShape.enumerated() where size == 1 {
          grad = grad.unsqueezed(dim: index)
        }
        return grad
      }
    )
  }

  @derivative(of: squeezed(dim:), wrt: self)
  @inlinable
  internal func _vjpSqueezed(dim: Int) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = squeezed(dim: dim)
    return (
      result,
      { v in
        let resolvedDim = dim >= 0 ? dim : dim + self.rank
        return v.unsqueezed(dim: resolvedDim)
      }
    )
  }

  @derivative(of: unsqueezed(dim:), wrt: self)
  @inlinable
  internal func _vjpUnsqueezed(dim: Int) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let result = unsqueezed(dim: dim)
    return (
      result,
      { v in
        let resolvedDim = dim >= 0 ? dim : dim + v.rank
        return v.squeezed(dim: resolvedDim)
      }
    )
  }
}
