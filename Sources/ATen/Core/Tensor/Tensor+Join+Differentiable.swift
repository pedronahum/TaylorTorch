import _Differentiation

extension Tensor {
  /// Reverse-mode derivative for `Tensor.cat`, slicing the upstream gradient
  /// along the concatenated axis so each operand receives its contributed
  /// segment.
  @derivative(of: cat)
  @inlinable
  internal static func _vjpCat(
    _ tensors: [Tensor],
    dim: Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.TangentVector) {
    precondition(!tensors.isEmpty, "cat: empty tensor list")

    let baseRank = tensors[0].rank
    let resolvedDim = _normalizeDimension(dim, rank: baseRank)
    let lengths = tensors.map { $0.shape[resolvedDim] }

    let result = cat(tensors, dim: dim)
    return (
      result,
      { v in
        var offset = 0
        var grads: [Tensor] = []
        grads.reserveCapacity(lengths.count)
        for length in lengths {
          let slice = v.narrow(dim: resolvedDim, start: Int64(offset), length: Int64(length))
          grads.append(slice)
          offset += length
        }
        return Array<Tensor>.TangentVector(grads)
      }
    )
  }

  /// Reverse-mode derivative for `Tensor.stack`, selecting the gradient slice
  /// associated with each operand along the stacked axis.
  @derivative(of: stack(_:dim:), wrt: tensors)
  @inlinable
  internal static func _vjpStack(
    _ tensors: [Tensor],
    dim: Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Tensor>.TangentVector) {
    precondition(!tensors.isEmpty, "stack: empty tensor list")

    let baseRank = tensors[0].rank
    let resolvedDim = _normalizeDimension(dim, rank: baseRank + 1)

    let result = stack(tensors, dim: dim)
    return (
      result,
      { v in
        var grads: [Tensor] = []
        grads.reserveCapacity(tensors.count)
        for index in tensors.indices {
          grads.append(v.select(dim: resolvedDim, index: Int64(index)))
        }
        return Array<Tensor>.TangentVector(grads)
      }
    )
  }
}
