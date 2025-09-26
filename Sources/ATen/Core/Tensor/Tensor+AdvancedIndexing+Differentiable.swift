import _Differentiation

extension Tensor {
  /// Reverse-mode derivative for `indexSelect(dim:indices:)`, scattering the
  /// upstream gradient into a tensor shaped like the source while accumulating
  /// contributions for repeated indices.
  @derivative(of: indexSelect(dim:indices:), wrt: self)
  @inlinable
  internal func _vjpIndexSelect<T: TorchSliceIndex & FixedWidthInteger>(
    dim: Int,
    indices: [T]
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = indexSelect(dim: dim, indices: indices)
    return (
      result,
      { v in
        let resolvedDim = _normalizeDimension(dim, rank: self.rank)
        let axisSize = self.shape[resolvedDim]

        let resolvedIndices: [Int] = indices.map { rawIndex in
          var value = Int(truncatingIfNeeded: rawIndex)
          if value < 0 { value += axisSize }
          precondition(value >= 0 && value < axisSize, "index out of range")
          return value
        }

        let gradDType = v.dtype ?? self.dtype ?? .float32
        let device = v.device

        if resolvedIndices.isEmpty {
          return Tensor.zeros(shape: self.shape, dtype: gradDType, device: device)
        }

        let indicesTensor = Tensor(
          array: resolvedIndices.map(Int64.init),
          shape: [resolvedIndices.count],
          device: device
        )

        let axisIndices = Tensor.arange(
          Int64(0),
          to: Int64(axisSize),
          step: Int64(1),
          dtype: .int64,
          device: device
        )

        let selectionMatrix = indicesTensor
          .unsqueezed(dim: 1)
          .eq(axisIndices.unsqueezed(dim: 0))
          .to(dtype: gradDType)

        let permuteOrder = _moveDimToEnd(rank: self.rank, dim: resolvedDim)
        let inversePermutation = _inversePermutation(permuteOrder)

        let baseGradient: Tensor
        if let currentType = v.dtype, currentType == gradDType {
          baseGradient = v
        } else {
          baseGradient = v.to(dtype: gradDType)
        }

        let permuted = baseGradient.permuted(permuteOrder)
        let permutedShape = permuted.shape
        let batch = permutedShape.dropLast().reduce(1, *)
        let features = permutedShape.last ?? 1

        let reshaped = permuted.reshaped([batch, features])
        let scattered2D = reshaped.matmul(selectionMatrix)

        var restoredShape = permutedShape
        restoredShape[restoredShape.count - 1] = axisSize

        let restored = scattered2D.reshaped(restoredShape)
        return restored.permuted(inversePermutation).reshaped(self.shape)
      }
    )
  }
}
