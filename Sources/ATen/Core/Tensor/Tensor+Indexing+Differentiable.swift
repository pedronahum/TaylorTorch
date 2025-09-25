import _Differentiation

/// Normalises a possibly negative index into the concrete axis range, clamping to
/// the equivalent positive offset when the caller provides negative positions.
@usableFromInline
@inline(__always)
internal func _canonicalizeIndex(_ index: Int, axisSize: Int) -> Int {
  var idx = index
  if idx < 0 { idx += axisSize }
  return idx
}

/// Resolves slice bounds into the closed-open interval `[0, axisSize)` while
/// supporting negative indices that wrap from the tail of the dimension.
@usableFromInline
@inline(__always)
internal func _canonicalizeSliceBounds(axisSize: Int, start: Int, end: Int) -> (Int, Int) {
  var s = start
  var e = end
  if s < 0 { s += axisSize }
  if e < 0 { e += axisSize }
  s = min(max(s, 0), axisSize)
  e = min(max(e, 0), axisSize)
  return (s, e)
}

/// Builds a permutation that moves the requested dimension to the end of the
/// order while preserving the relative ordering of all other axes.
@usableFromInline
@inline(__always)
internal func _moveDimToEnd(rank: Int, dim: Int) -> [Int] {
  let resolvedDim = _normalizeDimension(dim, rank: rank)
  var order = Array(0..<rank)
  order.remove(at: resolvedDim)
  order.append(resolvedDim)
  return order
}

/// Computes the inverse permutation for the provided axis order so gradients can
/// be restored to their original dimensional arrangement.
@usableFromInline
@inline(__always)
internal func _inversePermutation(_ perm: [Int]) -> [Int] {
  var inverse = [Int](repeating: 0, count: perm.count)
  for (index, value) in perm.enumerated() {
    inverse[value] = index
  }
  return inverse
}

extension Tensor {
  /// Reverse-mode derivative for `select(dim:index:)`, scattering incoming
  /// gradients back into the chosen slice and zeroing untouched positions.
  @derivative(of: select(dim:index:), wrt: self)
  @inlinable
  internal func _vjpSelect<T: TorchSliceIndex & FixedWidthInteger>(
    dim: Int,
    index: T
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = select(dim: dim, index: index)
    return (
      result,
      { v in
        let resolvedDim = _normalizeDimension(dim, rank: self.rank)
        let axisSize = self.shape[resolvedDim]
        let resolvedIndex = _canonicalizeIndex(Int(truncatingIfNeeded: index), axisSize: axisSize)
        precondition(resolvedIndex >= 0 && resolvedIndex < axisSize, "index out of range")

        let dtype = v.dtype ?? self.dtype ?? .float32
        let device = self.device

        var pieces: [Tensor] = []
        if resolvedIndex > 0 {
          var prefixShape = self.shape
          prefixShape[resolvedDim] = resolvedIndex
          pieces.append(Tensor.zeros(shape: prefixShape, dtype: dtype, device: device))
        }

        let middle = v.unsqueezed(dim: resolvedDim)
        pieces.append(middle)

        let suffixLength = axisSize - resolvedIndex - 1
        if suffixLength > 0 {
          var suffixShape = self.shape
          suffixShape[resolvedDim] = suffixLength
          pieces.append(Tensor.zeros(shape: suffixShape, dtype: dtype, device: device))
        }

        return Tensor.cat(pieces, dim: resolvedDim)
      }
    )
  }

  /// Reverse-mode derivative for `narrow(dim:start:length:)`, padding zeros on
  /// either side of the narrowed region when accumulating gradients.
  @derivative(of: narrow(dim:start:length:), wrt: self)
  @inlinable
  internal func _vjpNarrow<T: TorchSliceIndex & FixedWidthInteger>(
    dim: Int,
    start: T,
    length: T
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = narrow(dim: dim, start: start, length: length)
    return (
      result,
      { v in
        let resolvedDim = _normalizeDimension(dim, rank: self.rank)
        let axisSize = self.shape[resolvedDim]
        let resolvedStart = _canonicalizeIndex(Int(truncatingIfNeeded: start), axisSize: axisSize)
        let resolvedLength = Int(truncatingIfNeeded: length)

        precondition(resolvedLength >= 0, "length must be non-negative")
        precondition(resolvedStart >= 0 && resolvedStart <= axisSize, "start out of range")
        precondition(resolvedStart &+ resolvedLength <= axisSize, "narrow region exceeds axis")

        let dtype = v.dtype ?? self.dtype ?? .float32
        let device = self.device

        var pieces: [Tensor] = []
        if resolvedStart > 0 {
          var prefixShape = self.shape
          prefixShape[resolvedDim] = resolvedStart
          pieces.append(Tensor.zeros(shape: prefixShape, dtype: dtype, device: device))
        }

        pieces.append(v)

        let suffixLength = axisSize - resolvedStart - resolvedLength
        if suffixLength > 0 {
          var suffixShape = self.shape
          suffixShape[resolvedDim] = suffixLength
          pieces.append(Tensor.zeros(shape: suffixShape, dtype: dtype, device: device))
        }

        return Tensor.cat(pieces, dim: resolvedDim)
      }
    )
  }

  /// Reverse-mode derivative for strided `slice(dim:start:end:step:)`,
  /// scattering the upstream gradient along the sampled indices while leaving
  /// all other positions zero-initialised.
  @derivative(of: slice(dim:start:end:step:), wrt: self)
  @inlinable
  internal func _vjpSlice(
    dim: Int,
    start: Int = 0,
    end: Int? = nil,
    step: Int = 1
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = slice(dim: dim, start: start, end: end, step: step)
    return (
      result,
      { v in
        let resolvedDim = _normalizeDimension(dim, rank: self.rank)
        let axisSize = self.shape[resolvedDim]
        let resolvedStep = step
        precondition(resolvedStep > 0, "slice step must be positive")

        let upperBound = end ?? axisSize
        let (resolvedStart, resolvedEnd) = _canonicalizeSliceBounds(
          axisSize: axisSize,
          start: start,
          end: upperBound
        )
        let dtype = v.dtype ?? self.dtype ?? .float32
        let device = self.device

        if resolvedEnd <= resolvedStart || v.shape[resolvedDim] == 0 {
          return Tensor.zeros(shape: self.shape, dtype: dtype, device: device)
        }

        let outputShape = self.shape

        let axisIndices = Tensor.arange(
          0,
          to: axisSize,
          step: 1,
          dtype: .int64,
          device: device
        )

        let selectedIndices = Tensor.arange(
          resolvedStart,
          to: resolvedEnd,
          step: resolvedStep,
          dtype: .int64,
          device: device
        )

        let selectionMatrix =
          selectedIndices
          .unsqueezed(dim: 1)
          .eq(axisIndices.unsqueezed(dim: 0))
          .to(dtype: dtype)

        let permuteOrder = _moveDimToEnd(rank: self.rank, dim: resolvedDim)
        let inversePerm = _inversePermutation(permuteOrder)
        let vPermuted = v.permuted(permuteOrder)
        let permutedShape = vPermuted.shape
        let batch = permutedShape.dropLast().reduce(1, *)
        let features = permutedShape.last ?? 1

        let v2D = vPermuted.reshaped([batch, features])
        let scattered2D = v2D.matmul(selectionMatrix)

        var restoredShape = permutedShape
        restoredShape[restoredShape.count - 1] = axisSize
        let restored = scattered2D.reshaped(restoredShape)
        return restored.permuted(inversePerm).reshaped(outputShape)
      }
    )
  }
}
