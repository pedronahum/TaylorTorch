import _Differentiation

@usableFromInline
internal struct MeanReductionStep {
  @usableFromInline let dim: Int
  @usableFromInline let inputShape: [Int]
  @usableFromInline let axisSize: Int

  @usableFromInline
  internal init(dim: Int, inputShape: [Int], axisSize: Int) {
    self.dim = dim
    self.inputShape = inputShape
    self.axisSize = axisSize
  }
}

extension Tensor {
  /// Reverse-mode derivative for axis-based `select`, deferring to the index
  /// derivative once the logical axis has been resolved.
  @derivative(of: select(dim:index:), wrt: self)
  @inlinable
  internal func _vjpSelect<T: TorchSliceIndex & FixedWidthInteger>(
    dim axis: Axis,
    index: T
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let resolvedDim = axis.resolve(forRank: self.rank)
    return self._vjpSelect(dim: resolvedDim, index: index)
  }

  /// Reverse-mode derivative for axis-based `narrow`, reusing the integer-axis
  /// implementation after resolving the logical axis.
  @derivative(of: narrow(dim:start:length:), wrt: self)
  @inlinable
  internal func _vjpNarrow<T: TorchSliceIndex & FixedWidthInteger>(
    dim axis: Axis,
    start: T,
    length: T
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let resolvedDim = axis.resolve(forRank: self.rank)
    return self._vjpNarrow(dim: resolvedDim, start: start, length: length)
  }

  /// Reverse-mode derivative for axis-based `slice`, forwarding to the integer
  /// derivative once the axis is materialised.
  @derivative(of: slice(dim:start:end:step:), wrt: self)
  @inlinable
  internal func _vjpSlice(
    dim axis: Axis,
    start: Int = 0,
    end: Int? = nil,
    step: Int = 1
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let resolvedDim = axis.resolve(forRank: self.rank)
    return self._vjpSlice(dim: resolvedDim, start: start, end: end, step: step)
  }

  /// Reverse-mode derivative for axis-based `transposed`, simply swapping the
  /// same axes in the pullback via the integer derivative.
  @derivative(of: transposed(_:_:), wrt: self)
  @inlinable
  internal func _vjpTransposed(_ first: Axis, _ second: Axis) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let dim0 = first.resolve(forRank: self.rank)
    let dim1 = second.resolve(forRank: self.rank)
    return self._vjpTransposed(dim0, dim1)
  }

  /// Reverse-mode derivative for axis-based `sum`, replaying the reductions in
  /// reverse while expanding gradients along each eliminated dimension.
  @derivative(of: sum(along:keepdim:), wrt: self)
  @inlinable
  internal func _vjpSum(along axes: [Axis], keepdim: Bool) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    if axes.isEmpty {
      return (self, { $0 })
    }

    typealias ReductionStep = (dim: Int, inputShape: [Int])
    var steps: [ReductionStep] = []
    var current = self

    for axis in axes {
      let resolvedDim = axis.resolve(forRank: current.rank)
      steps.append((resolvedDim, current.shape))
      current = current.sum(dim: resolvedDim, keepdim: keepdim)
    }

    let result = current

    return (
      result,
      { v in
        var grad = v
        for step in steps.reversed() {
          if !keepdim {
            grad = grad.unsqueezed(dim: step.dim)
          }
          grad = grad.expanded(to: step.inputShape)
        }
        return grad
      }
    )
  }

  /// Reverse-mode derivative for axis-based `mean`, expanding gradients along
  /// reduced axes and dividing by each axis length.
  @derivative(of: mean(along:keepdim:), wrt: self)
  @inlinable
  internal func _vjpMean(along axes: [Axis], keepdim: Bool) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    if axes.isEmpty {
      return (self, { $0 })
    }

    var steps: [MeanReductionStep] = []
    var current = self

    for axis in axes {
      let resolvedDim = axis.resolve(forRank: current.rank)
      let shape = current.shape
      let axisSize = shape[resolvedDim]
      steps.append(MeanReductionStep(dim: resolvedDim, inputShape: shape, axisSize: axisSize))
      current = current.mean(dim: resolvedDim, keepdim: keepdim)
    }

    let result = current

    return (
      result,
      { v in
        var grad = v
        for step in steps.reversed() {
          if !keepdim {
            grad = grad.unsqueezed(dim: step.dim)
          }
          grad = grad.expanded(to: step.inputShape)
          grad = grad.dividing(Double(step.axisSize))
        }
        return grad
      }
    )
  }
}
