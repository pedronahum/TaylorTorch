@preconcurrency import ATenCXX
import _Differentiation

extension Tensor {
  @inlinable
  internal func _maxPool2dWithIndices(
    kernelSize: [Int64],
    stride: [Int64],
    padding: [Int64],
    dilation: [Int64],
    ceilMode: Bool
  ) -> (output: Tensor, indices: Tensor) {
    let pair = _impl.max_pool2d_with_indices(
      kernelSize, kernelSize.count,
      stride, stride.count,
      padding, padding.count,
      dilation, dilation.count,
      ceilMode
    )
    return (Tensor(pair.first), Tensor(pair.second))
  }

  @inlinable
  internal func _maxPool2dBackward(
    upstream: Tensor,
    kernelSize: [Int64],
    stride: [Int64],
    padding: [Int64],
    dilation: [Int64],
    ceilMode: Bool,
    indices: Tensor
  ) -> Tensor {
    Tensor(
      _impl.max_pool2d_with_indices_backward(
        upstream._impl,
        kernelSize, kernelSize.count,
        stride, stride.count,
        padding, padding.count,
        dilation, dilation.count,
        ceilMode,
        indices._impl
      )
    )
  }

  @inlinable
  internal func _avgPool2dBackward(
    upstream: Tensor,
    kernelSize: [Int64],
    stride: [Int64],
    padding: [Int64],
    ceilMode: Bool
  ) -> Tensor {
    Tensor(
      _impl.avg_pool2d_backward(
        upstream._impl,
        kernelSize, kernelSize.count,
        stride, stride.count,
        padding, padding.count,
        ceilMode
      )
    )
  }

  // ===== Conv2D VJP =====

  @derivative(of: conv2d, wrt: (self, weight, bias))
  @inlinable
  internal func _vjpConv2d(
    weight: Tensor,
    bias: Tensor?,
    stride: [Int64],
    padding: [Int64],
    dilation: [Int64],
    groups: Int64
  ) -> (
    value: Tensor,
    pullback: (Tensor.TangentVector) -> (
      Tensor.TangentVector, Tensor.TangentVector, Optional<Tensor>.TangentVector
    )
  ) {
    let value = self.conv2d(
      weight: weight,
      bias: bias,
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups
    )

    return (
      value,
      { v in
        let tup = stride.withUnsafeBufferPointer { sp in
          padding.withUnsafeBufferPointer { pp in
            dilation.withUnsafeBufferPointer { dp in
              ATenCXX.TTSTensor._conv2d_backward(
                v._impl,  // grad_out
                self._impl,  // input
                weight._impl,  // weight
                sp.baseAddress!, Int(sp.count),
                pp.baseAddress!, Int(pp.count),
                dp.baseAddress!, Int(dp.count),
                groups
              )
            }
          }
        }
        let gradInput = Tensor(ATenCXX.TTSTensor._conv2d_backward_get0(tup))
        let gradWeight = Tensor(ATenCXX.TTSTensor._conv2d_backward_get1(tup))
        let gradBias = Tensor(ATenCXX.TTSTensor._conv2d_backward_get2(tup))

        let tangentBias: Optional<Tensor>.TangentVector
        if bias != nil {
          // Convert the bias gradient into the optional tangent wrapper.
          tangentBias = Optional<Tensor>.TangentVector(gradBias as Tensor.TangentVector)
        } else {
          tangentBias = .zero
        }

        return (
          gradInput as Tensor.TangentVector,
          gradWeight as Tensor.TangentVector,
          tangentBias
        )
      }
    )
  }

}
