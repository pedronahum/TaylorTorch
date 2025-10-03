import ATenCXX
import _Differentiation

extension Tensor {
  @inlinable
  func conv2d(
    weight: Tensor,
    bias: Tensor?,
    stride: [Int64],
    padding: [Int64],
    dilation: [Int64],
    groups: Int64
  ) -> Tensor {
    let y: TTSTensor = stride.withUnsafeBufferPointer { sp in
      padding.withUnsafeBufferPointer { pp in
        dilation.withUnsafeBufferPointer { dp in
          if var b = bias {
            return withUnsafePointer(to: &b._impl) { bptr in
              ATenCXX.TTSTensor._conv2d(
                self._impl,  // input
                weight._impl,  // weight
                bptr,  // &bias._impl
                sp.baseAddress!, Int(sp.count),
                pp.baseAddress!, Int(pp.count),
                dp.baseAddress!, Int(dp.count),
                groups
              )
            }
          } else {
            return ATenCXX.TTSTensor._conv2d(
              self._impl,  // input
              weight._impl,  // weight
              nil,  // bias = nullptr
              sp.baseAddress!, Int(sp.count),
              pp.baseAddress!, Int(pp.count),
              dp.baseAddress!, Int(dp.count),
              groups
            )
          }
        }
      }
    }
    return Tensor(y)
  }

  @inlinable
  internal func _conv2dBackward(
    upstream: Tensor,
    weight: Tensor,
    stride: [Int64],
    padding: [Int64],
    dilation: [Int64],
    groups: Int64
  ) -> (input: Tensor, weight: Tensor, bias: Tensor) {

    let tup = stride.withUnsafeBufferPointer { sp in
      padding.withUnsafeBufferPointer { pp in
        dilation.withUnsafeBufferPointer { dp in
          ATenCXX.TTSTensor._conv2d_backward(
            upstream._impl,  // grad_out
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

    let gi = ATenCXX.TTSTensor._conv2d_backward_get0(tup)
    let gw = ATenCXX.TTSTensor._conv2d_backward_get1(tup)
    let gb = ATenCXX.TTSTensor._conv2d_backward_get2(tup)
    return (Tensor(gi), Tensor(gw), Tensor(gb))
  }

  // In Tensor+NN.swift
  @differentiable(reverse,wrt: (self))
  public func maxPool2d(
    kernelSize: [Int64], stride: [Int64],
    padding: [Int64], dilation: [Int64], ceilMode: Bool
  ) -> Tensor {
    // ✅ Wrap non-differentiable parameters to satisfy the compiler
    let ks = withoutDerivative(at: kernelSize)
    let s = withoutDerivative(at: stride)
    let p = withoutDerivative(at: padding)
    let d = withoutDerivative(at: dilation)
    let cm = withoutDerivative(at: ceilMode)

    let tensor = withoutDerivative(at: self)
    return Tensor(
      tensor._impl.max_pool2d(
        ks, ks.count, s, s.count, p, p.count, d, d.count, cm
      )
    )
  }

  @differentiable(reverse,wrt: (self))
  public func avgPool2d(
    kernelSize: [Int64], stride: [Int64],
    padding: [Int64], ceilMode: Bool
  ) -> Tensor {
    // ✅ Wrap non-differentiable parameters to satisfy the compiler
    let ks = withoutDerivative(at: kernelSize)
    let s = withoutDerivative(at: stride)
    let p = withoutDerivative(at: padding)
    let cm = withoutDerivative(at: ceilMode)

    let tensor = withoutDerivative(at: self)
    return Tensor(
      tensor._impl.avg_pool2d(
        ks, ks.count, s, s.count, p, p.count, cm
      )
    )
  }
}
