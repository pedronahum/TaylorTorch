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

// MARK: - Native Batch Normalization

extension Tensor {
  /// Native batch normalization forward pass.
  /// Returns (output, saveMean, saveInvstd) for use in backward pass.
  @inlinable
  public func nativeBatchNorm(
    weight: Tensor?,
    bias: Tensor?,
    runningMean: inout Tensor?,
    runningVar: inout Tensor?,
    training: Bool,
    momentum: Double,
    epsilon: Double
  ) -> (output: Tensor, saveMean: Tensor, saveInvstd: Tensor) {
    if var w = weight, var b = bias, var rm = runningMean, var rv = runningVar {
      let tup = withUnsafePointer(to: &w._impl) { wptr in
        withUnsafePointer(to: &b._impl) { bptr in
          withUnsafeMutablePointer(to: &rm._impl) { rmptr in
            withUnsafeMutablePointer(to: &rv._impl) { rvptr in
              ATenCXX.TTSTensor._native_batch_norm(
                self._impl, wptr, bptr, rmptr, rvptr, training, momentum, epsilon
              )
            }
          }
        }
      }
      runningMean = rm
      runningVar = rv
      let output = ATenCXX.TTSTensor._native_batch_norm_get0(tup)
      let saveMean = ATenCXX.TTSTensor._native_batch_norm_get1(tup)
      let saveInvstd = ATenCXX.TTSTensor._native_batch_norm_get2(tup)
      return (Tensor(output), Tensor(saveMean), Tensor(saveInvstd))
    } else if var w = weight, var b = bias {
      let tup = withUnsafePointer(to: &w._impl) { wptr in
        withUnsafePointer(to: &b._impl) { bptr in
          ATenCXX.TTSTensor._native_batch_norm(
            self._impl, wptr, bptr, nil, nil, training, momentum, epsilon
          )
        }
      }
      let output = ATenCXX.TTSTensor._native_batch_norm_get0(tup)
      let saveMean = ATenCXX.TTSTensor._native_batch_norm_get1(tup)
      let saveInvstd = ATenCXX.TTSTensor._native_batch_norm_get2(tup)
      return (Tensor(output), Tensor(saveMean), Tensor(saveInvstd))
    } else {
      let tup = ATenCXX.TTSTensor._native_batch_norm(
        self._impl, nil, nil, nil, nil, training, momentum, epsilon
      )
      let output = ATenCXX.TTSTensor._native_batch_norm_get0(tup)
      let saveMean = ATenCXX.TTSTensor._native_batch_norm_get1(tup)
      let saveInvstd = ATenCXX.TTSTensor._native_batch_norm_get2(tup)
      return (Tensor(output), Tensor(saveMean), Tensor(saveInvstd))
    }
  }

  /// Native batch normalization backward pass.
  /// Returns (gradInput, gradWeight, gradBias).
  @inlinable
  public func _nativeBatchNormBackward(
    upstream: Tensor,
    weight: Tensor?,
    runningMean: Tensor?,
    runningVar: Tensor?,
    saveMean: Tensor?,
    saveInvstd: Tensor?,
    training: Bool,
    epsilon: Double
  ) -> (gradInput: Tensor, gradWeight: Tensor, gradBias: Tensor) {
    if var w = weight, var rm = runningMean, var rv = runningVar,
       var sm = saveMean, var si = saveInvstd
    {
      let tup = withUnsafePointer(to: &w._impl) { wptr in
        withUnsafePointer(to: &rm._impl) { rmptr in
          withUnsafePointer(to: &rv._impl) { rvptr in
            withUnsafePointer(to: &sm._impl) { smptr in
              withUnsafePointer(to: &si._impl) { siptr in
                ATenCXX.TTSTensor._native_batch_norm_backward(
                  upstream._impl, self._impl, wptr, rmptr, rvptr, smptr, siptr,
                  training, epsilon
                )
              }
            }
          }
        }
      }
      let gi = ATenCXX.TTSTensor._native_batch_norm_get0(tup)
      let gw = ATenCXX.TTSTensor._native_batch_norm_get1(tup)
      let gb = ATenCXX.TTSTensor._native_batch_norm_get2(tup)
      return (Tensor(gi), Tensor(gw), Tensor(gb))
    } else if var w = weight, var sm = saveMean, var si = saveInvstd {
      let tup = withUnsafePointer(to: &w._impl) { wptr in
        withUnsafePointer(to: &sm._impl) { smptr in
          withUnsafePointer(to: &si._impl) { siptr in
            ATenCXX.TTSTensor._native_batch_norm_backward(
              upstream._impl, self._impl, wptr, nil, nil, smptr, siptr,
              training, epsilon
            )
          }
        }
      }
      let gi = ATenCXX.TTSTensor._native_batch_norm_get0(tup)
      let gw = ATenCXX.TTSTensor._native_batch_norm_get1(tup)
      let gb = ATenCXX.TTSTensor._native_batch_norm_get2(tup)
      return (Tensor(gi), Tensor(gw), Tensor(gb))
    } else {
      let tup = ATenCXX.TTSTensor._native_batch_norm_backward(
        upstream._impl, self._impl, nil, nil, nil, nil, nil,
        training, epsilon
      )
      let gi = ATenCXX.TTSTensor._native_batch_norm_get0(tup)
      let gw = ATenCXX.TTSTensor._native_batch_norm_get1(tup)
      let gb = ATenCXX.TTSTensor._native_batch_norm_get2(tup)
      return (Tensor(gi), Tensor(gw), Tensor(gb))
    }
  }
}

// MARK: - Native Linear

extension Tensor {
  /// Native linear forward pass using at::linear.
  /// Note: weight must be in PyTorch convention [out_features, in_features]
  @inlinable
  internal func nativeLinear(
    weight: Tensor,
    bias: Tensor?
  ) -> Tensor {
    if var b = bias {
      return Tensor(withUnsafePointer(to: &b._impl) { bptr in
        ATenCXX.TTSTensor._linear(self._impl, weight._impl, bptr)
      })
    } else {
      return Tensor(ATenCXX.TTSTensor._linear(self._impl, weight._impl, nil))
    }
  }

  /// Native linear backward pass using at::linear_backward.
  /// Returns (gradInput, gradWeight, gradBias).
  /// Note: weight must be in PyTorch convention [out_features, in_features]
  @inlinable
  internal func _nativeLinearBackward(
    upstream: Tensor,
    weight: Tensor
  ) -> (gradInput: Tensor, gradWeight: Tensor, gradBias: Tensor) {
    let tup = ATenCXX.TTSTensor._linear_backward(
      self._impl,
      upstream._impl,
      weight._impl
    )
    let gi = ATenCXX.TTSTensor._linear_backward_get0(tup)
    let gw = ATenCXX.TTSTensor._linear_backward_get1(tup)
    let gb = ATenCXX.TTSTensor._linear_backward_get2(tup)
    return (Tensor(gi), Tensor(gw), Tensor(gb))
  }
}

// Custom derivatives for pooling live alongside the primals to avoid duplicate symbols.
extension Tensor {
  @derivative(of: maxPool2d, wrt: self)
  @usableFromInline
  internal func _vjpMaxPool2d(
    kernelSize: [Int64],
    stride: [Int64],
    padding: [Int64],
    dilation: [Int64],
    ceilMode: Bool
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let (output, indices) = _maxPool2dWithIndices(
      kernelSize: kernelSize, stride: stride, padding: padding, dilation: dilation,
      ceilMode: ceilMode
    )
    return (
      output,
      { upstream in
        self._maxPool2dBackward(
          upstream: upstream,
          kernelSize: kernelSize,
          stride: stride,
          padding: padding,
          dilation: dilation,
          ceilMode: ceilMode,
          indices: indices
        )
      }
    )
  }

  @derivative(of: avgPool2d, wrt: self)
  @usableFromInline
  internal func _vjpAvgPool2d(
    kernelSize: [Int64],
    stride: [Int64],
    padding: [Int64],
    ceilMode: Bool
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let output = avgPool2d(
      kernelSize: kernelSize, stride: stride, padding: padding, ceilMode: ceilMode
    )
    return (
      output,
      { upstream in
        self._avgPool2dBackward(
          upstream: upstream,
          kernelSize: kernelSize,
          stride: stride,
          padding: padding,
          ceilMode: ceilMode
        )
      }
    )
  }
}
