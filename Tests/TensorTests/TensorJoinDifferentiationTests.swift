import Testing
import _Differentiation

@testable import Torch

@Test("Differentiation: cat partitions upstream gradient per operand")
func catGradientPartitionsUpstream() throws {
  let tensors = [
    Tensor(array: [0.0, 1.0, 2.0, 3.0], shape: [2, 2]),
    Tensor(array: [4.0, 5.0], shape: [2, 1]),
    Tensor(array: [6.0, 7.0, 8.0, 9.0, 10.0, 11.0], shape: [2, 3]),
  ]
  let dim = -1

  let (value, pullback) = valueWithPullback(at: tensors) { operands in
    Tensor.cat(operands, dim: dim)
  }

  let expectedValue = Tensor.cat(tensors, dim: dim)
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(
    array: [
      0.5, -1.0, 1.25, 0.0, 2.0, -0.75,
      -0.5, 1.0, -1.5, 2.5, -2.0, 0.5,
    ],
    shape: [2, 6]
  )

  let resolvedDim = _normalizeDimension(dim, rank: tensors[0].rank)
  let tangent = pullback(upstream)
  let grads = tangent.base

  var offset = 0
  for (index, grad) in grads.enumerated() {
    let length = tensors[index].shape[resolvedDim]
    let expectedGrad = upstream.narrow(
      dim: resolvedDim, start: Int64(offset), length: Int64(length))
    #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
    offset += length
  }
}

@Test("Differentiation: stack selects upstream slice for each operand")
func stackGradientSelectsMatchingSlice() throws {
  let tensors = [
    Tensor(array: [0.0, 1.0, 2.0, 3.0], shape: [2, 2]),
    Tensor(array: [4.0, 5.0, 6.0, 7.0], shape: [2, 2]),
    Tensor(array: [8.0, 9.0, 10.0, 11.0], shape: [2, 2]),
  ]
  let dim = -1

  let (value, pullback) = valueWithPullback(at: tensors) { operands in
    Tensor.stack(operands, dim: dim)
  }

  let expectedValue = Tensor.stack(tensors, dim: dim)
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(
    array: [
      0.1, -0.2, 0.3, -0.4, 0.5, -0.6,
      0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
    ],
    shape: [2, 2, 3]
  )

  let resolvedDim = _normalizeDimension(dim, rank: tensors[0].rank + 1)
  let tangent = pullback(upstream)
  let grads = tangent.base

  for (index, grad) in grads.enumerated() {
    let expectedGrad = upstream.select(dim: resolvedDim, index: Int64(index))
    #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}
