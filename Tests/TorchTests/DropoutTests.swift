// Tests/TorchTests/DropoutTests.swift
import Testing
import _Differentiation

@testable import Torch

private func onesLike(_ t: Tensor) -> Tensor {
  Tensor.ones(shape: t.shape, dtype: t.dtype!)
}

@Test("Dropout: inference path is identity (value & gradient)")
func dropoutInferenceIsIdentity() throws {
  let x = Tensor(array: [0.5, -1.0, 2.0, -3.0], shape: [4])
  var layer = Dropout(rate: 0.75)  // any rate; inference path is no-op

  // Forward
  let y = layer(x)
  #expect(y.isClose(to: x, rtol: 0, atol: 0, equalNan: false))

  // Backward
  let (value, pb) = valueWithPullback(at: x) { input in layer(input) }
  #expect(value.isClose(to: x, rtol: 0, atol: 0, equalNan: false))
  let upstream = Tensor(array: [1.0, 2.0, -1.5, 0.25], shape: [4])
  let grad = pb(upstream)
  #expect(grad.isClose(to: upstream, rtol: 0, atol: 0, equalNan: false))
}

@Test("Dropout: training applies mask & inverted scaling (deterministic mask)")
func dropoutTrainingAppliesMaskAndScaling() throws {
  // 2x4 tensor so we can see a pattern
  let x = Tensor(
    array: [
      1.0, 2.0, 3.0, 4.0,
      -1.0, -2.0, -3.0, -4.0,
    ], shape: [2, 4]
  )
  let rate = 0.5
  let keep = 1.0 - rate

  // Deterministic mask for testing: T F T F | F T T F
  let maskBool = Tensor(
    array: [
      true, false, true, false,
      false, true, true, false,
    ],
    shape: [2, 4]
  )

  let layer = Dropout(rate: rate, maskFactory: { _ in maskBool })
  let ctx = ForwardContext(training: true)

  // Forward
  let y = layer.call(x, context: ctx)
  let expected = x.multiplying(maskBool.to(dtype: x.dtype!)).dividing(keep)
  #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))

  // Backward: dy/dx = mask / keep
  let (value, pb) = valueWithPullback(at: x) { input in layer.call(input, context: ctx) }
  #expect(value.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))

  let upstream = Tensor(
    array: [0.5, -1.0, 1.5, 0.0, -0.25, 2.0, -2.5, 1.0],
    shape: [2, 4]
  )
  let grad = pb(upstream)
  let expectedGrad = upstream.multiplying(maskBool.to(dtype: x.dtype!)).dividing(keep)
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Dropout: training edge cases (rate 0 and 1)")
func dropoutTrainingEdgeCases() throws {
  let x = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let ctx = ForwardContext(training: true)

  // rate = 0 -> identity
  do {
    let layer = Dropout(rate: 0.0)
    let (y, pb) = valueWithPullback(at: x) { input in layer.call(input, context: ctx) }
    #expect(y.isClose(to: x, rtol: 0, atol: 0, equalNan: false))
    let upstream = Tensor(array: [0.1, -0.2, 0.3], shape: [3])
    let grad = pb(upstream)
    #expect(grad.isClose(to: upstream, rtol: 0, atol: 0, equalNan: false))
  }

  // rate = 1 -> all zeros, zero gradient
  do {
    let layer = Dropout(rate: 1.0)
    let (y, pb) = valueWithPullback(at: x) { input in layer.call(input, context: ctx) }
    #expect(y.isClose(to: Tensor.zeros(shape: x.shape, dtype: x.dtype!, device: x.device)))
    let upstream = Tensor(array: [0.5, 1.0, -2.0], shape: [3])
    let grad = pb(upstream)
    #expect(grad.isClose(to: Tensor.zeros(shape: x.shape, dtype: x.dtype!, device: x.device)))
  }
}
