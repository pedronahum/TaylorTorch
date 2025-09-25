import Testing
import _Differentiation

@testable import ATen

@Test("Differentiation: maskedFill scalar blocks masked gradient")
func maskedFillScalarStopsGradientThroughMaskedEntries() throws {
  let base = Tensor(
    array: [
      0.0, 1.0, 2.0,
      3.0, 4.0, 5.0,
    ],
    shape: [2, 3]
  )
  let mask = tensor([true, false, true], shape: [1, 3])

  let (value, pullback) = valueWithPullback(at: base) { tensor in
    tensor.maskedFill(where: mask, with: 10.0)
  }

  let expectedValue = Tensor(
    array: [
      10.0, 1.0, 10.0,
      10.0, 4.0, 10.0,
    ],
    shape: [2, 3]
  )
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(
    array: [
      0.7, -1.1, 0.3,
      1.5, -0.5, 2.0,
    ],
    shape: [2, 3]
  )
  let grad = pullback(upstream)

  let keepMask =
    mask
    .broadcasted(to: base.shape)
    .to(dtype: base.dtype!)
    .negated()
    .adding(1)
  let expectedGrad = upstream.to(dtype: base.dtype!).multiplying(keepMask)
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: maskedFill tensor partitions gradient between operands")
func maskedFillTensorSplitsGradientBetweenOperands() throws {
  let base = Tensor(
    array: [
      0.0, 1.0, 2.0,
      3.0, 4.0, 5.0,
    ],
    shape: [2, 3]
  )
  let mask = tensor([true, false], shape: [2, 1])
  let replacements = Tensor(array: [-10.0, -20.0, -30.0], shape: [1, 3])

  let (value, pullback) = valueWithPullback(at: base, replacements) { tensor, fills in
    tensor.maskedFill(where: mask, with: fills)
  }

  let expectedValue = Tensor(
    array: [
      -10.0, -20.0, -30.0,
      3.0, 4.0, 5.0,
    ],
    shape: [2, 3]
  )
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(
    array: [
      0.2, -0.4, 0.6,
      -0.8, 1.0, -1.2,
    ],
    shape: [2, 3]
  )

  let (gradBase, gradValues) = pullback(upstream)
  let maskNumeric =
    mask
    .broadcasted(to: base.shape)
    .to(dtype: base.dtype!)
  let keepMask = maskNumeric.negated().adding(1)

  let expectedGradBase = upstream.to(dtype: base.dtype!).multiplying(keepMask)
  #expect(gradBase.isClose(to: expectedGradBase, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let maskedUpstream = upstream.to(dtype: base.dtype!).multiplying(maskNumeric)
  let expectedGradValues = maskedUpstream.sum(dim: 0, keepdim: true)
  #expect(gradValues.isClose(to: expectedGradValues, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
