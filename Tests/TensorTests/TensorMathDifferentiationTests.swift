import Testing
import _Differentiation

@testable import ATen

private func onesLike(_ tensor: Tensor) -> Tensor {
  Tensor.ones(shape: tensor.shape, dtype: tensor.dtype!)
}

@Test("Differentiation: negated pullback matches analytic gradient")
func negatedGradientMatchesAnalytic() throws {
  let input = Tensor.arange(Double(-2), to: Double(3), step: Double(1), dtype: .float64)
  let (value, pullback) = valueWithPullback(at: input) { $0.negated() }
  #expect(value.isClose(to: input.negated(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let grad = pullback(onesLike(value))
  let expected = Tensor.full(-1.0, shape: input.shape)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: abs pullback tracks sign")
func absGradientProducesSign() throws {
  let input = Tensor(array: [-3.0, -1.5, 0.0, 2.0, 4.5], shape: [5])
  let (value, pullback) = valueWithPullback(at: input) { $0.abs() }
  #expect(value.isClose(to: input.abs(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let grad = pullback(onesLike(value))
  let expected = Tensor(array: [-1.0, -1.0, 0.0, 1.0, 1.0], shape: [5])
  #expect(grad.isClose(to: expected, rtol: 1e-5, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: relu pullback masks negatives")
func reluGradientMasksNegatives() throws {
  let input = Tensor(array: [-2.0, -0.5, 0.0, 1.5, 3.0], shape: [5])
  let (value, pullback) = valueWithPullback(at: input) { $0.relu() }
  #expect(value.isClose(to: input.relu(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let grad = pullback(onesLike(value))
  let expected = Tensor(array: [0.0, 0.0, 0.0, 1.0, 1.0], shape: [5])
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: exp pullback matches output")
func expGradientMatchesOutput() throws {
  let input = Tensor(array: [0.0, 0.5, 1.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: input) { $0.exp() }
  #expect(value.isClose(to: input.exp(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let grad = pullback(onesLike(value))
  #expect(grad.isClose(to: value, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: log pullback is reciprocal")
func logGradientIsReciprocal() throws {
  let input = Tensor(array: [1.0, 2.0, 4.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: input) { $0.log() }
  #expect(value.isClose(to: input.log(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let grad = pullback(onesLike(value))
  let expected = Tensor(array: [1.0, 0.5, 0.25], shape: [3])
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: sqrt pullback matches analytic derivative")
func sqrtGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, 4.0, 9.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: input) { $0.sqrt() }
  #expect(value.isClose(to: input.sqrt(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let grad = pullback(onesLike(value))
  let expected = Tensor(array: [0.5, 0.25, 1.0 / 6.0], shape: [3])
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: subtracting pullback matches analytic gradients")
func subtractingGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let rhs = Tensor(array: [-1.0, 0.5, 2.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.subtracting(rhs)
  }
  #expect(value.isClose(to: lhs.subtracting(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = onesLike(value)
  let (gradLhs, gradRhs) = pullback(upstream)
  #expect(gradLhs.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: upstream.negated(), rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: multiplying pullback matches analytic gradients")
func multiplyingGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let rhs = Tensor(array: [4.0, -5.0, 6.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.multiplying(rhs)
  }
  #expect(value.isClose(to: lhs.multiplying(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, 2.0, -1.5], shape: [3])
  let (gradLhs, gradRhs) = pullback(upstream)
  let expectedLhs = upstream.multiplying(rhs)
  let expectedRhs = upstream.multiplying(lhs)
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: dividing pullback matches analytic gradients")
func dividingGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [6.0, -2.0, 8.0], shape: [3])
  let rhs = Tensor(array: [3.0, -4.0, 2.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.dividing(rhs)
  }
  #expect(value.isClose(to: lhs.dividing(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, 0.5, -1.0], shape: [3])
  let (gradLhs, gradRhs) = pullback(upstream)
  let expectedLhs = upstream.dividing(rhs)
  let expectedRhs = upstream
    .multiplying(lhs)
    .dividing(rhs.multiplying(rhs))
    .negated()
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
