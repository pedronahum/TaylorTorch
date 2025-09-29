import Testing
import _Differentiation

@testable import Torch

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
  let expectedRhs =
    upstream
    .multiplying(lhs)
    .dividing(rhs.multiplying(rhs))
    .negated()
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: subtracting scalar pullback is identity")
func subtractingScalarGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let scalar: Double = 1.5
  let (value, pullback) = valueWithPullback(at: input) { $0.subtracting(scalar) }
  #expect(value.isClose(to: input.subtracting(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = onesLike(value)
  let grad = pullback(upstream)
  #expect(grad.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: multiplying scalar pullback rescales upstream")
func multiplyingScalarGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let scalar: Double = -2.5
  let (value, pullback) = valueWithPullback(at: input) { $0.multiplying(scalar) }
  #expect(value.isClose(to: input.multiplying(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.0, 2.0], shape: [3])
  let grad = pullback(upstream)
  let expected = upstream.multiplying(scalar)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: dividing scalar pullback scales by reciprocal")
func dividingScalarGradientMatchesAnalytic() throws {
  let input = Tensor(array: [4.0, -8.0, 12.0], shape: [3])
  let scalar: Double = 4.0
  let (value, pullback) = valueWithPullback(at: input) { $0.dividing(scalar) }
  #expect(value.isClose(to: input.dividing(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, 0.5, -2.0], shape: [3])
  let grad = pullback(upstream)
  let expected = upstream.dividing(scalar)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: pow scalar pullback matches analytic derivative")
func powScalarGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.5, 2.0, 2.5], shape: [3])
  let power: Double = 3.0
  let (value, pullback) = valueWithPullback(at: input) { $0.pow(power) }
  #expect(value.isClose(to: input.pow(power), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.0, 1.5], shape: [3])
  let grad = pullback(upstream)
  let expected = upstream.multiplying(power).multiplying(input.pow(power - 1))
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: pow tensor pullback matches analytic gradients")
func powTensorGradientMatchesAnalytic() throws {
  let base = Tensor(array: [2.0, 3.0], shape: [2])
  let exponent = Tensor(array: [3.0, 2.0], shape: [2])
  let (value, pullback) = valueWithPullback(at: base, exponent) { base, exponent in
    base.pow(exponent)
  }
  #expect(value.isClose(to: base.pow(exponent), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.5], shape: [2])
  let (gradBase, gradExponent) = pullback(upstream)
  let expectedBase = upstream.multiplying(exponent).multiplying(base.pow(exponent.subtracting(1)))
  let expectedExponent = upstream.multiplying(value).multiplying(base.log())
  #expect(gradBase.isClose(to: expectedBase, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradExponent.isClose(to: expectedExponent, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: clamp pullback masks clipped values")
func clampGradientMasksClampedValues() throws {
  let input = Tensor(array: [-1.0, 0.5, 2.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: input) { $0.clamp(min: 0.0, max: 1.0) }
  #expect(
    value.isClose(to: input.clamp(min: 0.0, max: 1.0), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, 2.0, -3.0], shape: [3])
  let grad = pullback(upstream)
  let expected = Tensor(array: [0.0, 2.0, 0.0], shape: [3])
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: sum pullback broadcasts upstream")
func sumGradientBroadcastsUpstream() throws {
  let input = Tensor.arange(Double(1), to: Double(5), step: 1).reshaped([2, 2])
  let (value, pullback) = valueWithPullback(at: input) { $0.sum() }
  #expect(value.isClose(to: input.sum(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = onesLike(value).multiplying(2.0)
  let grad = pullback(upstream)
  let expected = Tensor.full(2.0, shape: input.shape)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: sum(dim:) pullback expands upstream")
func sumDimGradientExpandsUpstream() throws {
  let input = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])
  let (value, pullback) = valueWithPullback(at: input) { $0.sum(dim: 1, keepdim: false) }
  #expect(
    value.isClose(to: input.sum(dim: 1, keepdim: false), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, -2.0], shape: [2])
  let grad = pullback(upstream)
  let expected = Tensor(array: [1.0, 1.0, -2.0, -2.0], shape: [2, 2])
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: mean pullback distributes upstream")
func meanGradientDistributesUpstream() throws {
  let input = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])
  let (value, pullback) = valueWithPullback(at: input) { $0.mean() }
  #expect(value.isClose(to: input.mean(), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = onesLike(value).multiplying(1.5)
  let grad = pullback(upstream)
  let expected = Tensor.full(1.5 / 4.0, shape: input.shape)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: mean(dim:) pullback expands and scales upstream")
func meanDimGradientExpandsUpstream() throws {
  let input = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])
  let (value, pullback) = valueWithPullback(at: input) { $0.mean(dim: 1, keepdim: true) }
  #expect(
    value.isClose(to: input.mean(dim: 1, keepdim: true), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [2.0, -1.0], shape: [2, 1])
  let grad = pullback(upstream)
  let expected = upstream.expanded(to: input.shape).dividing(2.0)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: matmul pullback matches analytic gradients")
func matmulGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3])
  let rhs = Tensor(array: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape: [3, 2])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.matmul(rhs)
  }
  #expect(value.isClose(to: lhs.matmul(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.0, 2.0, 0.0], shape: [2, 2])
  let (gradLhs, gradRhs) = pullback(upstream)
  let expectedLhs = upstream.matmul(rhs.transposed(-1, -2))
  let expectedRhs = lhs.transposed(-1, -2).matmul(upstream)
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: dot pullback matches analytic gradients")
func dotGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let rhs = Tensor(array: [-1.0, 0.5, 2.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.dot(rhs)
  }
  #expect(value.isClose(to: lhs.dot(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = onesLike(value).multiplying(3.0)
  let (gradLhs, gradRhs) = pullback(upstream)
  let expectedLhs = upstream.multiplying(rhs)
  let expectedRhs = upstream.multiplying(lhs)
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: tensor comparisons have zero pullback")
func tensorComparisonsProduceZeroGradient() throws {
  let lhs = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let rhs = Tensor(array: [1.5, 2.0, 2.5], shape: [3])
  let upstream = Tensor.ones(shape: [3], dtype: .float32)
  let zero = Tensor.zeros(shape: lhs.shape, dtype: lhs.dtype!)

  do {
    let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in lhs.eq(rhs) }
    #expect(value.equal(lhs.eq(rhs)))
    let (gradLhs, gradRhs) = pullback(upstream)
    #expect(gradLhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
    #expect(gradRhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in lhs.lt(rhs) }
    #expect(value.equal(lhs.lt(rhs)))
    let (gradLhs, gradRhs) = pullback(upstream)
    #expect(gradLhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
    #expect(gradRhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in lhs.le(rhs) }
    #expect(value.equal(lhs.le(rhs)))
    let (gradLhs, gradRhs) = pullback(upstream)
    #expect(gradLhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
    #expect(gradRhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in lhs.gt(rhs) }
    #expect(value.equal(lhs.gt(rhs)))
    let (gradLhs, gradRhs) = pullback(upstream)
    #expect(gradLhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
    #expect(gradRhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in lhs.ge(rhs) }
    #expect(value.equal(lhs.ge(rhs)))
    let (gradLhs, gradRhs) = pullback(upstream)
    #expect(gradLhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
    #expect(gradRhs.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}

@Test("Differentiation: tensor-scalar comparisons have zero pullback")
func tensorScalarComparisonsProduceZeroGradient() throws {
  let input = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let upstream = Tensor.ones(shape: [3], dtype: .float32)
  let zero = Tensor.zeros(shape: input.shape, dtype: input.dtype!)

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0.eq(2.0) }
    #expect(value.equal(input.eq(2.0)))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0.lt(2.0) }
    #expect(value.equal(input.lt(2.0)))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0.le(2.0) }
    #expect(value.equal(input.le(2.0)))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0.gt(2.0) }
    #expect(value.equal(input.gt(2.0)))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0.ge(2.0) }
    #expect(value.equal(input.ge(2.0)))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: zero, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}

@Test("Differentiation: where pullback respects mask")
func whereGradientRespectsMask() throws {
  let condition = tensor([true, false, true], shape: [3])
  let a = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let b = Tensor(array: [4.0, 5.0, 6.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: a, b) { a, b in
    TorchWhere.select(condition: condition, a, b)
  }
  #expect(
    value.isClose(
      to: TorchWhere.select(condition: condition, a, b), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.0, 2.0], shape: [3])
  let (gradA, gradB) = pullback(upstream)
  let mask = condition.to(dtype: a.dtype!)
  let ones = Tensor.ones(shape: mask.shape, dtype: mask.dtype!)
  let inverseMask = ones.subtracting(mask)
  let expectedGradA = upstream.multiplying(mask)
  let expectedGradB = upstream.multiplying(inverseMask)
  #expect(gradA.isClose(to: expectedGradA, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradB.isClose(to: expectedGradB, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
