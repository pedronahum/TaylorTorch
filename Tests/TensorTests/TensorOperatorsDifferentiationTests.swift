import Testing
import _Differentiation

@testable import ATen

private func onesLike(_ tensor: Tensor) -> Tensor {
  Tensor.ones(shape: tensor.shape, dtype: tensor.dtype!)
}

@Test("Differentiation: addition operator reduces gradients to operand shapes")
func additionOperatorGradientReducesToOperandShapes() throws {
  let lhs = Tensor(array: [1.0, -2.0], shape: [2, 1])
  let rhs = Tensor(array: [-3.0, 0.5, 4.0], shape: [1, 3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { $0 + $1 }
  #expect(value.isClose(to: lhs.adding(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.0, 2.0, 1.5, -0.5, 0.25], shape: [2, 3])
  let (gradLhs, gradRhs) = pullback(upstream)
  let expectedLhs = upstream.sum(dim: 1, keepdim: true)
  let expectedRhs = upstream.sum(dim: 0, keepdim: true)
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: subtraction operator pullback matches analytic gradients")
func subtractionOperatorGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let rhs = Tensor(array: [-1.0, 0.5, 2.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { $0 - $1 }
  #expect(value.isClose(to: lhs.subtracting(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.0, 2.0], shape: [3])
  let (gradLhs, gradRhs) = pullback(upstream)
  #expect(gradLhs.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: upstream.negated(), rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: multiplication operator pullback matches analytic gradients")
func multiplicationOperatorGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let rhs = Tensor(array: [4.0, -5.0, 6.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { $0 * $1 }
  #expect(value.isClose(to: lhs.multiplying(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.25, 2.0, -1.5], shape: [3])
  let (gradLhs, gradRhs) = pullback(upstream)
  let expectedLhs = upstream.multiplying(rhs)
  let expectedRhs = upstream.multiplying(lhs)
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: division operator pullback matches analytic gradients")
func divisionOperatorGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [6.0, -8.0, 10.0], shape: [3])
  let rhs = Tensor(array: [3.0, -4.0, 2.5], shape: [3])
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { $0 / $1 }
  #expect(value.isClose(to: lhs.dividing(rhs), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, 0.5, -1.5], shape: [3])
  let (gradLhs, gradRhs) = pullback(upstream)
  let expectedLhs = upstream.dividing(rhs)
  let expectedRhs = upstream.multiplying(lhs).dividing(rhs.multiplying(rhs)).negated()
  #expect(gradLhs.isClose(to: expectedLhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: tensor-scalar addition and subtraction operators propagate upstream")
func tensorScalarAdditionSubtractionGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let scalar: Double = 2.5
  let upstream = Tensor(array: [0.5, -1.0, 2.0], shape: [3])

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0 + scalar }
    #expect(value.isClose(to: input.adding(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0 - scalar }
    #expect(value.isClose(to: input.subtracting(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}

@Test("Differentiation: tensor-scalar scaling operators match analytic gradients")
func tensorScalarScalingGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let scalar: Double = -1.5
  let upstream = Tensor(array: [0.25, -0.5, 2.0], shape: [3])

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0 * scalar }
    #expect(value.isClose(to: input.multiplying(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    let expected = upstream.multiplying(scalar)
    #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { $0 / scalar }
    #expect(value.isClose(to: input.dividing(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    let expected = upstream.dividing(scalar)
    #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}

@Test("Differentiation: scalar-tensor addition and subtraction operators propagate upstream")
func scalarTensorAdditionSubtractionGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let scalar: Double = -0.75
  let upstream = Tensor(array: [1.0, 0.5, -1.5], shape: [3])

  do {
    let (value, pullback) = valueWithPullback(at: input) { scalar + $0 }
    #expect(value.isClose(to: input.adding(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    #expect(grad.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { scalar - $0 }
    #expect(value.isClose(to: scalar - input, rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    let expected = upstream.negated()
    #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}

@Test("Differentiation: scalar-tensor scaling operators match analytic gradients")
func scalarTensorScalingGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, -2.0, 4.0], shape: [3])
  let scalar: Double = 2.0
  let upstream = Tensor(array: [0.5, -1.0, 1.5], shape: [3])

  do {
    let (value, pullback) = valueWithPullback(at: input) { scalar * $0 }
    #expect(value.isClose(to: input.multiplying(scalar), rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    let expected = upstream.multiplying(scalar)
    #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  do {
    let (value, pullback) = valueWithPullback(at: input) { scalar / $0 }
    #expect(value.isClose(to: Tensor(scalar).dividing(input), rtol: 1e-6, atol: 1e-6, equalNan: false))
    let grad = pullback(upstream)
    let expected = upstream.multiplying(scalar).dividing(input.multiplying(input)).negated()
    #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}
