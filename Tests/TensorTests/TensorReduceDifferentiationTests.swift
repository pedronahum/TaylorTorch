import Testing
import _Differentiation

@testable import ATen

@Test("Differentiation: min reduction splits pullback across ties")
func minReductionPullbackSplitsTies() throws {
  let input = Tensor(array: [3.0, -2.0, 4.0, -2.0], shape: [4])

  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.min()
  }

  let expectedValue = Tensor(-2.0)
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(1.5)
  let grad = pullback(upstream)
  let expectedGrad = Tensor(array: [0.0, 0.75, 0.0, 0.75], shape: [4])
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: max reduction splits pullback across ties")
func maxReductionPullbackSplitsTies() throws {
  let input = Tensor(array: [1.0, 4.0, -3.0, 4.0], shape: [4])

  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.max()
  }

  let expectedValue = Tensor(4.0)
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(-2.0)
  let grad = pullback(upstream)
  let expectedGrad = Tensor(array: [0.0, -1.0, 0.0, -1.0], shape: [4])
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: minimum routes pullback with broadcasting and ties")
func minimumRoutesGradientWithBroadcastingAndTies() throws {
  let lhs = Tensor(array: [1.0, 4.0, 4.0, 2.0], shape: [2, 2])
  let rhs = Tensor(array: [3.0, 2.0], shape: [1, 2])

  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.minimum(rhs)
  }

  let expectedValue = Tensor(array: [1.0, 2.0, 3.0, 2.0], shape: [2, 2])
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, 1.0, 1.5, -2.0], shape: [2, 2])
  let (lhsGrad, rhsGrad) = pullback(upstream)

  let expectedLhsGrad = Tensor(array: [0.5, 0.0, 0.0, -1.0], shape: [2, 2])
  #expect(lhsGrad.isClose(to: expectedLhsGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let expectedRhsGrad = Tensor(array: [1.5, 0.0], shape: [1, 2])
  #expect(rhsGrad.isClose(to: expectedRhsGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: maximum routes pullback with broadcasting and ties")
func maximumRoutesGradientWithBroadcastingAndTies() throws {
  let lhs = Tensor(array: [-1.0, 2.0, 5.0, 2.0], shape: [2, 2])
  let rhs = Tensor(array: [0.0, 2.0], shape: [1, 2])

  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.maximum(rhs)
  }

  let expectedValue = Tensor(array: [0.0, 2.0, 5.0, 2.0], shape: [2, 2])
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, -3.0, 2.0, 4.0], shape: [2, 2])
  let (lhsGrad, rhsGrad) = pullback(upstream)

  let expectedLhsGrad = Tensor(array: [0.0, -1.5, 2.0, 2.0], shape: [2, 2])
  #expect(lhsGrad.isClose(to: expectedLhsGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let expectedRhsGrad = Tensor(array: [1.0, 0.5], shape: [1, 2])
  #expect(rhsGrad.isClose(to: expectedRhsGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
