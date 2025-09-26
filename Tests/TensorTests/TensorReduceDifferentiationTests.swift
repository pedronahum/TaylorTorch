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

@Test("Differentiation: argmax pullback targets winning elements")
func argmaxRoutesGradientToWinners() throws {
  let input = Tensor(
    array: [1.0, 5.0, 3.0, 5.0, 2.0, -1.0, 0.0, 4.0],
    shape: [2, 4]
  )

  let (result, pullback) = valueWithPullback(at: input) { tensor in
    tensor.argmax(dim: 1, keepdim: false)
  }

  let expectedIndices = Tensor(array: [Int64(1), Int64(3)], shape: [2])
  #expect(result == expectedIndices)

  let upstream = Tensor(array: [0.7, -1.2], shape: [2])
  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2],
    shape: [2, 4]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: argmin keepdim scatter matches minima")
func argminKeepdimRoutesGradient() throws {
  let input = Tensor(array: [3.0, -4.0, 6.0, -2.0], shape: [2, 2])

  let (result, pullback) = valueWithPullback(at: input) { tensor in
    tensor.argmin(dim: 1, keepdim: true)
  }

  let expectedIndices = Tensor(array: [Int64(1), Int64(1)], shape: [2, 1])
  #expect(result == expectedIndices)

  let upstream = Tensor(array: [2.0, -3.5], shape: [2, 1])
  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [0.0, 2.0, 0.0, -3.5],
    shape: [2, 2]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: sort pullback restores original order")
func sortPullbackRestoresOrder() throws {
  let input = Tensor(array: [3.0, 1.0, 2.0, 4.0, 0.0, 5.0], shape: [2, 3])

  let (pair, pullback) = valueWithPullback(at: input) { tensor in
    tensor.sort(dim: 1, descending: false)
  }

  let expectedValues = Tensor(array: [1.0, 2.0, 3.0, 0.0, 4.0, 5.0], shape: [2, 3])
  let expectedIndices = Tensor(
    array: [Int64(1), Int64(2), Int64(0), Int64(1), Int64(0), Int64(2)],
    shape: [2, 3]
  )
  #expect(pair.values.isClose(to: expectedValues, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(pair.indices == expectedIndices)

  // --- FIX START ---
  // The TangentVector is just a Tensor, so create it directly.
  let upstream = Tensor(array: [0.1, 0.2, 0.3, -0.5, 0.4, 0.0], shape: [2, 3])
  // --- FIX END ---

  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [0.3, 0.1, 0.2, 0.4, -0.5, 0.0],
    shape: [2, 3]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-5, equalNan: false))
}

@Test("Differentiation: topk pullback scatters to selected elements")
func topkPullbackScattersSelections() throws {
  let input = Tensor(array: [1.0, 5.0, 2.0, 4.0, 3.0, 0.0, 8.0, 7.0], shape: [2, 4])

  let (pair, pullback) = valueWithPullback(at: input) { tensor in
    tensor.topk(2, dim: 1, largest: true, sorted: true)
  }

  let expectedValues = Tensor(array: [5.0, 4.0, 8.0, 7.0], shape: [2, 2])
  let expectedIndices = Tensor(array: [Int64(1), Int64(3), Int64(2), Int64(3)], shape: [2, 2])
  #expect(pair.values.isClose(to: expectedValues, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(pair.indices == expectedIndices)

  // --- FIX START ---
  // The TangentVector is just a Tensor, so create it directly.
  let upstream = Tensor(array: [1.0, -0.5, 0.25, -1.25], shape: [2, 2])
  // --- FIX END ---

  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.25, -1.25],
    shape: [2, 4]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
