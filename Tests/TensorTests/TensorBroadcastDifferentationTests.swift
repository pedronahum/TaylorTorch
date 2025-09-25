import Testing
import _Differentiation

@testable import ATen

@Test("Differentiation: expanded pullbacks reduce broadcast dimensions")
func expandedGradientReducesBroadcastDimensions() throws {
  let input = Tensor(array: [1.0, 2.0, 3.0], shape: [1, 3])

  let (expandedValue, expandedPullback) = valueWithPullback(at: input) { tensor in
    tensor.expanded(to: [2, 3], implicit: false)
  }
  #expect(expandedValue.isClose(to: input.expanded(to: [2, 3]), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, -2.0, 3.0, 4.0, -5.0, 6.0], shape: [2, 3])
  let gradExpanded = expandedPullback(upstream)
  let expectedExpanded = upstream.sum(dim: 0, keepdim: true)
  #expect(gradExpanded.isClose(to: expectedExpanded, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let reference = Tensor.zeros(shape: [2, 3], dtype: .float64)
  let (expandedAsValue, expandedAsPullback) = valueWithPullback(at: input) { tensor in
    tensor.expanded(as: reference)
  }
  #expect(expandedAsValue.isClose(to: input.expanded(as: reference), rtol: 1e-6, atol: 1e-6, equalNan: false))
  let gradExpandedAs = expandedAsPullback(upstream)
  #expect(gradExpandedAs.isClose(to: expectedExpanded, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: broadcasted pullback sums broadcast axis")
func broadcastedGradientSumsBroadcastAxis() throws {
  let input = Tensor(array: [1.0, -1.0], shape: [2, 1])
  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.broadcasted(to: [2, 3])
  }
  #expect(value.isClose(to: input.broadcasted(to: [2, 3]), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, 2.0, 3.0, -1.0, -2.0, -3.0], shape: [2, 3])
  let grad = pullback(upstream)
  let expected = upstream.sum(dim: 1, keepdim: true)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

