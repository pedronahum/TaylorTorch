import Testing
import _Differentiation

@testable import ATen

@Test("Differentiation: select pullback scatters upstream along axis")
func selectPullbackScattersAxis() throws {
  let input = Tensor(
    array: [1.0, 2.0, 3.0,
            4.0, 5.0, 6.0],
    shape: [2, 3]
  )

  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.select(dim: 1, index: Int64(1))
  }

  let expectedValue = Tensor(array: [2.0, 5.0], shape: [2])
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [10.0, -2.0], shape: [2])
  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [0.0, 10.0, 0.0,
            0.0, -2.0, 0.0],
    shape: [2, 3]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: narrow pullback pads zeros outside window")
func narrowPullbackPadsOutsideWindow() throws {
  let input = Tensor(
    array: [1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0],
    shape: [3, 5]
  )

  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.narrow(dim: 1, start: Int64(1), length: Int64(3))
  }

  let expectedValue = Tensor(
    array: [2.0, 3.0, 4.0,
            7.0, 8.0, 9.0,
            12.0, 13.0, 14.0],
    shape: [3, 3]
  )
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(
    array: [1.0, 2.0, 3.0,
            -1.0, -2.0, -3.0,
            4.0, 5.0, 6.0],
    shape: [3, 3]
  )
  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [0.0, 1.0, 2.0, 3.0, 0.0,
            0.0, -1.0, -2.0, -3.0, 0.0,
            0.0, 4.0, 5.0, 6.0, 0.0],
    shape: [3, 5]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: slice pullback scatters using stride")
func slicePullbackHandlesStride() throws {
  let input = Tensor(
    array: [0.0, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0],
    shape: [2, 5]
  )

  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.slice(dim: 1, start: -4, end: nil, step: 2)
  }

  let expectedValue = Tensor(
    array: [1.0, 3.0,
            6.0, 8.0],
    shape: [2, 2]
  )
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.5, 2.0, -2.5], shape: [2, 2])
  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [0.0, 0.5, 0.0, -1.5, 0.0,
            0.0, 2.0, 0.0, -2.5, 0.0],
    shape: [2, 5]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
