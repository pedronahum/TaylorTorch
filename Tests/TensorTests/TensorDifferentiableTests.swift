import Testing
import _Differentiation

@testable import ATen

private func onesLike(_ tensor: Tensor) -> Tensor {
  Tensor.ones(shape: tensor.shape, dtype: tensor.dtype!)
}

@Test("Differentiation: adding with alpha scales other gradient")
func addingWithAlphaGradientMatchesAnalytic() throws {
  let lhs = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])
  let rhs = Tensor(array: [-1.0, 0.5, -0.5, 2.0], shape: [2, 2])
  let alpha = Scalar.double(0.5)
  let (value, pullback) = valueWithPullback(at: lhs, rhs) { lhs, rhs in
    lhs.adding(rhs, alpha: alpha)
  }
  #expect(value.isClose(to: lhs.adding(rhs, alpha: alpha), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.5, -1.0, 1.5, 2.0], shape: [2, 2])
  let (gradLhs, gradRhs) = pullback(upstream)
  #expect(gradLhs.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
  let expectedRhs = upstream.multiplying(0.5)
  #expect(gradRhs.isClose(to: expectedRhs, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: adding scalar pullback is identity")
func addingScalarGradientMatchesAnalytic() throws {
  let input = Tensor(array: [1.0, -2.0, 3.5], shape: [3])
  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.adding(2.25)
  }
  #expect(value.isClose(to: input.adding(2.25), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = onesLike(value).multiplying(3.0)
  let grad = pullback(upstream)
  #expect(grad.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

