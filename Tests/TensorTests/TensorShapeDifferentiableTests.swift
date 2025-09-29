import Testing
import _Differentiation

@testable import Torch

private func onesLike(_ tensor: Tensor) -> Tensor {
  Tensor.ones(shape: tensor.shape, dtype: tensor.dtype!)
}

@Test("Differentiation: transposed pullback is transpose")
func transposedGradientMatchesAnalytic() throws {
  let input = Tensor.arange(Double(0), to: Double(6), step: 1, dtype: .float64).reshaped([2, 3])
  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.transposed(0, 1)
  }
  #expect(value.isClose(to: input.transposed(0, 1), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [1.0, -2.0, 3.0, 4.0, -5.0, 6.0], shape: [3, 2])
  let grad = pullback(upstream)
  let expected = upstream.transposed(0, 1)
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: permuted pullback inverts permutation")
func permutedGradientMatchesAnalytic() throws {
  let input = Tensor.arange(Double(0), to: Double(24), step: 1, dtype: .float64).reshaped([2, 3, 4])
  let order = [2, 0, 1]
  let (value, pullback) = valueWithPullback(at: input) { tensor in
    tensor.permuted(order)
  }
  #expect(value.isClose(to: input.permuted(order), rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor.arange(Double(0), to: Double(24), step: 1, dtype: .float64).reshaped([
    4, 2, 3,
  ])
  let grad = pullback(upstream)
  let expected = upstream.permuted([1, 2, 0])
  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: reshape and flatten pullbacks restore original shape")
func shapeChangingGradientsRestoreOriginalShape() throws {
  let input = Tensor.arange(Double(0), to: Double(12), step: 1, dtype: .float64).reshaped([2, 2, 3])

  let (reshapedValue, reshapedPullback) = valueWithPullback(at: input) { tensor in
    tensor.reshaped([3, 4])
  }
  #expect(
    reshapedValue.isClose(to: input.reshaped([3, 4]), rtol: 1e-6, atol: 1e-6, equalNan: false))
  let reshapedUpstream = Tensor.arange(Double(0), to: Double(12), step: 1, dtype: .float64)
    .reshaped([3, 4])
  let gradReshaped = reshapedPullback(reshapedUpstream)
  let expectedReshaped = reshapedUpstream.reshaped([2, 2, 3])
  #expect(gradReshaped.isClose(to: expectedReshaped, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let (flattenedValue, flattenedPullback) = valueWithPullback(at: input) { tensor in
    tensor.flattened(startDim: 1, endDim: 2)
  }
  #expect(
    flattenedValue.isClose(
      to: input.flattened(startDim: 1, endDim: 2), rtol: 1e-6, atol: 1e-6, equalNan: false))
  let flattenedUpstream = Tensor.arange(Double(0), to: Double(12), step: 1, dtype: .float64)
    .reshaped([2, 6])
  let gradFlattened = flattenedPullback(flattenedUpstream)
  let expectedFlattened = flattenedUpstream.reshaped([2, 2, 3])
  #expect(gradFlattened.isClose(to: expectedFlattened, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: squeeze and unsqueeze pullbacks restore axes")
func squeezeUnsqueezeGradientsRestoreAxes() throws {
  let squeezedInput = Tensor.arange(Double(0), to: Double(6), step: 1, dtype: .float64).reshaped([
    1, 2, 1, 3,
  ])
  let (squeezedValue, squeezedPullback) = valueWithPullback(at: squeezedInput) { tensor in
    tensor.squeezed()
  }
  #expect(
    squeezedValue.isClose(to: squeezedInput.squeezed(), rtol: 1e-6, atol: 1e-6, equalNan: false))
  let squeezedUpstream = Tensor(array: [1.0, -2.0, 3.0, 4.0, -5.0, 6.0], shape: [2, 3])
  let gradSqueezed = squeezedPullback(squeezedUpstream)
  let expectedSqueezed = squeezedUpstream.unsqueezed(dim: 0).unsqueezed(dim: 2)
  #expect(gradSqueezed.isClose(to: expectedSqueezed, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let squeezedDimInput = Tensor(array: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 1, 3])
  let (squeezedDimValue, squeezedDimPullback) = valueWithPullback(at: squeezedDimInput) { tensor in
    tensor.squeezed(dim: 1)
  }
  #expect(
    squeezedDimValue.isClose(
      to: squeezedDimInput.squeezed(dim: 1), rtol: 1e-6, atol: 1e-6, equalNan: false))
  let squeezedDimUpstream = Tensor(array: [1.0, -1.0, 2.0, -2.0, 3.0, -3.0], shape: [2, 3])
  let gradSqueezedDim = squeezedDimPullback(squeezedDimUpstream)
  let expectedSqueezedDim = squeezedDimUpstream.unsqueezed(dim: 1)
  #expect(gradSqueezedDim.isClose(to: expectedSqueezedDim, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let unsqueezedInput = Tensor(array: [1.0, -1.0, 2.0, -2.0], shape: [2, 2])
  let (unsqueezedValue, unsqueezedPullback) = valueWithPullback(at: unsqueezedInput) { tensor in
    tensor.unsqueezed(dim: 1)
  }
  #expect(
    unsqueezedValue.isClose(
      to: unsqueezedInput.unsqueezed(dim: 1), rtol: 1e-6, atol: 1e-6, equalNan: false))
  let unsqueezedUpstream = onesLike(unsqueezedValue).multiplying(2.0)
  let gradUnsqueezed = unsqueezedPullback(unsqueezedUpstream)
  let expectedUnsqueezed = unsqueezedUpstream.squeezed(dim: 1)
  #expect(gradUnsqueezed.isClose(to: expectedUnsqueezed, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
