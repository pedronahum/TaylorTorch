import Testing

@testable import Torch

private func makeSequentialTensor(count: Int) -> Tensor {
  Tensor.arange(Int64(0), to: Int64(count), step: Int64(1))
}

@Test("Reshape and inference adjust shapes")
func reshapeOperationsWork() throws {
  let tensor = makeSequentialTensor(count: 6).reshaped([2, 3])
  #expect(tensor.shape == [2, 3])

  let inferred = tensor.reshaped(inferring: [3, -1])
  #expect(inferred.shape == [3, 2])
  let flattened = inferred.flattened()
  #expect(flattened.shape == [6])
}

@Test("Transpose, permute, and squeeze operations")
func transposeAndSqueezeBehave() throws {
  let tensor = Tensor.arange(Int64(0), to: Int64(6), step: Int64(1)).reshaped([2, 3])
  let transposed = tensor.transposed(0, 1)
  #expect(transposed.shape == [3, 2])

  let permuted = tensor.unsqueezed(dim: 0).permuted([1, 0, 2])
  #expect(permuted.shape == [2, 1, 3])

  let squeezed = permuted.squeezed()
  #expect(squeezed.shape == [2, 3])
}

@Test("Broadcast and expand produce expected views")
func broadcastingHelpersProduceExpectedShapes() throws {
  let base = Tensor.arange(Double(0), to: Double(3), step: Double(1), dtype: .float64)
  let expanded = base.unsqueezed(dim: 0).expanded(to: [2, 3])
  #expect(expanded.shape == [2, 3])

  let other = Tensor.ones(shape: [2, 3], dtype: .float64)
  let matched = base.broadcasted(to: [2, 3])
  #expect(matched.shape == [2, 3])

  let expandedAs = base.unsqueezed(dim: 0).expanded(as: other)
  #expect(expandedAs.shape == [2, 3])
}

@Test("Unfold creates sliding window views")
func unfoldProducesSlidingWindow() throws {
  let tensor = Tensor.arange(Int64(0), to: Int64(6), step: Int64(1)).reshaped([1, 6])
  let unfolded = tensor.unfolded(dim: 1, size: 3, step: 2)
  #expect(unfolded.shape == [1, 2, 3])
}
