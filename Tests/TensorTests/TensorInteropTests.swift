import Testing
@testable import ATen

@Test("Host buffer borrowing falls back to copies when needed")
func hostBufferAccessWorks() throws {
  let tensor = Tensor.arange(Float(0), to: Float(4), step: Float(1))
  // ✅ Removed 'try'
  let sum: Float = tensor.withHostBuffer(as: Float.self) { buffer in
    #expect(buffer.count == 4)
    return buffer.reduce(0, +)
  }
  #expect(sum == 6)

  // ✅ Removed 'try'
  let result = tensor.withMutableHostBuffer(as: Float.self) { buffer -> Float in
    buffer.indices.forEach { buffer[$0] *= 2 }
    return buffer.reduce(0, +)
  }
  #expect(result.result == 12)
  #expect(result.tensor.toArray(as: Float.self) == [0, 2, 4, 6])
}

@Test("CodableTensor round-trips tensors")
func codableTensorRoundTrips() throws {
  let original = Tensor.arange(Int64(1), to: Int64(5), step: Int64(1))
  let codable = CodableTensor(original)
  let decoded = codable.makeTensor()
  #expect(decoded.shape == original.shape)
  #expect(decoded.dtype == original.dtype)
  #expect(decoded.toArray(as: Int64.self) == original.toArray(as: Int64.self))
}

@Test("Cat and stack combine tensors")
func catAndStackProduceExpectedShapes() throws {
  let a = Tensor.arange(Float(0), to: Float(4), step: Float(1)).reshaped([2, 2])
  let b = Tensor.full(10.0, shape: [2, 2])

  let concatenated = Tensor.cat([a, b], dim: 0)
  #expect(concatenated.shape == [4, 2])

  let stacked = Tensor.stack([a, b], dim: 0)
  #expect(stacked.shape == [2, 2, 2])
}