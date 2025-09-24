import Testing
@testable import ATen

// ✅ 1. Correct way to define a custom tag
extension Tag {
    @Tag static var factory: Self
}

@Test("Zeros factory produces correct shape and dtype")
func zerosFactoryProducesExpectedTensor() throws {
  let tensor = Tensor.zeros(shape: [2, 3], dtype: .float32)
  #expect(tensor.shape == [2, 3])
  #expect(tensor.dtype == .float32)
  let values = tensor.toArray(as: Float.self)
  #expect(values.count == 6)
  #expect(values.allSatisfy { $0 == 0 })
}

// ✅ 2. Use the new static property for the tag
@Test("Ones factory produces ones", .tags(.factory))
func onesFactoryProducesOnes() throws {
  let tensor = Tensor.ones(shape: [4], dtype: .float64)
  #expect(tensor.shape == [4])
  #expect(tensor.dtype == .float64)
  let values = tensor.toArray(as: Double.self)
  #expect(values == Array(repeating: 1.0, count: 4))
}

@Test("Full factory broadcasts scalar")
func fullFactoryBroadcastsScalar() throws {
  let tensor = Tensor.full(5.0, shape: [2, 2])
  #expect(tensor.shape == [2, 2])
  #expect(tensor.dtype == .float64)
  let values = tensor.toArray(as: Double.self)
  #expect(values == Array(repeating: 5.0, count: 4))
}

@Test("Variadic factories forward to array-based versions")
func variadicFactoriesForward() throws {
  let zeros = Tensor.zeros(2, 2)
  #expect(zeros.shape == [2, 2])
  let ones = Tensor.ones(3, 1, dtype: .float32)
  #expect(ones.shape == [3, 1])
  let full = Tensor.full(3.0, 2, 2, device: .cpu)
  #expect(full.shape == [2, 2])
  let values = full.toArray(as: Double.self)
  #expect(values.allSatisfy { $0 == 3.0 })
}

@Test("Tensor literal convenience functions build tensors")
func literalHelpersBuildTensors() throws {
  // ✅ 3. Rename local variable 'tensor' to 't1' to avoid conflict
  let t1 = tensor([1, 2, 3], shape: [3])
  #expect(t1.shape == [3])
  #expect(t1.dtype == .int64)
  let values = t1.toArray(as: Int64.self)
  #expect(values == [1, 2, 3])

  let vararg = tensor([Float](repeating: 0.5, count: 4), 2, 2)
  #expect(vararg.shape == [2, 2])
  let floats = vararg.toArray(as: Float.self)
  #expect(floats == [0.5, 0.5, 0.5, 0.5])
}

@Test("Arange and linspace cover numeric ranges")
func rangeFactoriesProduceExpectedValues() throws {
  let ints = Tensor.arange(Int32(0), to: Int32(6), step: Int32(2))
  #expect(ints.shape == [3])
  let intValues = ints.toArray(as: Int32.self)
  #expect(intValues == [0, 2, 4])

  let floats = Tensor.arange(Double(0), to: Double(3), step: Double(0.5), dtype: .float64)
  let floatValues = floats.toArray(as: Double.self)
  #expect(floatValues.count == 6)
  #expect(abs(floatValues.first ?? 1) < 1e-9)
  #expect(abs((floatValues.last ?? 0) - 2.5) < 1e-9)

  let lin = Tensor.linspace(start: 0.0, end: 1.0, steps: 5)
  let linValues = lin.toArray(as: Float.self)
  let expected: [Float] = [0, 0.25, 0.5, 0.75, 1.0]
  #expect(linValues.count == expected.count)
  for (lhs, rhs) in zip(linValues, expected) {
    #expect(abs(lhs - rhs) < 1e-5)
  }
}

@Test("Random factories respect requested shape")
func randomFactoriesReturnCorrectShape() throws {
  let rand = Tensor.rand(shape: [2, 3])
  #expect(rand.shape == [2, 3])
  #expect(rand.dtype == .float32)

  let randn = Tensor.randn(shape: [4], dtype: .float64)
  #expect(randn.shape == [4])
  #expect(randn.dtype == .float64)
}
