import Testing
import _Differentiation

@testable import Torch

@Test("Differentiation: indexSelect pullback scatters along dim 0")
func indexSelectPullbackScattersAlongDimZero() throws {
  let base = Tensor(
    array: [
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
    ],
    shape: [3, 3]
  )
  let indices: [Int32] = [2, 0]

  let (value, pullback) = valueWithPullback(at: base) { tensor in
    tensor.indexSelect(dim: 0, indices: indices)
  }

  let expectedValue = Tensor(
    array: [
      7.0, 8.0, 9.0,
      1.0, 2.0, 3.0,
    ],
    shape: [2, 3]
  )
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(
    array: [
      0.5, -1.0, 2.0,
      3.5, 0.0, -0.5,
    ],
    shape: [2, 3]
  )
  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [
      3.5, 0.0, -0.5,
      0.0, 0.0, 0.0,
      0.5, -1.0, 2.0,
    ],
    shape: [3, 3]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: indexSelect accumulates repeated indices")
func indexSelectPullbackAccumulatesRepeatedIndices() throws {
  let base = Tensor(array: [0.0, 1.0, 2.0, 3.0], shape: [4])
  let indices: [Int] = [1, 1, 3]

  let (_, pullback) = valueWithPullback(at: base) { tensor in
    tensor.indexSelect(dim: 0, indices: indices)
  }

  let upstream = Tensor(array: [0.25, -0.75, 1.5], shape: [3])
  let grad = pullback(upstream)
  let expectedGrad = Tensor(array: [0.0, -0.5, 0.0, 1.5], shape: [4])
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

// In your test file

@Test("Differentiation: indexSelect normalizes negative indices (with debug prints)")
func indexSelectPullbackHandlesNegativeIndicesDebug() throws {
  let base = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [4])
  let indices: [Int] = [-1, -3]

  let (value, pullback) = valueWithPullback(at: base) { tensor in
    tensor.indexSelect(dim: 0, indices: indices)
  }

  let expectedValue = Tensor(array: [4.0, 2.0], shape: [2])

  // ✅ DEBUG: Print the result of the forward pass
  print("--- Debugging Test ---")
  print("Forward Pass Value: \(value)")
  print("Expected Value:     \(expectedValue)")

  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))
  print("✅ Forward pass is correct.")

  let upstream = Tensor(array: [2.5, -1.25], shape: [2])
  let grad = pullback(upstream)

  let expectedGrad = Tensor(array: [0.0, -1.25, 0.0, 2.5], shape: [4])

  // ✅ DEBUG: Print the result of the backward pass
  print("Backward Pass Grad: \(grad)")
  print("Expected Grad:      \(expectedGrad)")
  print("--- End Debugging ---")

  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

/*
@Test("Differentiation: indexSelect normalizes negative indices")
func indexSelectPullbackHandlesNegativeIndices() throws {
  let base = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [4])
  let indices: [Int] = [-1, -3]

  let (value, pullback) = valueWithPullback(at: base) { tensor in
    tensor.indexSelect(dim: 0, indices: indices)
  }

  let expectedValue = Tensor(array: [4.0, 2.0], shape: [2])
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [2.5, -1.25], shape: [2])
  let grad = pullback(upstream)
  let expectedGrad = Tensor(array: [0.0, -1.25, 0.0, 2.5], shape: [4])
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
*/

@Test("Differentiation: indexSelect pullback respects non-leading dimensions")
func indexSelectPullbackHandlesInnerDimensions() throws {
  let base = Tensor(
    array: [
      0.0, 1.0,
      2.0, 3.0,
      4.0, 5.0,
      6.0, 7.0,
      8.0, 9.0,
      10.0, 11.0,
    ],
    shape: [2, 3, 2]
  )
  let indices: [Int32] = [2, 0]

  let (_, pullback) = valueWithPullback(at: base) { tensor in
    tensor.indexSelect(dim: 1, indices: indices)
  }

  let upstream = Tensor(
    array: [
      0.1, 0.2,
      0.3, 0.4,
      0.5, 0.6,
      0.7, 0.8,
    ],
    shape: [2, 2, 2]
  )
  let grad = pullback(upstream)
  let expectedGrad = Tensor(
    array: [
      0.3, 0.4,
      0.0, 0.0,
      0.1, 0.2,
      0.7, 0.8,
      0.0, 0.0,
      0.5, 0.6,
    ],
    shape: [2, 3, 2]
  )
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: indexSelect empty indices yield zero gradient")
func indexSelectPullbackHandlesEmptyIndices() throws {
  let base = Tensor(
    array: [
      1.0, 2.0,
      3.0, 4.0,
      5.0, 6.0,
    ],
    shape: [3, 2]
  )
  let indices = [Int32]()

  let (value, pullback) = valueWithPullback(at: base) { tensor in
    tensor.indexSelect(dim: 0, indices: indices)
  }

  let expectedValue = Tensor.zeros(shape: [0, 2], dtype: .float64, device: base.device)
  #expect(value.isClose(to: expectedValue, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor.zeros(shape: [0, 2], dtype: .float64, device: base.device)
  let grad = pullback(upstream)
  let expectedGrad = Tensor.zeros(shape: [3, 2], dtype: .float64, device: base.device)
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

// MARK: - 🎯 1. Core Functionality

@Test("Core: Simple 1D Scatter")
func testCoreFunctionality_Simple1D() throws {
  let selfTensor = Tensor.zeros(shape: [4], dtype: .float32)
  let mask = Tensor(array: [true, false, true, false], shape: [4])
  // ✅ FIX: Cast the array to [Float32]
  let source = Tensor(array: [10.0, 20.0] as [Float32], shape: [2])

  let output = selfTensor.maskedScatter(where: mask, source: source)
  let expected = Tensor(array: [10.0, 0.0, 20.0, 0.0] as [Float32], shape: [4])

  #expect(output.isClose(to: expected))
}

@Test("Core: Multi-dimensional Scatter")
func testCoreFunctionality_MultiDimensional() throws {
  let selfTensor = Tensor.zeros(shape: [2, 3], dtype: .int32)
  let mask = Tensor(array: [true, false, true, false, true, false], shape: [2, 3])
  // ✅ FIX: Cast the array to [Int32]
  let source = Tensor(array: [10, 20, 30] as [Int32], shape: [3])

  let output = selfTensor.maskedScatter(where: mask, source: source)
  let expected = Tensor(array: [10, 0, 20, 0, 30, 0] as [Int32], shape: [2, 3])

  #expect(output.isClose(to: expected))
}

// MARK: - 📐 2. Broadcasting Behavior

@Test("Broadcasting: `self` (scalar) to `mask`")
func testBroadcasting_SelfToMask() throws {
  let selfTensor = Tensor(0.0 as Float)  // Use Float for self to match source
  let mask = Tensor(array: [true, false, true, true], shape: [2, 2])
  // ✅ FIX: Cast the array to [Float]
  let source = Tensor(array: [1.0, 2.0, 3.0] as [Float], shape: [3])

  let output = selfTensor.maskedScatter(where: mask, source: source)
  let expected = Tensor(array: [1.0, 0.0, 2.0, 3.0] as [Float], shape: [2, 2])

  #expect(output.isClose(to: expected))
}

@Test("Broadcasting: `mask` (1D) to `self` (2D)")
func testBroadcasting_MaskToSelf() throws {
  let selfTensor = Tensor.zeros(shape: [2, 3], dtype: .float32)
  let mask = Tensor(array: [true, false, true], shape: [3])
  // ✅ FIX: Cast the array to [Float32]
  let source = Tensor(array: [10, 20, 30, 40] as [Float32], shape: [4])

  let output = selfTensor.maskedScatter(where: mask, source: source)
  let expected = Tensor(array: [10, 0, 20, 30, 0, 40] as [Float32], shape: [2, 3])

  #expect(output.isClose(to: expected))
}

// MARK: - 🔢 3. Data Type Handling

@Test("DataTypes: Integer Scatter")
func testDataTypes_Integers() throws {
  // This test was okay because both arrays inferred to the same `Int` (Int64) type.
  let selfTensor = Tensor(array: [0, 0], shape: [2])
  let mask = Tensor(array: [true, false], shape: [2])
  let source = Tensor(array: [-100], shape: [1])
  let output = selfTensor.maskedScatter(where: mask, source: source)
  let expected = Tensor(array: [-100, 0], shape: [2])
  #expect(output.isClose(to: expected))
}

@Test("DataTypes: Mixed Floating-Point Precision")
func testDataTypes_MixedPrecision() throws {
  let selfTensor = Tensor.zeros(shape: [2], dtype: .float32)  // Lower precision
  let mask = Tensor(array: [true, true], shape: [2])
  // This correctly infers [Double], which is different from selfTensor.
  let source = Tensor(array: [1.0, 1.0], shape: [2])

  let output = selfTensor.maskedScatter(where: mask, source: source)

  // The output dtype should match selfTensor, as per the implementation.
  #expect(output.dtype == .float32)
  #expect(output.isClose(to: Tensor.ones(shape: [2], dtype: .float32)))
}

// MARK: - 🧪 4. Edge Cases

@Test("EdgeCases: Mask is All `true`")
func testEdgeCases_MaskAllTrue() throws {
  let selfTensor = Tensor.zeros(shape: [2, 2], dtype: .int32)
  let mask = Tensor.ones(shape: [2, 2], dtype: .bool)
  let source = Tensor.arange(0, to: 4, step: 1).reshaped([4])  // This is Int64

  let output = selfTensor.maskedScatter(where: mask, source: source)
  // ✅ FIX: Cast the array to [Int32] to match output
  let expected = Tensor(array: [0, 1, 2, 3] as [Int32], shape: [2, 2])

  #expect(output.isClose(to: expected))
}

@Test("EdgeCases: Mask is All `false`")
func testEdgeCases_MaskAllFalse() throws {
  let selfTensor = Tensor.ones(shape: [2, 2], dtype: .float32)
  let mask = Tensor.zeros(shape: [2, 2], dtype: .bool)
  let source = Tensor(array: [Float](), shape: [0])

  let output = selfTensor.maskedScatter(where: mask, source: source)
  #expect(output.isClose(to: selfTensor))
}

@Test("EdgeCases: Empty Tensors")
func testEdgeCases_EmptyTensors() throws {
  let selfTensor = Tensor.zeros(shape: [3, 0], dtype: .int32)
  let mask = Tensor.zeros(shape: [3, 0], dtype: .bool)
  let source = Tensor(array: [Int32](), shape: [0])

  let output = selfTensor.maskedScatter(where: mask, source: source)
  #expect(output.shape == [3, 0])
}

/*
@Test("maskedScatter pointwise: mask selects from source else keep self")
func maskedScatterPointwise() throws {
  let self1D = Tensor.arange(Int32(0), to: Int32(5), step: 1, dtype: .int32)
  let mask = Tensor(
    array: [
      false, false, false, true, true,
      true, true, false, true, true,
    ], shape: [2, 5])
  let source = Tensor(array: (0..<10).map(Int32.init), shape: [2, 5])

  let out = self1D.maskedScatter(where: mask, source: source)
  // Here the result equals where(mask, source, self.broadcasted)
  let expected = TorchWhere.select(condition: mask, source, self1D.broadcasted(to: [2, 5]))
  #expect(out.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))
}


@Test("maskedScatter")
func testMaskedScatter() {
  // In PyTorch, the initial tensor `self` is broadcast to the mask's shape.
  // In Swift, we explicitly create a base tensor with the target shape.
  let base = Tensor(
    array: [
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
    ], shape: [2, 5])

  // The boolean mask to select indices for scattering.
  let mask = Tensor([
    [false, false, false, true, true],
    [true, true, false, true, true],
  ])

  // The source tensor providing the values to be scattered.
  let source = Tensor(
    array: [
      [0, 1, 2, 3, 4],
      [5, 6, 7, 8, 9],
    ], shape: [2, 5])

  // To replicate PyTorch's `masked_scatter`, we perform three steps:

  // 1. Find the coordinates of all `true` values in the mask.
  let indices = mask.nonZeroIndices()  // Shape: [6, 2]

  // 2. Flatten the source tensor and take as many elements as there are `true` values.
  let numUpdates = Int(indices.shape[0])
  let updates = source.flattened().prefix(numUpdates)  // [0, 1, 2, 3, 4, 5]

  // 3. Scatter the `updates` into the `base` tensor at the specified `indices`.
  let result = base.scattered(at: indices, with: updates)

  // Define the expected output tensor.
  let expected = Tensor(
    array: [
      [0, 0, 0, 0, 1],
      [2, 3, 0, 4, 5],
    ], shape: [2, 5])

  // Assert that the computed result matches the expected tensor.
  #expect(result.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))
}*/

@Test("Differentiation: indexPut N-D (overwrite/add) gradient")
func indexPutPullback() throws {
  let x = Tensor(array: [0.0, 1.0, 2.0, 3.0], shape: [4])
  let idx = Tensor(array: [1, 3], shape: [2], device: x.device)  // Int64 by default
  let val = Tensor(array: [5.0, 7.0], shape: [2])

  // overwrite
  do {
    let (y, pb) = valueWithPullback(
      at: x, val, of: { a, b in a.indexPut(indices: [idx], values: b, accumulate: false) })
    #expect(
      y.isClose(
        to: Tensor(array: [0.0, 5.0, 2.0, 7.0], shape: [4]), rtol: 0, atol: 0, equalNan: false))
    let upstream = Tensor(array: [0.5, 1.0, -2.0, 3.0], shape: [4])
    let (gx, gv) = pb(upstream)
    let expectedGX = Tensor(array: [0.5, 0.0, -2.0, 0.0], shape: [4])  // zero at overwritten sites
    let expectedGV = Tensor(array: [1.0, 3.0], shape: [2])  // gather
    #expect(gx.isClose(to: expectedGX, rtol: 1e-6, atol: 1e-6, equalNan: false))
    #expect(gv.isClose(to: expectedGV, rtol: 1e-6, atol: 1e-6, equalNan: false))
  }

  // accumulate (add)
  do {
    let (y, pb) = valueWithPullback(
      at: x, val, of: { a, b in a.indexPut(indices: [idx], values: b, accumulate: true) })
    #expect(
      y.isClose(
        to: Tensor(array: [0.0, 6.0, 2.0, 10.0], shape: [4]), rtol: 0, atol: 0, equalNan: false))
    let upstream = Tensor(array: [0.25, -1.0, 2.0, 4.0], shape: [4])
    let (gx, gv) = pb(upstream)
    #expect(gx.isClose(to: upstream, rtol: 1e-6, atol: 1e-6, equalNan: false))  // identity
    #expect(
      gv.isClose(
        to: Tensor(array: [-1.0, 4.0], shape: [2]), rtol: 1e-6, atol: 1e-6, equalNan: false))
  }
}
