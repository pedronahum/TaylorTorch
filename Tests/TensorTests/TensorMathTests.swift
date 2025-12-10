import Foundation
import RealModuleDifferentiable
import Testing

@testable import Torch

@Test("Unary math operations are applied element-wise")
func unaryOperationsBehave() throws {
  let tensor = Tensor.arange(Double(-2), to: Double(3), step: Double(1), dtype: .float64)
  let relu = tensor.relu()
  let reluValues = relu.toArray(as: Double.self)
  #expect(reluValues == [0, 0, 0, 1, 2])

  let absTensor = tensor.abs()
  let absValues = absTensor.toArray(as: Double.self)
  #expect(absValues == [2, 1, 0, 1, 2])

  let expTensor = Tensor.full(1.0, shape: [1]).exp()
  let expValues = expTensor.toArray(as: Double.self)

  // Use RealModuleDifferentiable for proper differentiable exp function
  let value = expValues.first ?? 0.0
  let expected = Double.exp(1.0)  // e^1 using differentiable exp
  let difference = Swift.abs(value - expected)
  #expect(difference < 1e-6)
}

@Test("Binary tensor operations match operator overloads")
func binaryTensorOperationsWork() throws {
  let lhs = Tensor.arange(Double(0), to: Double(4), step: Double(1))
  let rhs = Tensor.full(2.0, shape: [4])

  let sum = lhs + rhs
  let sumValues = sum.toArray(as: Double.self)
  #expect(sumValues == [2, 3, 4, 5])

  let diff = lhs - rhs
  #expect(diff.toArray(as: Double.self) == [-2, -1, 0, 1])

  var mutable = lhs
  mutable += rhs
  #expect(mutable.toArray(as: Double.self) == [2, 3, 4, 5])
}

@Test("Tensor-scalar arithmetic and comparisons work")
func tensorScalarArithmeticAndComparisons() throws {
  let tensor = Tensor.arange(Double(0), to: Double(3), step: Double(1))
  let scaled = tensor * 2.0
  #expect(scaled.toArray(as: Double.self) == [0, 2, 4])

  let comparisons = tensor .< 2.0
  #expect(comparisons.dtype == .bool)
  let mask = comparisons.toArray(as: Bool.self)
  #expect(mask == [true, true, false])

  let flipped = 2.0 - tensor
  #expect(flipped.toArray(as: Double.self) == [2, 1, 0])
}

@Test("Reductions return expected values and indices")
func reductionFunctionsReturnValuesAndIndices() throws {
  let tensor = Tensor.arange(Double(0), to: Double(6), step: Double(1)).reshaped([2, 3])
  let sum = tensor.sum()
  #expect(sum.toArray(as: Double.self) == [15])

  let mean = tensor.mean(dim: 1)
  #expect(mean.shape == [2])
  #expect(mean.toArray(as: Double.self) == [1.0, 4.0])

  let maxResult = tensor.max(dim: 1)
  #expect(maxResult.values.toArray(as: Double.self) == [2, 5])
  #expect(maxResult.indices.toArray(as: Int64.self) == [2, 2])

  let topk = tensor.flattened().topk(3)
  #expect(topk.values.toArray(as: Double.self) == [5, 4, 3])
  #expect(topk.indices.toArray(as: Int64.self) == [5, 4, 3])
}

@Test("TorchWhere selects between tensors")
func torchWhereSelectsValues() throws {
  let condition = tensor([true, false, true], shape: [3])
  let a = Tensor.arange(Double(0), to: Double(3), step: Double(1))
  let b = Tensor.full(9.0, shape: [3])
  let result = TorchWhere.select(condition: condition, a, b)
  #expect(result.toArray(as: Double.self) == [0, 9, 2])
}

@Test("Masked operations respect mask and broadcast")
func maskedOperationsWork() throws {
  let base = Tensor.arange(Double(0), to: Double(4), step: Double(1))
  let mask = tensor([true, false, true, false], shape: [4])
  let filled = base.maskedFill(where: mask, with: -1.0)
  #expect(filled.toArray(as: Double.self) == [-1, 1, -1, 3])

  let tensorFill = tensor([-1.0, -2.0, -3.0, -4.0], shape: [4])
  let filledByTensor = base.maskedFill(where: mask, with: tensorFill)
  #expect(filledByTensor.toArray(as: Double.self) == [-1, 1, -3, 3])

  let selected = base.maskedSelect(where: mask)
  #expect(selected.shape == [2])
  #expect(selected.toArray(as: Double.self) == [0, 2])

  let any = mask.any()
  let all = mask.all()
  #expect(any.toArray(as: Bool.self) == [true])
  #expect(all.toArray(as: Bool.self) == [false])
}

@Test("Equatable and isClose comparisons")
func equatableAndIsCloseWork() throws {
  let lhs = Tensor.arange(Double(0), to: Double(3), step: Double(1))
  let rhs = Tensor.arange(Double(0), to: Double(3), step: Double(1))
  #expect(lhs == rhs)

  let slightlyDifferent = rhs + 1e-7
  #expect(lhs.isClose(to: slightlyDifferent, rtol: 1e-5, atol: 1e-6, equalNan: false))
}
