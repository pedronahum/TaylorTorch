import Foundation
import RealModuleDifferentiable
import Testing
import _Differentiation

@testable import Torch

private func tensorSign(_ tensor: Tensor) -> Tensor {
  let zero = Tensor(0.0, device: tensor.device)
  let dtype = tensor.dtype ?? .float64
  let positive = tensor.gt(zero).to(dtype: dtype)
  let negative = tensor.lt(zero).to(dtype: dtype)
  return positive - negative
}

// Note: `sum` and `mean` tests are already as simple as possible.
@Test("sum reduces all elements and propagates gradient")
func sumForwardAndBackward() throws {
  let tensor = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: tensor) { tensor in
    Torch.sum(tensor)
  }
  #expect(value.isClose(to: Tensor(6.0)))
  let grad = pullback(Tensor(1.0))
  #expect(grad.isClose(to: Tensor(array: [1.0, 1.0, 1.0], shape: [3])))
}

@Test("mean reduces all elements and distributes gradient")
func meanForwardAndBackward() throws {
  let tensor = Tensor(array: [1.0, 2.0, 3.0], shape: [3])
  let (value, pullback) = valueWithPullback(at: tensor) { tensor in
    Torch.mean(tensor)
  }
  #expect(value.isClose(to: Tensor(2.0)))
  let grad = pullback(Tensor(1.0))
  #expect(grad.isClose(to: Tensor(array: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], shape: [3])))
}

// ✅ REFACTORED TESTS BELOW

@Test("l1Loss matches absolute difference sum and gradient sign")
func l1LossForwardAndBackward() throws {
  let predicted = Tensor(array: [0.0, 1.0, 2.0], shape: [3])
  let expected = Tensor(array: [1.0, 0.0, 3.0], shape: [3])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    l1Loss(predicted: predicted, expected: expected)
  }

  let expectedForward = abs(expected - predicted).sum()
  #expect(value.isClose(to: expectedForward))

  let grad = pullback(Tensor(1.0))
  let diff = predicted - expected
  let expectedGrad = tensorSign(diff)
  #expect(grad.isClose(to: expectedGrad))
}

@Test("l2Loss matches squared error sum and gradient scale")
func l2LossForwardAndBackward() throws {
  let predicted = Tensor(array: [0.0, 1.0, 2.0], shape: [3])
  let expected = Tensor(array: [1.0, 0.0, 3.0], shape: [3])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    l2Loss(predicted: predicted, expected: expected)
  }

  let diff = predicted - expected
  let expectedForward = power(diff, 2).sum()
  #expect(value.isClose(to: expectedForward))

  let grad = pullback(Tensor(1.0))
  let expectedGrad = 2.0 * diff
  #expect(grad.isClose(to: expectedGrad))
}

@Test("meanAbsoluteError averages absolute difference and gradient sign")
func meanAbsoluteErrorForwardAndBackward() throws {
  let predicted = Tensor(array: [0.0, 1.0, 2.0], shape: [3])
  let expected = Tensor(array: [1.0, 0.0, 3.0], shape: [3])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    meanAbsoluteError(predicted: predicted, expected: expected)
  }

  let expectedForward = abs(expected - predicted).mean()
  #expect(value.isClose(to: expectedForward))

  let grad = pullback(Tensor(1.0))
  let diff = predicted - expected
  let n = Double(predicted.count)
  let expectedGrad = tensorSign(diff) / n
  #expect(grad.isClose(to: expectedGrad))
}

@Test("meanSquaredError averages squared error and gradient scale")
func meanSquaredErrorForwardAndBackward() throws {
  let predicted = Tensor(array: [0.0, 1.0, 2.0], shape: [3])
  let expected = Tensor(array: [1.0, 0.0, 3.0], shape: [3])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    meanSquaredError(predicted: predicted, expected: expected)
  }

  let diff = predicted - expected
  let expectedForward = power(diff, 2).mean()
  #expect(value.isClose(to: expectedForward))

  let grad = pullback(Tensor(1.0))
  let n = Double(predicted.count)
  let expectedGrad = 2.0 * diff / n
  #expect(grad.isClose(to: expectedGrad))
}

@Test("meanSquaredLogarithmicError matches definition and gradient")
func meanSquaredLogarithmicErrorForwardAndBackward() throws {
  let predicted = Tensor(array: [1.0, 3.0], shape: [2])
  let expected = Tensor(array: [2.0, 4.0], shape: [2])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    meanSquaredLogarithmicError(predicted: predicted, expected: expected)
  }

  let expectedForward = power(log(expected + 1) - log(predicted + 1), 2).mean()
  #expect(value.isClose(to: expectedForward))

  let grad = pullback(Tensor(1.0))
  let n = Double(predicted.count)
  let expectedGrad = -2.0 / n * (log(expected + Tensor(1.0)) - log(predicted + Tensor(1.0))) / (predicted + Tensor(1.0))
  #expect(grad.isClose(to: expectedGrad))
}

@Test("meanAbsolutePercentageError scales relative absolute error")
func meanAbsolutePercentageErrorForwardAndBackward() throws {
  let predicted = Tensor(array: [8.0, 10.0], shape: [2])
  let expected = Tensor(array: [10.0, 5.0], shape: [2])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    meanAbsolutePercentageError(predicted: predicted, expected: expected)
  }

  let expectedForward = (Tensor(100.0) * abs(expected - predicted) / abs(expected)).mean()
  #expect(value.isClose(to: expectedForward))

  let grad = pullback(Tensor(1.0))
  let n = Double(predicted.count)
  let diff = predicted - expected
  let expectedGrad = Tensor(100.0) / Tensor(n) * tensorSign(diff) / abs(expected)
  #expect(grad.isClose(to: expectedGrad))
}

@Test("hingeLoss applies margin violation penalty")
func hingeLossForwardAndBackward() throws {
  let predicted = Tensor(array: [0.5, 0.2, 2.0], shape: [3])
  let expected = Tensor(array: [-1.0, 1.0, 1.0], shape: [3])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    hingeLoss(predicted: predicted, expected: expected)
  }

  let margin = Tensor(1.0) - expected * predicted
  let expectedForward = maximum(Tensor(0.0), margin).mean()
  #expect(value.isClose(to: expectedForward))

  // Gradient calculation is clearer with the original Swift loop.
  let grad = pullback(Tensor(1.0))
  let n = Double(predicted.count)
  let hingeValues = zip([0.5, 0.2, 2.0], [-1.0, 1.0, 1.0]).map { (p, e) -> Double in
    max(0.0, 1.0 - e * p)
  }
  let gradValues = zip(hingeValues, [-1.0, 1.0, 1.0]).map { (hinge, e) -> Double in
    hinge > 0.0 ? (-e) / n : 0.0
  }
  let expectedGrad = Tensor(array: gradValues, shape: [3])
  #expect(grad.isClose(to: expectedGrad))
}

@Test("squaredHingeLoss squares violations and scales gradient")
func squaredHingeLossForwardAndBackward() throws {
  let predicted = Tensor(array: [0.5, 0.2, 2.0], shape: [3])
  let expected = Tensor(array: [-1.0, 1.0, 1.0], shape: [3])

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    squaredHingeLoss(predicted: predicted, expected: expected)
  }

  let margin = Tensor(1.0) - expected * predicted
  let expectedForward = power(maximum(Tensor(0.0), margin), 2).mean()
  #expect(value.isClose(to: expectedForward))

  // Gradient calculation is clearer with the original Swift loop.
  let grad = pullback(Tensor(1.0))
  let n = Double(predicted.count)
  let hingeValues = zip([0.5, 0.2, 2.0], [-1.0, 1.0, 1.0]).map { (p, e) -> Double in
    max(0.0, 1.0 - e * p)
  }
  let gradValues = zip(hingeValues, [-1.0, 1.0, 1.0]).map { (hinge, e) -> Double in
    hinge > 0.0 ? (2.0 * hinge * (-e)) / n : 0.0
  }
  let expectedGrad = Tensor(array: gradValues, shape: [3])
  #expect(grad.isClose(to: expectedGrad))
}

// The remaining tests (categoricalHinge, softplus, etc.) are already well-structured
// or their plain-Swift calculations are clearer for validation. No changes needed.
@Test("categoricalHingeLoss enforces margin between classes")
func categoricalHingeLossForwardAndBackward() throws {
  let predictedValues: [[Double]] = [[0.2, 0.5, -0.1], [-0.2, 0.1, 0.4]]
  let expectedValues: [[Double]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
  let predicted = Tensor(array: predictedValues.flatMap { $0 }, shape: [2, 3])
  let expected = Tensor(array: expectedValues.flatMap { $0 }, shape: [2, 3])
  let batch = Double(predictedValues.count)

  let (value, pullback) = valueWithPullback(at: predicted) { predicted in
    categoricalHingeLoss(predicted: predicted, expected: expected)
  }

  let losses = zip(predictedValues, expectedValues).map { (pRow, eRow) -> Double in
    let positive = zip(pRow, eRow).reduce(0.0) { $0 + $1.0 * $1.1 }
    let masked = zip(pRow, eRow).map { (p, e) -> Double in (1.0 - e) * p }
    let negative = masked.max() ?? 0.0
    return max(0.0, negative - positive + 1.0)
  }
  let expectedForward = losses.reduce(0.0, +) / batch
  #expect(value.isClose(to: Tensor(expectedForward)))

  let grad = pullback(Tensor(1.0))
  var gradArray = Array(repeating: Array(repeating: 0.0, count: 3), count: 2)
  for (index, (pRow, eRow)) in zip(predictedValues, expectedValues).enumerated() {
    let positive = zip(pRow, eRow).reduce(0.0) { $0 + $1.0 * $1.1 }
    let masked = zip(pRow, eRow).map { (p, e) -> Double in (1.0 - e) * p }
    guard let maxEntry = masked.enumerated().max(by: { $0.element < $1.element }) else { continue }
    let marginLoss = max(0.0, maxEntry.element - positive + 1.0)
    if marginLoss > 0.0 {
      gradArray[index][maxEntry.offset] = 1.0 / batch
      for classIndex in 0..<eRow.count where eRow[classIndex] == 1.0 {
        gradArray[index][classIndex] -= 1.0 / batch
      }
    }
  }
  let expectedGrad = Tensor(array: gradArray.flatMap { $0 }, shape: [2, 3])
  #expect(grad.isClose(to: expectedGrad))
}

@Test("softplus matches smooth ReLU and gradient is sigmoid")
func softplusForwardAndBackward() throws {
  let values = [-1.0, 0.0, 2.0]
  let tensor = Tensor(array: values, shape: [3])

  let (value, pullback) = valueWithPullback(at: tensor) { tensor in
    softplus(tensor)
  }

  // Use RealModuleDifferentiable for proper differentiable math functions
  let expectedForward = values.map { v -> Double in Double.log(onePlus: Double.exp(v)) }
  let expectedTensor = Tensor(array: expectedForward, shape: [3])
  #expect(value.isClose(to: expectedTensor, rtol: 1e-6, atol: 1e-6))

  let grad = pullback(Tensor(array: [1.0, 1.0, 1.0], shape: [3]))
  let gradValues = values.map { v -> Double in 1.0 / (1.0 + Double.exp(-v)) }
  let expectedGrad = Tensor(array: gradValues, shape: [3])
  #expect(grad.isClose(to: expectedGrad, rtol: 1e-6, atol: 1e-6))
}
