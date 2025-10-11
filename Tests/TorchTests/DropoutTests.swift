// Tests/TorchTests/DropoutTests.swift
import Testing
import _Differentiation

@testable import Torch

@Test("Dropout: inference path is identity (value & gradient)")
func dropoutInferenceIsIdentity() throws {
  let x = Tensor(array: [0.5, -1.0, 2.0, -3.0], shape: [4])
  let layer = Dropout(probability: 0.75)  // any p; inference path is no-op

  // Forward
  let y = layer(x)
  #expect(y.isClose(to: x, rtol: 0, atol: 0, equalNan: false))

  // Backward
  let (value, pb) = valueWithPullback(at: x) { input in layer(input) }
  #expect(value.isClose(to: x, rtol: 0, atol: 0, equalNan: false))
  let upstream = Tensor(array: [1.0, 2.0, -1.5, 0.25], shape: [4])
  let grad = pb(upstream)
  #expect(grad.isClose(to: upstream, rtol: 0, atol: 0, equalNan: false))
}

@Test("Dropout: training invariants (masking + inverted scaling) without test-only hooks")
func dropoutTrainingInvariants() throws {
  // Use many elements to make the kept-fraction concentration tight.
  let n = 10_000
  let x = Tensor.ones(shape: [n], dtype: .float64)
  let p: Float = 0.2
  let keep = 1.0 - p
  let layer = Dropout(probability: p)

  let (_, pb) = withLearningPhase(.training) {
    valueWithPullback(at: x) { input in layer(input).sum() }
  }
  //print("ok so far")
  // Gradient of sum(layer(x)) is mask/keep. Use it to recover dropout statistics.
  let grad = pb(Tensor(1.0, dtype: .float64))
  let y = grad  // for x == 1, forward output equals grad (mask/keep)

  let zero = Tensor(0.0, dtype: .float64)
  let ratio = Tensor(Double(1.0 / keep), dtype: .float64)

  let hasNegative = y.lt(zero).any().toArray(as: Bool.self)[0]
  #expect(!hasNegative)
  let exceedsRatio = y.gt(ratio).any().toArray(as: Bool.self)[0]
  #expect(!exceedsRatio)

  let hasZero = y.eq(zero).any().toArray(as: Bool.self)[0]
  #expect(hasZero)
  let hasRatio = y.eq(ratio).any().toArray(as: Bool.self)[0]
  #expect(hasRatio)

  let keptMask = y.gt(0.0).to(dtype: .float64)
  let keptFraction = keptMask.mean()
  #expect(keptFraction.isClose(to: Tensor(Double(keep), dtype: .float64), rtol: 0.0, atol: 0.02, equalNan: false))

}

@Test("Dropout: training edge case p=0 behaves like identity")
func dropoutTraining_p0_isIdentity() throws {
  let x = Tensor(array: [1.0, -2.0, 3.0], shape: [3])
  let layer = Dropout(probability: 0.0)

  // Forward + backward under training should still be identity when p == 0.
  let (y, pb) = withLearningPhase(.training) {
    valueWithPullback(at: x) { input in layer(input) }
  }
  #expect(y.isClose(to: x, rtol: 0, atol: 0, equalNan: false))

  let upstream = Tensor(array: [0.1, -0.2, 0.3], shape: [3])
  let grad = pb(upstream)
  #expect(grad.isClose(to: upstream, rtol: 0, atol: 0, equalNan: false))
}
