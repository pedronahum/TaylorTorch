import Testing
import _Differentiation

@testable import Torch

@Test("Dense = Linear + activation (forward + grads)")
func dense_matches_linear_plus_activation() throws {
  // Data
  let x = Tensor(array: [0.5, -1.0, 2.0, 1.5, 0.0, -0.5], shape: [2, 3])

  // Parameters
  let W = Tensor(array: [1.0, 0.0, -1.0, 0.5, 2.0, 1.0], shape: [2, 3])
  let b = Tensor(array: [0.1, -0.2], shape: [2])

  let lin = Linear(weight: W, bias: b)
  let dense = Dense(linear: lin, activation: Activations.relu)

  // Forward: Dense == relu(Linear(x))
  let yDense = dense(x)
  let yManual = lin(x).relu()
  #expect(yDense.isClose(to: yManual, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Gradients under L = sum(y)
  let (_, pbDense) = valueWithPullback(at: dense) { m in m(x).sum() }
  let gDense = pbDense(Tensor(1.0))

  // Manual baseline (same computation)
  let (_, pbLin) = valueWithPullback(at: lin) { l in l(x).relu().sum() }
  let gLin = pbLin(Tensor(1.0))

  #expect(gDense.linear.weight.isClose(to: gLin.weight, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gDense.linear.bias.isClose(to: gLin.bias, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
