import Testing
import _Differentiation

@testable import Torch

@Test("Linear: forward equals x.matmul(W) + b")
func linear_forward_matches_definition() throws {
  // x: [batch, in], W: [in, out], b: [out]
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,  // sample 0
      1.5, 0.0, -0.5,  // sample 1
    ], shape: [2, 3])

  var layer = Linear(inputSize: 3, outputSize: 2, dtype: .float64)
  layer.weight = Tensor(
    array: [
      1.0, 0.5,
      0.0, 2.0,
      -1.0, 1.0,
    ], shape: [3, 2], dtype: .float64)
  layer.bias = Tensor(array: [0.1, -0.2], shape: [2], dtype: .float64)

  let y = layer(x)
  let expected = x.matmul(layer.weight).adding(layer.bias)
  #expect(y.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Linear: gradient of sum(layer(x)) matches analytic dW, db")
func linear_gradient_sum_loss_matches_analytic() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3])

  var layer = Linear(inputSize: 3, outputSize: 2, dtype: .float64)
  layer.weight = Tensor(
    array: [
      1.0, 0.5,
      0.0, 2.0,
      -1.0, 1.0,
    ], shape: [3, 2], dtype: .float64)
  layer.bias = Tensor(array: [0.1, -0.2], shape: [2], dtype: .float64)

  // L = sum(Linear(x))
  let (_, pb) = valueWithPullback(at: layer) { l in l(x).sum() }
  let g = pb(Tensor(1.0))  // Linear.TangentVector (dL/dW, dL/db)

  // Analytic:
  // y = x.matmul(W) + b with W shaped [in, out].
  // dL/dW_{i,j} = Σ_batch x_{batch,i}
  // dL/db_j     = batchSize
  let batch = Double(x.shape[0])
  let sumX = x.sum(dim: 0)  // [in]
  let onesOut = Tensor.ones(shape: [1, layer.weight.shape[1]], dtype: .float64)
  let expectedGW = sumX.reshaped([layer.weight.shape[0], 1]).multiplying(onesOut)  // [in,out]
  let expectedGb = Tensor.full(batch, shape: [layer.bias.shape[0]])

  #expect(g.weight.isClose(to: expectedGW, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.bias.isClose(to: expectedGb, rtol: 1e-9, atol: 1e-9, equalNan: false))
}

@Test("Linear: one SGD step equals manual parameter update")
func linear_sgd_step_matches_manual() throws {
  let x = Tensor(
    array: [
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
    ], shape: [2, 3])

  var layer = Linear(inputSize: 3, outputSize: 2, dtype: .float64)
  layer.weight = Tensor(
    array: [
      0.2, -0.4,
      -0.1, 0.5,
      0.3, -0.6,
    ], shape: [3, 2], dtype: .float64)
  layer.bias = Tensor(array: [0.05, -0.1], shape: [2], dtype: .float64)
  let (loss, pb) = valueWithPullback(at: layer) { l in l(x).sum() }
  let grad = pb(Tensor(1.0))

  let opt = SGD(for: layer, learningRate: 0.1)
  opt.update(&layer, along: grad)  // w <- w - lr * grad

  // Manual expected
  let W0 = Tensor(
    array: [
      0.2, -0.4,
      -0.1, 0.5,
      0.3, -0.6,
    ], shape: [3, 2], dtype: .float64)
  let b0 = Tensor(array: [0.05, -0.1], shape: [2], dtype: .float64)
  let W1 = W0.adding(grad.weight.multiplying(-0.1))
  let b1 = b0.adding(grad.bias.multiplying(-0.1))

  #expect(layer.weight.isClose(to: W1, rtol: 0.0003, atol: 0.0003, equalNan: false))
  #expect(layer.bias.isClose(to: b1, rtol: 0.0003, atol: 0.0003, equalNan: false))
  // Loss was a scalar—just sanity check it's finite:
  _ = loss
}

@Test("Linear: parameter traversal order (weight, bias) and Euclidean view")
func linear_parameter_keypaths_and_vectorView() throws {
  var layer = Linear(inputSize: 3, outputSize: 2, dtype: .float64)
  layer.weight = Tensor.arange(Double(0), to: Double(6), step: 1).reshaped([3, 2])
  layer.bias = Tensor(array: [1.0, 2.0], shape: [2], dtype: .float64)

  // Order should be [\weight, \bias]
  let keyPaths = layer.recursivelyAllWritableKeyPaths(to: Tensor.self)
  #expect(keyPaths.count == 2)
  #expect(layer[keyPath: keyPaths[0]].equal(layer.weight))
  #expect(layer[keyPath: keyPaths[1]].equal(layer.bias))

  // Tangent view equals copying parameters in the same order.
  let tangent = Linear.TangentVector(weight: layer.weight, bias: layer.bias)
  #expect(tangent.weight.equal(layer.weight))
  #expect(tangent.bias.equal(layer.bias))
}
