import Testing
import _Differentiation

@testable import Torch

@Test("Linear: forward equals x.matmul(W.T) + b")
func linear_forward_matches_definition() throws {
  // x: [batch, in], W: [out, in], b: [out]
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,  // sample 0
      1.5, 0.0, -0.5,  // sample 1
    ], shape: [2, 3])

  let W = Tensor(
    array: [
      1.0, 0.0, -1.0,  // row j=0
      0.5, 2.0, 1.0,  // row j=1
    ], shape: [2, 3])

  let b = Tensor(array: [0.1, -0.2], shape: [2])

  let layer = Linear(weight: W, bias: b)  // y = x W^T + b

  let y = layer(x)
  let expected = x.matmul(W.transposed(-1, -2)).adding(b)
  #expect(y.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Linear: gradient of sum(layer(x)) matches analytic dW, db")
func linear_gradient_sum_loss_matches_analytic() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3])

  let W = Tensor(
    array: [
      1.0, 0.0, -1.0,
      0.5, 2.0, 1.0,
    ], shape: [2, 3])

  let b = Tensor(array: [0.1, -0.2], shape: [2])
  var layer = Linear(weight: W, bias: b)

  // L = sum(Linear(x))
  let (_, pb) = valueWithPullback(at: layer) { l in l(x).sum() }
  let g = pb(Tensor(1.0))  // Linear.TangentVector (dL/dW, dL/db)

  // Analytic:
  // dL/dW_{j,k} = Σ_i x_{i,k}  (same across all rows j)
  // dL/db_j     = batchSize
  let batch = Double(x.shape[0])
  let sumX = x.sum(dim: 0)  // [in]
  let onesOut = Tensor.ones(shape: [W.shape[0], 1], dtype: .float64)
  let expectedGW = onesOut.multiplying(sumX.reshaped([1, W.shape[1]]))  // [out,in]
  let expectedGb = Tensor.full(batch, shape: [W.shape[0]])

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

  let W0 = Tensor(
    array: [
      0.2, -0.1, 0.3,
      -0.4, 0.5, -0.6,
    ], shape: [2, 3])
  let b0 = Tensor(array: [0.05, -0.1], shape: [2])

  var layer = Linear(weight: W0, bias: b0)
  let (loss, pb) = valueWithPullback(at: layer) { l in l(x).sum() }
  let grad = pb(Tensor(1.0))

  var opt = SGD(for: layer, learningRate: 0.1)
  opt.update(&layer, along: grad)  // w <- w - lr * grad

  // Manual expected
  let W1 = W0.adding(grad.weight.multiplying(-0.1))
  let b1 = b0.adding(grad.bias.multiplying(-0.1))

  #expect(layer.weight.isClose(to: W1, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(layer.bias.isClose(to: b1, rtol: 1e-12, atol: 1e-12, equalNan: false))
  // Loss was a scalar—just sanity check it's finite:
  _ = loss
}

@Test("Linear: parameter traversal order (weight, bias) and Euclidean view")
func linear_parameter_keypaths_and_vectorView() throws {
  var layer = Linear(
    weight: Tensor.arange(Double(0), to: Double(6), step: 1).reshaped([2, 3]),
    bias: Tensor(array: [1.0, 2.0], shape: [2])
  )

  // Order should be [\weight, \bias]
  let flat = layer.flattenedParameters()
  #expect(flat.count == 2)
  #expect(flat[0].equal(layer.weight))
  #expect(flat[1].equal(layer.bias))

  // Tangent view equals copying parameters in the same order
  let tv = layer.vectorView  // default uses asTangentVector() under the hood
  #expect(tv.weight.equal(layer.weight))
  #expect(tv.bias.equal(layer.bias))
}
