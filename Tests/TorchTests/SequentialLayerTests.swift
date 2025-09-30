import Testing
import _Differentiation

@testable import Torch

@Test("Sequential: forward equals l2(l1(x))")
func sequential_forward_matches_composition() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3])

  let l1 = Linear(
    weight: Tensor(
      array: [
        1.0, 0.0, -1.0,
        0.5, 2.0, 1.0,
      ], shape: [2, 3]),
    bias: Tensor(array: [0.1, -0.2], shape: [2]))

  let l2 = Linear(
    weight: Tensor(
      array: [
        2.0, -1.0,
        1.5, 0.5,
      ], shape: [2, 2]),
    bias: Tensor(array: [0.0, 0.25], shape: [2]))

  let model = Sequential(l1, l2)

  let y = model(x)
  let expected = l2(l1(x))
  #expect(y.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Sequential: gradient of sum(l2(l1(x))) matches analytic chain rule")
func sequential_gradient_sum_loss_matches_analytic() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3])

  var l1 = Linear(
    weight: Tensor(
      array: [
        1.0, 0.0, -1.0,
        0.5, 2.0, 1.0,
      ], shape: [2, 3]),
    bias: Tensor(array: [0.1, -0.2], shape: [2]))

  var l2 = Linear(
    weight: Tensor(
      array: [
        2.0, -1.0,
        1.5, 0.5,
      ], shape: [2, 2]),
    bias: Tensor(array: [0.0, 0.25], shape: [2]))

  var model = Sequential(l1, l2)

  // L = sum(y2) where y2 = l2(l1(x))
  let (_, pb) = valueWithPullback(at: model) { m in m(x).sum() }
  let g = pb(Tensor(1.0))  // Sequential.TangentVector: (l1: TV, l2: TV)

  // -------- Analytic gradients --------
  // y1 = l1(x) = x W1^T + b1    with shape [B, H]
  // y2 = l2(y1) = y1 W2^T + b2  with shape [B, O]
  // L = sum(y2) -> dL/dy2 = 1
  // dL/dy1[i,k] = sum_j W2[j,k] = colSum(W2)[k]  (same for each i)
  // => dL/dW1[j,k] = (sum_i x[i,k]) * colSum(W2)[j]
  //    dL/db1[j]   = B * colSum(W2)[j]
  // dL/dW2[j,k] = sum_i y1[i,k]   (same across all j)
  // dL/db2[j]   = B
  let y1 = l1(x)  // [B, H]
  let B = Double(x.shape[0])
  let sumX = x.sum(dim: 0)  // [in]
  let colSumW2 = l2.weight.sum(dim: 0)  // [H]
  let onesOut1 = Tensor.ones(shape: [l1.weight.shape[0], 1], dtype: .float64)
  let expectedGW1 =
    onesOut1
    .multiplying(colSumW2.reshaped([l1.weight.shape[0], 1]))  // broadcast along columns
    .multiplying(sumX.reshaped([1, l1.weight.shape[1]]))  // [out1, in]
  let expectedGb1 = colSumW2.multiplying(B)

  let sumY1 = y1.sum(dim: 0)  // [H]
  let onesOut2 = Tensor.ones(shape: [l2.weight.shape[0], 1], dtype: .float64)
  let expectedGW2 = onesOut2.multiplying(sumY1.reshaped([1, l2.weight.shape[1]]))  // [out2,H]
  let expectedGb2 = Tensor.full(B, shape: [l2.weight.shape[0]])

  #expect(g.l1.weight.isClose(to: expectedGW1, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.l1.bias.isClose(to: expectedGb1, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.l2.weight.isClose(to: expectedGW2, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.l2.bias.isClose(to: expectedGb2, rtol: 1e-9, atol: 1e-9, equalNan: false))
}

@Test("Sequential: one SGD step equals manual update using analytic grads")
func sequential_sgd_step_matches_manual() throws {
  let x = Tensor(
    array: [
      1.0, 2.0, 3.0,
      -1.0, 0.5, -0.5,
    ], shape: [2, 3])

  let l1 = Linear(
    weight: Tensor(
      array: [
        0.2, -0.1, 0.3,
        -0.4, 0.5, -0.6,
      ], shape: [2, 3]),
    bias: Tensor(array: [0.05, -0.1], shape: [2]))

  let l2 = Linear(
    weight: Tensor(
      array: [
        1.0, 0.0,
        -1.0, 2.0,
      ], shape: [2, 2]),
    bias: Tensor(array: [0.0, 0.1], shape: [2]))

  var model = Sequential(l1, l2)

  // Compute gradient of L = sum(model(x))
  let (_, pb) = valueWithPullback(at: model) { m in m(x).sum() }
  let g = pb(Tensor(1.0))

  // Take one SGD step
  var sgd = SGD(for: model, learningRate: 0.05)
  sgd.update(&model, along: g)

  // Manual expected params
  let exp_l1_w = l1.weight.adding(g.l1.weight.multiplying(-0.05))
  let exp_l1_b = l1.bias.adding(g.l1.bias.multiplying(-0.05))
  let exp_l2_w = l2.weight.adding(g.l2.weight.multiplying(-0.05))
  let exp_l2_b = l2.bias.adding(g.l2.bias.multiplying(-0.05))

  #expect(model.l1.weight.isClose(to: exp_l1_w, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(model.l1.bias.isClose(to: exp_l1_b, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(model.l2.weight.isClose(to: exp_l2_w, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(model.l2.bias.isClose(to: exp_l2_b, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("Sequential: parameter traversal order and flattenedParameters()")
func sequential_parameter_keypaths_and_flattening() throws {
  let l1 = Linear(
    weight: Tensor(array: [0, 1, 2, 3], shape: [2, 2]),
    bias: Tensor(array: [4, 5], shape: [2])
  )
  let l2 = Linear(
    weight: Tensor(array: [6, 7, 8, 9], shape: [2, 2]),
    bias: Tensor(array: [10, 11], shape: [2])
  )
  var model = Sequential(l1, l2)

  let flat = model.flattenedParameters()
  #expect(flat.count == 4)

  // Expected order: l1.weight, l1.bias, l2.weight, l2.bias
  #expect(flat[0].equal(l1.weight))
  #expect(flat[1].equal(l1.bias))
  #expect(flat[2].equal(l2.weight))
  #expect(flat[3].equal(l2.bias))

  // Round‑trip assign (sanity)
  var copy = model
  copy.assignFlattenedParameters(flat)
  #expect(copy.l1.weight.equal(l1.weight))
  #expect(copy.l2.bias.equal(l2.bias))
}
