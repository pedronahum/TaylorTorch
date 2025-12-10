// Tests/TorchTests/SequentialLayerTests.swift
import Testing
import _Differentiation

@testable import Torch

// MARK: - Helpers

private func tensor1D(_ elements: [Double]) -> Tensor {
  Tensor(array: elements, shape: [elements.count], dtype: .float64)
}

private func tensor2D(_ rows: [[Double]]) -> Tensor {
  precondition(!rows.isEmpty)
  let cols = rows[0].count
  precondition(rows.allSatisfy { $0.count == cols }, "tensor2D: ragged rows")
  return Tensor(array: rows.flatMap { $0 }, shape: [rows.count, cols], dtype: .float64)
}

private func makeLinear(weightRows: [[Double]], bias: [Double]) -> Linear {
  var layer = Linear(
    inputSize: weightRows.count,
    outputSize: weightRows.first?.count ?? 0,
    dtype: .float64)
  layer.weight = tensor2D(weightRows)
  layer.bias = tensor1D(bias)
  return layer
}

@Test("Sequential (builder): forward equals l2(l1(x))")
func sequential_forward_matches_composition() throws {
  let x = tensor2D([
    [0.5, -1.0, 2.0],
    [1.5, 0.0, -0.5],
  ])

  let l1 = makeLinear(
    weightRows: [
      [1.0, 0.5],
      [0.0, 2.0],
      [-1.0, 1.0],
    ],
    bias: [0.1, -0.2])
  let l2 = makeLinear(
    weightRows: [
      [2.0, 1.5],
      [-1.0, 0.5],
    ],
    bias: [0.0, 0.25])

  let model = Sequential {
    l1
    l2
  }

  let y = model(x)
  let expected = l2(l1(x))
  #expect(y.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Sequential (builder): gradient of sum(l2(l1(x))) matches analytic chain rule")
func sequential_gradient_sum_loss_matches_analytic() throws {
  let x = tensor2D([
    [0.5, -1.0, 2.0],
    [1.5, 0.0, -0.5],
  ])

  let l1 = makeLinear(
    weightRows: [
      [1.0, 0.5],
      [0.0, 2.0],
      [-1.0, 1.0],
    ],
    bias: [0.1, -0.2])
  let l2 = makeLinear(
    weightRows: [
      [2.0, 1.5],
      [-1.0, 0.5],
    ],
    bias: [0.0, 0.25])

  let model = Sequential {
    l1
    l2
  }

  // L = sum(y2) where y2 = l2(l1(x))
  let (_, pb) = valueWithPullback(at: model) { m in m(x).sum() }
  let g = pb(Tensor(1.0))  // g.body is Chain<Linear,Linear>.TangentVector

  // -------- Analytic gradients --------
  // y1 = l1(x) = x W1 + b1    with shape [B, H]
  // y2 = l2(y1) = y1 W2 + b2  with shape [B, O]
  // L = sum(y2) -> dL/dy2 = 1
  let y1 = l1(x)  // [B, H]
  let B = Double(x.shape[0])
  let sumX = x.sum(dim: 0)  // [in]
  let rowSumW2 = l2.weight.sum(dim: 1)  // [H]
  let expectedGW1 = sumX.reshaped([l1.weight.shape[0], 1])
    .multiplying(rowSumW2.reshaped([1, l1.weight.shape[1]]))  // [in, H]
  let expectedGb1 = rowSumW2.multiplying(B)

  let sumY1 = y1.sum(dim: 0)  // [H]
  let expectedGW2 = sumY1.reshaped([l2.weight.shape[0], 1])
    .multiplying(Tensor.ones(shape: [1, l2.weight.shape[1]], dtype: .float64))  // [H, O]
  let expectedGb2 = Tensor.full(B, shape: [l2.weight.shape[1]])

  #expect(g.body.first.weight.isClose(to: expectedGW1, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.body.first.bias.isClose(to: expectedGb1, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.body.second.weight.isClose(to: expectedGW2, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.body.second.bias.isClose(to: expectedGb2, rtol: 1e-9, atol: 1e-9, equalNan: false))
}

@Test("Sequential (builder): one SGD step equals manual update using analytic grads")
func sequential_sgd_step_matches_manual() throws {
  let x = tensor2D([
    [1.0, 2.0, 3.0],
    [-1.0, 0.5, -0.5],
  ])

  let l1 = makeLinear(
    weightRows: [
      [0.2, -0.4],
      [-0.1, 0.5],
      [0.3, -0.6],
    ],
    bias: [0.05, -0.1])
  let l2 = makeLinear(
    weightRows: [
      [1.0, -1.0],
      [0.0, 2.0],
    ],
    bias: [0.0, 0.1])

  var model = Sequential {
    l1
    l2
  }

  // Compute gradient of L = sum(model(x))
  let (_, pb) = valueWithPullback(at: model) { m in m(x).sum() }
  let g = pb(Tensor(1.0))

  // Take one SGD step
  let sgd = SGD(for: model, learningRate: 0.05)
  sgd.update(&model, along: g)

  // Manual expected params from g.body.first / g.body.second
  let exp_l1_w = l1.weight.adding(g.body.first.weight.multiplying(-0.05))
  let exp_l1_b = l1.bias.adding(g.body.first.bias.multiplying(-0.05))
  let exp_l2_w = l2.weight.adding(g.body.second.weight.multiplying(-0.05))
  let exp_l2_b = l2.bias.adding(g.body.second.bias.multiplying(-0.05))

  let tol: Double = 1e-8
  #expect(model.body.first.weight.isClose(to: exp_l1_w, rtol: tol, atol: tol, equalNan: false))
  #expect(model.body.first.bias.isClose(to: exp_l1_b, rtol: tol, atol: tol, equalNan: false))
  #expect(model.body.second.weight.isClose(to: exp_l2_w, rtol: tol, atol: tol, equalNan: false))
  #expect(model.body.second.bias.isClose(to: exp_l2_b, rtol: tol, atol: tol, equalNan: false))
}

// DISABLED: Crashes on Linux due to KeyPath issues with complex models
// See KNOWN_ISSUES.md for details
@Test("Sequential (builder): parameter traversal order and flattenedParameters()", .disabled())
func sequential_parameter_keypaths_and_flattening() throws {
  let l1 = makeLinear(
    weightRows: [
      [0.0, 2.0],
      [1.0, 3.0],
    ],
    bias: [4.0, 5.0])
  let l2 = makeLinear(
    weightRows: [
      [6.0, 8.0],
      [7.0, 9.0],
    ],
    bias: [10.0, 11.0])
  let model = Sequential {
    l1
    l2
  }

  let keyPaths = model.recursivelyAllWritableKeyPaths(to: Tensor.self)
  #expect(keyPaths.count == 4)

  let flat = keyPaths.map { model[keyPath: $0] }
  #expect(flat[0].equal(model.body.first.weight))
  #expect(flat[1].equal(model.body.first.bias))
  #expect(flat[2].equal(model.body.second.weight))
  #expect(flat[3].equal(model.body.second.bias))

  // Round‑trip assign (sanity)
  var copy = model
  for (kp, value) in zip(keyPaths, flat) {
    copy[keyPath: kp] = value
  }
  #expect(copy.body.first.weight.equal(model.body.first.weight))
  #expect(copy.body.first.bias.equal(model.body.first.bias))
  #expect(copy.body.second.weight.equal(model.body.second.weight))
  #expect(copy.body.second.bias.equal(model.body.second.bias))
}

