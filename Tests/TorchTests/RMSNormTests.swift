// Tests/TorchTests/RMSNormTests.swift
import Testing
import _Differentiation

@testable import Torch

@Test("RMSNorm: forward equals manual formula (last-dim)")
func rmsnorm_forward_manual() throws {
  // x: [B, F]
  let x = Tensor(
    array: [
      1.0, -2.0, 3.0,
      -1.0, 0.0, 2.0,
    ], shape: [2, 3], dtype: .float64)

  let gamma = Tensor(array: [0.5, 2.0, -1.0], shape: [3], dtype: .float64)
  let beta = Tensor(array: [0.1, -0.2, 0.3], shape: [3], dtype: .float64)

  var layer = RMSNorm(features: 3, epsilon: 1e-8, dtype: .float64)
  layer.weight = gamma
  layer.bias = beta

  // Manual baseline: y = (x / sqrt(mean(x^2, -1, keepdim))) * γ + β
  let last = x.rank - 1
  let rms = x.multiplying(x).mean(dim: last, keepdim: true).adding(1e-8).sqrt()
  let yManual = x.dividing(rms)
    .multiplying(gamma.reshaped([1, 3]))
    .adding(beta.reshaped([1, 3]))

  let y = layer(x)
  #expect(y.isClose(to: yManual, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("RMSNorm: analytic grads for γ, β under L = sum(y)")
func rmsnorm_parameter_grads() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3], dtype: .float64)

  var layer = RMSNorm(features: 3, epsilon: 1e-8, dtype: .float64)
  layer.weight = Tensor.ones(shape: [3], dtype: .float64)
  layer.bias = Tensor.zeros(shape: [3], dtype: .float64)

  // L = sum(RMSNorm(x)) → dL/dγ = sum(x/rms), dL/dβ = B
  let (_, pb) = valueWithPullback(at: layer) { m in m(x).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  let rms = x.multiplying(x).mean(dim: x.rank - 1, keepdim: true).adding(1e-8).sqrt()
  let expectedDgamma = (x.dividing(rms)).sum(dim: 0)  // [F]
  let expectedDbeta = Tensor.full(Double(x.shape[0]), shape: [x.shape[1]])

  #expect(g.weight.isClose(to: expectedDgamma, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(g.bias.isClose(to: expectedDbeta, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
