// Tests/TorchTests/LayerNormTests.swift
import Testing
import _Differentiation

@testable import Torch

private func rowwiseMean(_ t: Tensor) -> Tensor {
  t.mean(dim: 1, keepdim: false)
}
private func rowwiseVar(_ t: Tensor) -> Tensor {
  let m = t.mean(dim: 1, keepdim: true)
  return (t.subtracting(m).multiplying(t.subtracting(m))).mean(dim: 1, keepdim: false)
}

@Test("LayerNorm: normalizes last axis to ~zero mean and unit variance")
func layerNormNormalizesLastDim() throws {
  // 2x3 toy data; choose epsilon=0 so variance matches exactly.
  let x = Tensor(
    array: [
      -1.0, 0.0, 1.0,
      2.0, 4.0, 6.0,
    ], shape: [2, 3])
  let ln = LayerNorm(featureCount: 3, epsilon: 0.0)
  let y = ln(x)

  // Each row has ~zero mean and unit variance.
  let m = rowwiseMean(y).toArray(as: Double.self)
  let v = rowwiseVar(y).toArray(as: Double.self)
  #expect(Swift.abs(m[0]) < 1e-6 && Swift.abs(m[1]) < 1e-6)
  #expect(Swift.abs(v[0] - 1.0) < 1e-6 && Swift.abs(v[1] - 1.0) < 1e-6)
}

@Test("LayerNorm: affine parameters scale/shift the normalized output")
func layerNormAppliesAffine() throws {
  let x = Tensor(
    array: [
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
    ], shape: [2, 3])
  var ln = LayerNorm(featureCount: 3, epsilon: 0.0)
  // Set gamma=2, beta=3
  ln.gamma = Tensor.full(2.0, shape: [3])
  ln.beta = Tensor.full(3.0, shape: [3])

  // y = 2 * norm(x) + 3
  let y = ln(x)

  // Remove shift then rescale to check normalization still holds.
  let norm = y.subtracting(3.0).dividing(2.0)
  let m = rowwiseMean(norm).toArray(as: Double.self)
  let v = rowwiseVar(norm).toArray(as: Double.self)
  #expect(Swift.abs(m[0]) < 1e-6 && Swift.abs(m[1]) < 1e-6)
  #expect(Swift.abs(v[0] - 1.0) < 1e-6 && Swift.abs(v[1] - 1.0) < 1e-6)
}

@Test("LayerNorm: parameter gradients match simple analytic cases")
func layerNormParameterGradients() throws {
  // With L = sum(ln(x)), dL/dβ = number of broadcasts (batch size),
  // and dL/dγ = sum(normalized(x)) which is ~0 for each row by construction.
  let x = Tensor(
    array: [
      -1.0, 0.0, 1.0,
      4.0, 5.0, 6.0,
    ], shape: [2, 3])

  let ln = LayerNorm(featureCount: 3, epsilon: 0.0)

  let (_, pullback) = valueWithPullback(at: ln) { layer in
    layer(x).sum()
  }
  let g = pullback(Tensor(1.0))

  // Gradients are broadcast over the batch dimension. Collapse the leading axes
  // to compare against the analytic expectations.
  let dGamma = g.gamma.sum(dim: 0)
  let dBeta = g.beta

  let expectedGamma = Tensor.zeros(shape: [3], dtype: dGamma.dtype ?? .float32, device: dGamma.device)
  let expectedBeta = Tensor.full(Double(x.shape[0]), shape: [3], device: dBeta.device)
    .to(dtype: dBeta.dtype ?? .float32)

  #expect(dGamma.isClose(to: expectedGamma, rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(dBeta.isClose(to: expectedBeta, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("LayerNorm: contextual call equals pure call (no running stats)")
func layerNormContextualCallMatches() throws {
  let x = Tensor.arange(Double(0), to: Double(12), step: Double(1)).reshaped([2, 2, 3])
  let ln = LayerNorm(featureCount: 3, epsilon: 1e-5)
  let y1 = ln(x)
  let y2 = ln.call(x, context: .init(training: true))
  #expect(y1.isClose(to: y2, rtol: 0, atol: 0, equalNan: false))
}

@Test("LayerNorm: works on higher-rank tensors (normalize last dim)")
func layerNormHigherRank() throws {
  // Shape [B=2, T=2, F=4], normalize across last axis (F)
  let x = Tensor(
    array: [
      // t0
      1.0, 2.0, 3.0, 4.0,
      10.0, 8.0, 6.0, 4.0,
      // t1
      -1.0, -2.0, -3.0, -4.0,
      4.0, 4.0, 4.0, 5.0,
    ], shape: [2, 2, 4])

  let ln = LayerNorm(featureCount: 4, epsilon: 0.0)
  let y = ln(x)

  // For every (b, t), last-dim mean≈0 and var≈1.
  let mean = y.mean(dim: 2, keepdim: false).toArray(as: Double.self)
  let varr =
    (y.subtracting(y.mean(dim: 2, keepdim: true))
    .multiplying(y.subtracting(y.mean(dim: 2, keepdim: true))))
    .mean(dim: 2, keepdim: false)
    .toArray(as: Double.self)

  for m in mean { #expect(Swift.abs(m) < 1e-6) }
  for v in varr { #expect(Swift.abs(v - 1.0) < 1e-6) }
}
