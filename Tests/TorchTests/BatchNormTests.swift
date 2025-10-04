// Tests/TorchTests/BatchNormTests.swift
import Testing
import _Differentiation

@testable import Torch

// ---------- BatchNorm1D ----------

@Test("BatchNorm1D: training forward equals manual normalization")
func bn1d_training_forward_matches_manual() throws {
  // x: [N,F] with simple values
  let x = Tensor(
    array: [
      0.0, 1.0, 2.0,
      3.0, 4.0, 5.0,
    ], shape: [2, 3], dtype: .float64)

  var bn = BatchNorm1D(numFeatures: 3, momentum: 0.1, epsilon: 1e-5, dtype: .float64)

  // Choose nontrivial affine parameters.
  bn.weight = Tensor(array: [1.0, 2.0, -1.0], shape: [3], dtype: .float64)
  bn.bias = Tensor(array: [0.5, -0.25, 0.0], shape: [3], dtype: .float64)

  let ctx = ForwardContext(training: true)
  let (_, pb) = valueWithPullback(at: bn) { m in m.call(x, context: ctx).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  // Analytic for L = sum(gamma * z + beta): dL/dgamma = sum(z), dL/dbeta = number of elements
  let mean = x.mean(dim: 0)
  let varT = x.subtracting(mean.reshaped([1, 3]))
    .multiplying(x.subtracting(mean.reshaped([1, 3])))
    .mean(dim: 0)
  let z = x.subtracting(mean.reshaped([1, 3])).dividing(varT.adding(1e-5).sqrt().reshaped([1, 3]))
  let expected_dgamma = z.sum(dim: 0)
  let expected_dbeta = Tensor.full(Double(x.shape[0]), shape: [x.shape[1]])

  #expect(g.weight.isClose(to: expected_dgamma, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(g.bias.isClose(to: expected_dbeta, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("BatchNorm1D: running stats are updated in training")
func bn1d_running_stats_update() throws {
  let x = Tensor(
    array: [
      0.0, 1.0,
      2.0, 3.0,
    ], shape: [2, 2], dtype: .float64)

  let mom = 0.1
  let bn = BatchNorm1D(numFeatures: 2, momentum: mom, epsilon: 1e-5, dtype: .float64)

  let ctx = ForwardContext(training: true)
  _ = bn.call(x, context: ctx)

  let m = x.mean(dim: 0)
  let centered = x.subtracting(m.reshaped([1, 2]))
  let v = centered.multiplying(centered).mean(dim: 0)

  let expectedMean = m.multiplying(mom)  // prev was zeros
  let expectedVar = Tensor.ones(shape: [2], dtype: .float64).multiplying(1.0 - mom).adding(
    v.multiplying(mom))

  #expect(bn.runningMean.value.isClose(to: expectedMean, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(bn.runningVar.value.isClose(to: expectedVar, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// ---------- BatchNorm2D ----------

@Test("BatchNorm2D: training forward equals manual (NCHW)")
func bn2d_training_forward_matches_manual_nchw() throws {
  let N = 1
  let C = 2
  let H = 2
  let W = 3
  let x = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])

  var bn = BatchNorm2D(
    numFeatures: C, momentum: 0.1, epsilon: 1e-5, dataFormat: .nchw, dtype: .float64)
  bn.weight = Tensor(array: [1.5, -0.5], shape: [C], dtype: .float64)
  bn.bias = Tensor(array: [0.1, 0.3], shape: [C], dtype: .float64)

  let ctx = ForwardContext(training: true)
  let y = bn.call(x, context: ctx)

  // Manual stats over N,H,W
  var mean = x
  for d in [0, 2, 3].sorted(by: >) { mean = mean.mean(dim: d) }  // [C]
  let centered = x.subtracting(mean.reshaped([1, C, 1, 1]))
  var varT = centered.multiplying(centered)
  for d in [0, 2, 3].sorted(by: >) { varT = varT.mean(dim: d) }  // [C]
  let denom = varT.adding(1e-5).sqrt().reshaped([1, C, 1, 1])
  let norm = centered.dividing(denom)
  let expected = norm.multiplying(bn.weight.reshaped([1, C, 1, 1])).adding(
    bn.bias.reshaped([1, C, 1, 1]))

  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("BatchNorm2D: NHWC path equals NCHW after permutation")
func bn2d_nhwc_parity() throws {
  let N = 1
  let C = 2
  let H = 2
  let W = 3
  let xNCHW = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  var bnNCHW = BatchNorm2D(numFeatures: C, dataFormat: .nchw, dtype: .float64)
  var bnNHWC = BatchNorm2D(numFeatures: C, dataFormat: .nhwc, dtype: .float64)

  // Same affine and same initial running stats
  let w = Tensor(array: [1.25, -0.75], shape: [C], dtype: .float64)
  let b = Tensor(array: [0.2, -0.1], shape: [C], dtype: .float64)
  bnNCHW.weight = w
  bnNCHW.bias = b
  bnNHWC.weight = w
  bnNHWC.bias = b

  let ctx = ForwardContext(training: true)
  let yNCHW = bnNCHW.call(xNCHW, context: ctx)  // [N,C,H,W]
  let yNHWC = bnNHWC.call(xNHWC, context: ctx)  // [N,H,W,C]
  #expect(
    yNHWC.permuted([0, 3, 1, 2]).isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("BatchNorm2D: eval uses running stats")
func bn2d_eval_uses_running_stats() throws {
  let N = 2
  let C = 1
  let H = 2
  let W = 2
  let x = Tensor(
    array: [
      1.0, 2.0, 3.0, 4.0,
      2.0, 3.0, 4.0, 5.0,
    ], shape: [N, C, H, W], dtype: .float64)

  var bn = BatchNorm2D(
    numFeatures: C, momentum: 0.1, epsilon: 1e-5, dataFormat: .nchw, dtype: .float64)
  bn.weight = Tensor(array: [2.0], shape: [C], dtype: .float64)
  bn.bias = Tensor(array: [0.5], shape: [C], dtype: .float64)

  // One training pass to update running stats
  _ = bn.call(x, context: ForwardContext(training: true))

  // Now eval; must use running stats
  let yEval = bn(x)  // callAsFunction (eval)
  let m = bn.runningMean.value
  let v = bn.runningVar.value
  let expected =
    x
    .subtracting(m.reshaped([1, C, 1, 1]))
    .dividing(v.adding(1e-5).sqrt().reshaped([1, C, 1, 1]))
    .multiplying(bn.weight.reshaped([1, C, 1, 1]))
    .adding(bn.bias.reshaped([1, C, 1, 1]))

  #expect(yEval.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
