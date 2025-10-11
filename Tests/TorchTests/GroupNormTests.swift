// Tests/TorchTests/GroupNormTests.swift
import Testing
import _Differentiation

@testable import Torch

@Test("GroupNorm: forward equals manual grouping (NCHW)")
func groupnorm_forward_manual_nchw() throws {
  // N=1, C=4, H=2, W=1; G=2 → Cg=2
  let x = Tensor(
    array: [
      // c0  c1  |  c2  c3
      1.0, 2.0, -1.0, 0.0,
      3.0, -1.0, 2.0, -2.0,
    ], shape: [1, 4, 2, 1], dtype: .float64)

  var gn = GroupNorm(featureCount: 4, groups: 2, axis: 1, epsilon: 1e-5, dtype: .float64)
  gn.gamma = Tensor(array: [1.5, 0.5, -1.0, 2.0], shape: [4], dtype: .float64)
  gn.beta = Tensor(array: [0.1, -0.2, 0.0, 0.25], shape: [4], dtype: .float64)

  let spatial = x.shape.dropFirst(2).reduce(1, *)
  let channelsPerGroup = x.shape[1] / gn.groups
  let grouped = x.reshaped([x.shape[0], gn.groups, channelsPerGroup * spatial])
  let mean = grouped.mean(dim: 2, keepdim: true)
  let centered = grouped.subtracting(mean)
  let var_ = centered.multiplying(centered).mean(dim: 2, keepdim: true)
  let epsTensor = Tensor(Double(gn.epsilon), dtype: .float64)
  let norm = centered.dividing(var_.adding(epsTensor).sqrt())
  let normNCHW = norm.reshaped(x.shape)

  let gamma = gn.gamma.reshaped([1, 4, 1, 1])
  let beta = gn.beta.reshaped([1, 4, 1, 1])
  let expected = normNCHW.multiplying(gamma).adding(beta)

  let y = gn(x)
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("GroupNorm: NHWC parity with NCHW")
func groupnorm_nhwc_parity() throws {
  let xNCHW = Tensor(
    array: [
      1.0, 2.0, -1.0, 0.0,
      3.0, -1.0, 2.0, -2.0,
    ], shape: [1, 4, 2, 1], dtype: .float64)
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  var nchw = GroupNorm(featureCount: 4, groups: 2, axis: 1, dtype: .float64)
  var nhwc = nchw
  nchw.gamma = Tensor(array: [1.5, 0.5, -1.0, 2.0], shape: [4], dtype: .float64)
  nchw.beta = Tensor(array: [0.1, -0.2, 0.0, 0.25], shape: [4], dtype: .float64)
  nhwc.gamma = nchw.gamma
  nhwc.beta = nchw.beta
  nhwc.axis = -1

  let yNCHW = nchw(xNCHW)
  let yNHWC = nhwc(xNHWC)
  #expect(yNHWC.permuted([0, 3, 1, 2])
    .isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("GroupNorm: analytic grads for γ, β under L = sum(y)")
func groupnorm_parameter_grads() throws {
  let x = Tensor.arange(Double(0), to: Double(1 * 4 * 2 * 2), step: 1, dtype: .float64)
    .reshaped([1, 4, 2, 2])

  var gn = GroupNorm(featureCount: 4, groups: 2, axis: 1, dtype: .float64)
  gn.gamma = Tensor.ones(shape: [4], dtype: .float64)
  gn.beta = Tensor.zeros(shape: [4], dtype: .float64)

  let (_, pb) = valueWithPullback(at: gn) { m in m(x).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  // Manual expectation matches the forward formula with current parameters.
  let spatial = x.shape.dropFirst(2).reduce(1, *)
  let channelsPerGroup = x.shape[1] / gn.groups
  let grouped = x.reshaped([x.shape[0], gn.groups, channelsPerGroup * spatial])
  let mean = grouped.mean(dim: 2, keepdim: true)
  let centered = grouped.subtracting(mean)
  let var_ = centered.multiplying(centered).mean(dim: 2, keepdim: true)
  let epsTensor = Tensor(Double(gn.epsilon), dtype: .float64)
  let norm = centered.dividing(var_.adding(epsTensor).sqrt()).reshaped(x.shape)

  let expectedDgamma = norm.sum(dim: 0).sum(dim: 1).sum(dim: 1)
  let nhw = Double(x.shape[0] * x.shape[2] * x.shape[3])
  var expectedDbeta = Tensor.full(nhw, shape: [x.shape[1]], device: g.beta.device)
  if let dtype = g.beta.dtype, expectedDbeta.dtype != dtype {
    expectedDbeta = expectedDbeta.to(dtype: dtype)
  }

  #expect(g.gamma.isClose(to: expectedDgamma, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(g.beta.isClose(to: expectedDbeta, rtol: 1e-9, atol: 1e-9, equalNan: false))
}

@Test("GroupNorm: input gradient matches finite differences (NCHW)")
func groupnorm_input_gradients_match_fd() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0, 0.0,
      1.5, 3.0, -0.5, -2.0,
      0.25, -0.75, 1.25, 2.5,
      -1.5, 0.5, -2.0, 1.0,
    ], shape: [1, 4, 2, 2], dtype: .float64)

  var gn = GroupNorm(featureCount: 4, groups: 2, axis: 1, epsilon: 1e-5, dtype: .float64)
  gn.gamma = Tensor(array: [1.3, -0.7, 0.5, 2.0], shape: [4], dtype: .float64)
  gn.beta = Tensor(array: [0.1, 0.0, -0.2, 0.25], shape: [4], dtype: .float64)

  let (_, pb) = valueWithPullback(at: x) { input in gn(input).sum() }
  let gradAnalytic = pb(Tensor(1.0, dtype: .float64))

  let eps: Double = 1e-3
  let base = x.toArray(as: Double.self)
  var numeric = [Double](repeating: 0, count: base.count)
  for i in 0..<base.count {
    var plus = base
    var minus = base
    plus[i] += eps
    minus[i] -= eps
    let yPlus = gn(Tensor(array: plus, shape: x.shape, dtype: .float64)).sum().toArray(as: Double.self)[0]
    let yMinus = gn(Tensor(array: minus, shape: x.shape, dtype: .float64)).sum().toArray(as: Double.self)[0]
    numeric[i] = (yPlus - yMinus) / (2.0 * eps)
  }

  let gradNumeric = Tensor(array: numeric, shape: x.shape, dtype: .float64)
  #expect(gradAnalytic.isClose(to: gradNumeric, rtol: 1e-5, atol: 1e-6, equalNan: false))
}

@Test("GroupNorm: NHWC gradients match NCHW after transpose")
func groupnorm_nhwc_gradients_parity() throws {
  let xNCHW = Tensor(
    array: [
      0.0, 1.0, 2.0, 3.0,
      -1.0, -0.5, 0.5, 1.0,
      2.5, -2.0, 1.5, -1.5,
      -0.25, 0.75, -1.25, 1.25,
    ], shape: [1, 4, 2, 2], dtype: .float64)
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  var nchw = GroupNorm(featureCount: 4, groups: 2, axis: 1, epsilon: 1e-5, dtype: .float64)
  nchw.gamma = Tensor(array: [1.25, -0.4, 0.75, 1.5], shape: [4], dtype: .float64)
  nchw.beta = Tensor(array: [0.0, 0.2, -0.3, 0.5], shape: [4], dtype: .float64)

  var nhwc = nchw
  nhwc.axis = -1

  let (_, pbXnchw) = valueWithPullback(at: xNCHW) { input in nchw(input).sum() }
  let gradXnchw = pbXnchw(Tensor(1.0, dtype: .float64))

  let (_, pbXnhwc) = valueWithPullback(at: xNHWC) { input in nhwc(input).sum() }
  let gradXnhwc = pbXnhwc(Tensor(1.0, dtype: .float64))
  let gradXnhwcAsNCHW = gradXnhwc.permuted([0, 3, 1, 2])
  #expect(gradXnhwcAsNCHW.isClose(to: gradXnchw, rtol: 1e-12, atol: 1e-12, equalNan: false))

  let (_, pbLayerNCHW) = valueWithPullback(at: nchw) { layer in layer(xNCHW).sum() }
  let gradsNCHW = pbLayerNCHW(Tensor(1.0, dtype: .float64))

  let (_, pbLayerNHWC) = valueWithPullback(at: nhwc) { layer in layer(xNHWC).sum() }
  let gradsNHWC = pbLayerNHWC(Tensor(1.0, dtype: .float64))

  #expect(gradsNHWC.gamma.isClose(to: gradsNCHW.gamma, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gradsNHWC.beta.isClose(to: gradsNCHW.beta, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
