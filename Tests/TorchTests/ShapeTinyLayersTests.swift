// Sources/TorchTests/ShapeTinyLayersTests.swift
import Testing
import _Differentiation

@testable import Torch

@Test("Flatten (all but batch) forward shape + gradient identity")
func flatten_all_but_batch_forward_and_grad() throws {
  // x: [B=2, C=3, H=2, W=2] -> [2, 12]
  let x = Tensor.arange(Double(0), to: Double(24), step: 1, dtype: .float64)
    .reshaped([2, 3, 2, 2])
  let layer = Flatten()  // startDim:1 endDim:-1

  let y = layer(x)
  #expect(y.shape == [2, 12])

  // L = sum(y) → dL/dx = ones_like(x)
  let (_, pb) = valueWithPullback(at: x) { t in layer(t).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))
  #expect(g.isClose(to: Tensor.ones(shape: x.shape, dtype: .float64)))
}

@Test("Sequential: Flatten → Linear composes and matches manual baseline")
func sequential_flatten_then_linear() throws {
  // Pretend CNN output [B, C, H, W]; flatten to [B, C*H*W] then Linear
  let B = 2
  let C = 2
  let H = 2
  let W = 3
  let x = Tensor.arange(Double(0), to: Double(B * C * H * W), step: 1, dtype: .float64)
    .reshaped([B, C, H, W])

  // Linear expects [B, in], W: [out, in]
  let inFeatures = C * H * W
  let outFeatures = 4
  let weight = Tensor.arange(Double(0), to: Double(outFeatures * inFeatures), step: 1, dtype: .float64)
    .reshaped([inFeatures, outFeatures])
  let bias = Tensor.zeros(shape: [outFeatures], dtype: .float64)

  var linear = Linear(inputSize: inFeatures, outputSize: outFeatures, dtype: .float64)
  linear.weight = weight
  linear.bias = bias
  let model = Sequential {
    Flatten()
    linear
  }
  let y = model(x)

  let manual = x.reshaped([B, inFeatures]).matmul(weight).adding(bias)
  #expect(y.isClose(to: manual, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Gradients w.r.t Linear params under L = sum(y) should match manual.
  let (_, pbModel) = valueWithPullback(at: model) { m in m(x).sum() }
  let gModel = pbModel(Tensor(1.0, dtype: .float64))
  let linearGrad = gModel.body.second

  let (_, pbManual) = valueWithPullback(at: weight, bias) { W, b in
    x.reshaped([B, inFeatures]).matmul(W).adding(b).sum()
  }
  let (gW, gb) = pbManual(Tensor(1.0, dtype: .float64))

  #expect(linearGrad.weight.isClose(to: gW, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(linearGrad.bias.isClose(to: gb, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
