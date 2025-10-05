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

@Test("Reshape with -1 inference round-trips element count")
func reshape_with_inference_forward_and_grad() throws {
  let x = Tensor.arange(Double(0), to: Double(12), step: 1, dtype: .float64)
    .reshaped([3, 2, 2])  // 12 elems
  let layer = Reshape([2, -1, 2])  // -> [2, 3, 2]

  let y = layer(x)
  #expect(y.shape == [2, 3, 2])

  let (_, pb) = valueWithPullback(at: x) { t in layer(t).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))
  #expect(g.isClose(to: Tensor.ones(shape: x.shape, dtype: .float64)))
}

@Test("Permute reorders axes and transpose swaps two dims")
func permute_and_transpose_forward_and_grad() throws {
  // [B, H, W, C] → [B, C, H, W] and back
  let x = Tensor.arange(Double(0), to: Double(2 * 2 * 3 * 4), step: 1, dtype: .float64)
    .reshaped([2, 2, 3, 4])

  let toNCHW = Permute([0, 3, 1, 2])
  let y = toNCHW(x)
  #expect(y.shape == [2, 4, 2, 3])

  let swapHW = Permute([0, 1, 3, 2])
  let z = swapHW(y)
  #expect(z.shape == [2, 4, 3, 2])

  // Grad identity under sum loss
  do {
    let (_, pb) = valueWithPullback(at: x) { t in toNCHW(t).sum() }
    let g = pb(Tensor(1.0, dtype: .float64))
    #expect(g.isClose(to: Tensor.ones(shape: x.shape, dtype: .float64)))
  }
  do {
    let (_, pb) = valueWithPullback(at: y) { t in swapHW(t).sum() }
    let g = pb(Tensor(1.0, dtype: .float64))
    #expect(g.isClose(to: Tensor.ones(shape: y.shape, dtype: .float64)))
  }
}

@Test("Squeeze/Unsqueeze adjust rank and keep differentiation smooth")
func squeeze_unsqueeze_forward_and_grad() throws {
  let x = Tensor.arange(Double(0), to: Double(6), step: 1, dtype: .float64).reshaped([1, 2, 3, 1])

  let sqAll = Squeeze()
  let y = sqAll(x)
  #expect(y.shape == [2, 3])

  let usq = Unsqueeze(-1)  // append trailing 1
  let z = usq(y)
  #expect(z.shape == [2, 3, 1])

  // Grad identity under sum loss
  do {
    let (_, pb) = valueWithPullback(at: x) { t in sqAll(t).sum() }
    let g = pb(Tensor(1.0, dtype: .float64))
    #expect(g.isClose(to: Tensor.ones(shape: x.shape, dtype: .float64)))
  }
  do {
    let (_, pb) = valueWithPullback(at: y) { t in usq(t).sum() }
    let g = pb(Tensor(1.0, dtype: .float64))
    #expect(g.isClose(to: Tensor.ones(shape: y.shape, dtype: .float64)))
  }
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
    .reshaped([outFeatures, inFeatures])
  let bias = Tensor.zeros(shape: [outFeatures], dtype: .float64)

  let model = Sequential(Flatten(), Linear(weight: weight, bias: bias))
  let y = model(x)

  let manual = x.reshaped([B, inFeatures]).matmul(weight.transposed(-1, -2)).adding(bias)
  #expect(y.isClose(to: manual, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Gradients w.r.t Linear params under L = sum(y) should match manual.
  let (_, pbModel) = valueWithPullback(at: model) { m in m(x).sum() }
  let gModel = pbModel(Tensor(1.0, dtype: .float64))

  let (_, pbManual) = valueWithPullback(at: weight, bias) { W, b in
    x.reshaped([B, inFeatures]).matmul(W.transposed(-1, -2)).adding(b).sum()
  }
  let (gW, gb) = pbManual(Tensor(1.0, dtype: .float64))

  #expect(gModel.l2.weight.isClose(to: gW, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gModel.l2.bias.isClose(to: gb, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
