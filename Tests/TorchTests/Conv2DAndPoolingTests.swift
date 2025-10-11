// Tests/TorchTests/Conv2DAndPoolingTests.swift
import Testing
import _Differentiation

@testable import Torch

// --- 1) 1x1 identity via depthwise groups (outC == inC, groups == inC) ---

@Test("Conv2D: 1x1 depthwise with weight=1,bias=0 acts as identity (NCHW)")
func conv2d_depthwise_identity_nchw() throws {
  let N = 1
  let C = 3
  let H = 2
  let W = 2
  let x = Tensor.arange(Double(1), to: Double(N * C * H * W + 1), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])

  var layer = Conv2D(
    kaimingUniformInChannels: C, outChannels: C, kernelSize: (1, 1), groups: C, dtype: .float64)
  // Make it an exact identity: weight entries = 1, bias = 0.
  layer.weight = Tensor(
    array: Array(repeating: 1.0, count: C), shape: [C, 1, 1, 1], dtype: .float64)
  layer.bias = Tensor.zeros(shape: [C], dtype: .float64)

  let y = layer(x)
  #expect(y.isClose(to: x, rtol: 0, atol: 0, equalNan: false))
}

// --- 2) Grouped 1x1 equals per-channel scale + bias ---

@Test("Conv2D: 1x1 depthwise equals per-channel scale+bias")
func conv2d_depthwise_scale_bias() throws {
  let N = 1
  let C = 2
  let H = 2
  let W = 3
  let x = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])

  var layer = Conv2D(
    kaimingUniformInChannels: C, outChannels: C, kernelSize: (1, 1), groups: C, dtype: .float64)
  // scales: [2.0, -1.0], bias: [0.5, 1.5]
  layer.weight = Tensor(array: [2.0, -1.0], shape: [C, 1, 1, 1], dtype: .float64)
  layer.bias = Tensor(array: [0.5, 1.5], shape: [C], dtype: .float64)

  let y = layer(x)
  let scaled = x.multiplying(layer.weight.reshaped([1, C, 1, 1]))
  let expected = scaled.adding(layer.bias.reshaped([1, C, 1, 1]))
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// --- 3) Gradient sanity on tiny 1x1 conv (sum loss) ---

@Test("Conv2D: gradients for 1x1 depthwise & sum loss are sane")
func conv2d_grad_simple_sum() throws {
  let N = 1
  let C = 2
  let H = 2
  let W = 2
  let x = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])

  var layer = Conv2D(
    kaimingUniformInChannels: C, outChannels: C, kernelSize: (1, 1), groups: C, dtype: .float64)
  // Simple diagonal scaling to make expected grads easy to reason about.
  layer.weight = Tensor(array: [1.5, -0.5], shape: [C, 1, 1, 1], dtype: .float64)
  layer.bias = Tensor(array: [0.25, -1.0], shape: [C], dtype: .float64)

  // y = layer(x); L = sum(y)
  let expectedSum = layer(x).sum()
  let (y, pb) = valueWithPullback(at: layer, x) { l, i in l(i).sum() }
  #expect(y.isClose(to: expectedSum, rtol: 0, atol: 0, equalNan: false))

  let (gradLayer, gradInput) = pb(Tensor(1.0))
  // For 1x1 depthwise, dL/dW[c] = sum(x[c,:,:]), dL/db[c] = H*W, dL/dx[c,:,:] = W[c]
  let sumWidth = x.sum(dim: 3)  // [N, C, H]
  let sumHW = sumWidth.sum(dim: 2)  // [N, C]
  let expectedWeight = sumHW.sum(dim: 0).reshaped([C, 1, 1, 1])
  let expectedBias = Tensor
    .full(Double(H * W), shape: [C], device: gradLayer.bias.device)
    .to(dtype: gradLayer.bias.dtype ?? .float32)
  let expectedInput = layer.weight.reshaped([1, C, 1, 1]).broadcasted(to: x.shape)

  #expect(gradInput.isClose(to: expectedInput, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gradLayer.weight.isClose(to: expectedWeight, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gradLayer.bias.isClose(to: expectedBias, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// --- 4) NHWC parity via explicit transposes ---

@Test("Conv2D: NHWC parity via explicit transposes (1x1 depthwise)")
func conv2d_nhwc_equals_nchw_via_permute() throws {
  let N = 1
  let C = 2
  let H = 2
  let W = 3
  let xNCHW = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])  // N C H W -> N H W C

  var layer = Conv2D(
    kaimingUniformInChannels: C, outChannels: C, kernelSize: (1, 1), groups: C, dtype: .float64)
  layer.weight = Tensor(array: [1.5, -0.5], shape: [C, 1, 1, 1], dtype: .float64)
  layer.bias = Tensor(array: [0.25, -1.0], shape: [C], dtype: .float64)

  let yNCHW = layer(xNCHW)  // [N,C,H,W]
  // Emulate NHWC path by transposing to NCHW, applying the same layer, then transposing back.
  let yNHWC = layer(xNHWC.permuted([0, 3, 1, 2])).permuted([0, 2, 3, 1])  // [N,H,W,C]

  #expect(
    xNHWC.permuted([0, 3, 1, 2]).isClose(to: xNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(
    yNHWC.permuted([0, 3, 1, 2]).isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// --- 5) Pooling smoke tests (NCHW primitives) ---

@Test("MaxPool2D: simple 2x2 stride-2 pooling (NCHW)")
func maxpool2d_simple() throws {
  let x = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([1, 1, 4, 4])
  let pool = MaxPool2D(kernelSize: (2, 2))

  let y = pool(x)
  let expected = Tensor(
    array: [
      5.0, 7.0,
      13.0, 15.0,
    ], shape: [1, 1, 2, 2], dtype: .float64)

  #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))
}

@Test("MaxPool2D: NHWC parity via explicit transposes")
func maxpool2d_nhwc_parity_via_permute() throws {
  let xNCHW = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([
    1, 1, 4, 4,
  ])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  let pool = MaxPool2D(kernelSize: (2, 2))
  let yNCHW = pool(xNCHW)
  let yNHWC = pool(xNHWC.permuted([0, 3, 1, 2])).permuted([0, 2, 3, 1])

  #expect(
    yNHWC.permuted([0, 3, 1, 2]).isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("AvgPool2D: simple 2x2 stride-2 pooling (NCHW)")
func avgpool2d_simple() throws {
  let x = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([1, 1, 4, 4])
  let pool = AvgPool2D(kernelSize: (2, 2))

  let y = pool(x)
  let expected = Tensor(array: [2.5, 4.5, 10.5, 12.5], shape: [1, 1, 2, 2], dtype: .float64)
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("AvgPool2D: padding excluded from denominator by default")
func avgpool2d_padding_excludes_padded_elements() throws {
  let x = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [1, 1, 2, 2], dtype: .float64)
  let pool = AvgPool2D(kernelSize: (2, 2), stride: (1, 1), padding: (1, 1))

  let y = pool(x)
  let expected = Tensor(
    array: [
      1.0, 1.5, 2.0,
      2.0, 2.5, 3.0,
      3.0, 3.5, 4.0,
    ],
    shape: [1, 1, 3, 3],
    dtype: .float64)

  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("AvgPool2D: NHWC parity via explicit transposes")
func avgpool2d_nhwc_parity_via_permute() throws {
  let xNCHW = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([
    1, 1, 4, 4,
  ])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  let p = AvgPool2D(kernelSize: (2, 2))
  let yNCHW = p(xNCHW)
  let yNHWC = p(xNHWC.permuted([0, 3, 1, 2])).permuted([0, 2, 3, 1])
  #expect(
    yNHWC.permuted([0, 3, 1, 2]).isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
