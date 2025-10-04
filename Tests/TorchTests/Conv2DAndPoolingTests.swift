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

  var layer = Conv2D.kaimingUniform(
    inC: C, outC: C, kH: 1, kW: 1, groups: C, dtype: .float64, dataFormat: .nchw)
  // Make it an exact identity: weight entries = 1, bias = 0.
  layer.weight = Tensor(array: Array(repeating: 1.0, count: C), shape: [C, 1, 1, 1], dtype: .float64)
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
  let x = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64).reshaped([
    N, C, H, W,
  ])

  var layer = Conv2D.kaimingUniform(
    inC: C, outC: C, kH: 1, kW: 1, groups: C, dtype: .float64, dataFormat: .nchw)
  // scales: [2.0, -1.0], bias: [0.5, 1.5]
  layer.weight = Tensor(array: [2.0, -1.0], shape: [C, 1, 1, 1], dtype: .float64)
  layer.bias = Tensor(array: [0.5, 1.5], shape: [C], dtype: .float64)

  let y = layer(x)
  let scaled = x.multiplying(layer.weight.reshaped([1, C, 1, 1]))
  let expected = scaled.adding(layer.bias.reshaped([1, C, 1, 1]))
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// --- 3) Gradient sanity on tiny 1x1 conv (sum loss) ---

@Test("Conv2D: gradients for 1x1 single-channel match analytic (sum loss)")
func conv2d_gradients_1x1_single_channel() throws {
  // N=1, C=1, H=2, W=2  |  outC=1, groups=1, k=1
  let x = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [1, 1, 2, 2], dtype: .float64)

  var layer = Conv2D.kaimingUniform(
    inC: 1, outC: 1, kH: 1, kW: 1, dtype: .float64, dataFormat: .nchw)
  // Start from zeros for clean analytic comparison.
  layer.weight = Tensor.zeros(shape: [1, 1, 1, 1], dtype: .float64)
  layer.bias = Tensor.zeros(shape: [1], dtype: .float64)

  // L = sum(conv(x))
  let (_, pb) = valueWithPullback(at: layer) { l in l(x).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  // Analytic:
  // y = w * x + b  (elementwise for 1x1)
  // dL/dw = sum(x), dL/db = N*H*W
  let expectedDw = Tensor(array: [x.sum().toArray(as: Double.self)[0]], shape: [1, 1, 1, 1])
  let expectedDb = Tensor(array: [Double(x.shape.reduce(1, *))], shape: [1])

  #expect(g.weight.isClose(to: expectedDw, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(g.bias.isClose(to: expectedDb, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("Conv2D: simple valid convolution without bias matches manual result")
func conv2d_simple_valid() throws {
  let x = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([1, 1, 4, 4])
  let weight = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [1, 1, 2, 2], dtype: .float64)
  let bias = Tensor.zeros(shape: [1], dtype: .float64)

  var layer = Conv2D(
    weight: weight,
    bias: bias,
    stride: (1, 1),
    padding: .valid,
    dilation: (1, 1),
    groups: 1,
    dataFormat: .nchw
  )

  let y = layer(x)
  let expected = Tensor(
    array: [
      34.0, 44.0, 54.0,
      74.0, 84.0, 94.0,
      114.0, 124.0, 134.0,
    ],
    shape: [1, 1, 3, 3],
    dtype: .float64)

  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("Conv2D: backward gradients match manual convolution example")
func conv2d_backward_manual() throws {
  let x = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [1, 1, 2, 2], dtype: .float64)

  var layer = Conv2D(
    weight: Tensor(array: [2.0], shape: [1, 1, 1, 1], dtype: .float64),
    bias: Tensor(array: [0.5], shape: [1], dtype: .float64),
    stride: (1, 1),
    padding: .valid,
    dilation: (1, 1),
    groups: 1,
    dataFormat: .nchw
  )

  let (_, pb) = valueWithPullback(at: x, layer) { x, layer in
    layer(x).sum()
  }
  let (gradInput, gradLayer) = pb(Tensor(1.0, dtype: .float64))

  let expectedInput = Tensor(
    array: [
      2.0, 2.0,
      2.0, 2.0,
    ],
    shape: [1, 1, 2, 2],
    dtype: .float64)
  let expectedWeight = Tensor(array: [10.0], shape: [1, 1, 1, 1], dtype: .float64)
  let expectedBias = Tensor(array: [4.0], shape: [1], dtype: .float64)

  #expect(gradInput.isClose(to: expectedInput, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gradLayer.weight.isClose(to: expectedWeight, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gradLayer.bias.isClose(to: expectedBias, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// --- 4) NHWC compatibility path mirrors NCHW ---

@Test("Conv2D: NHWC path equals NCHW (1x1 depthwise)")
func conv2d_nhwc_equals_nchw() throws {
  let N = 1
  let C = 2
  let H = 2
  let W = 3
  let xNCHW = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])  // N C H W -> N H W C

  var layerNCHW = Conv2D.kaimingUniform(
    inC: C, outC: C, kH: 1, kW: 1, groups: C, dtype: .float64, dataFormat: .nchw)
  var layerNHWC = layerNCHW
  layerNHWC.dataFormat = .nhwc

  // Make both layers identical and diagonal scaling per channel.
  layerNCHW.weight = Tensor(array: [1.5, -0.5], shape: [C, 1, 1, 1], dtype: .float64)
  layerNCHW.bias = Tensor(array: [0.25, -1.0], shape: [C], dtype: .float64)
  layerNHWC.weight = layerNCHW.weight
  layerNHWC.bias = layerNCHW.bias

  let yNCHW = layerNCHW(xNCHW)  // [N,C,H,W]
  let yNHWC = layerNHWC(xNHWC)  // [N,H,W,C]
  let yNHWC_to_NCHW = yNHWC.permuted([0, 3, 1, 2])

  #expect(xNHWC.permuted([0, 3, 1, 2]).isClose(to: xNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(yNHWC_to_NCHW.isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// --- 5) Pooling smoke tests ---

@Test("MaxPool2D: simple 2x2 stride-2 pooling (NCHW)")
func maxpool2d_simple() throws {
  // N=1,C=1,H=4,W=4, values 0..15
  let x = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([1, 1, 4, 4])
  let pool = MaxPool2D(kernel: (2, 2), dataFormat: .nchw)

  let y = pool(x)
  // Expected: max over each 2x2 block
  let expected = Tensor(
    array: [
      5.0, 7.0,
      13.0, 15.0,
    ], shape: [1, 1, 2, 2], dtype: .float64)

  #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))
}

@Test("MaxPool2D: NHWC path equals NCHW")
func maxpool2d_nhwc_parity() throws {
  let xNCHW = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([1, 1, 4, 4])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  let poolNCHW = MaxPool2D(kernel: (2, 2), dataFormat: .nchw)
  var poolNHWC = poolNCHW
  poolNHWC.dataFormat = .nhwc

  let yNCHW = poolNCHW(xNCHW)
  let yNHWC = poolNHWC(xNHWC)

  #expect(yNHWC.permuted([0, 3, 1, 2]).isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("AvgPool2D: simple 2x2 stride-2 pooling (NCHW)")
func avgpool2d_simple() throws {
  // Mirror the doctest: averaging each 2x2 block of a 4x4 tensor.
  let x = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([1, 1, 4, 4])
  let pool = AvgPool2D(kernel: (2, 2), dataFormat: .nchw)

  let y = pool(x)
  let expected = Tensor(
    array: [2.5, 4.5, 10.5, 12.5],
    shape: [1, 1, 2, 2],
    dtype: .float64)

  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("AvgPool2D: padding excluded from denominator by default")
func avgpool2d_padding_excludes_padded_elements() throws {
  let x = Tensor(array: [1.0, 2.0, 3.0, 4.0], shape: [1, 1, 2, 2], dtype: .float64)
  let pool = AvgPool2D(kernel: (2, 2), stride: (1, 1), padding: (1, 1), dataFormat: .nchw)

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

@Test("AvgPool2D: simple 2x2 stride-2 pooling (NHWC parity)")
func avgpool2d_nhwc_parity() throws {
  let xNCHW = Tensor.arange(Double(0), to: Double(16), step: 1, dtype: .float64).reshaped([
    1, 1, 4, 4,
  ])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  let pNCHW = AvgPool2D(kernel: (2, 2), dataFormat: .nchw)
  var pNHWC = pNCHW
  pNHWC.dataFormat = .nhwc

  let yNCHW = pNCHW(xNCHW)
  let yNHWC = pNHWC(xNHWC)
  #expect(
    yNHWC.permuted([0, 3, 1, 2]).isClose(
      to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
