import Testing
import _Differentiation

@testable import Torch

// Helpers
private func makeDiagPointwiseWeight(scales: [Double], dtype: DType) -> Tensor {
  let c = scales.count
  var data = Array(repeating: 0.0, count: c * c)
  for i in 0..<c {
    data[i * c + i] = scales[i]
  }
  let matrix = Tensor(array: data, shape: [c, c], dtype: dtype)
  return matrix.reshaped([c, c, 1, 1])
}

// 1) Depthwise 1x1 identity, then pointwise per-channel scale+bias.
@Test("DSConv2D: depthwise 1x1 identity + pointwise scale/bias (NCHW)")
func dsconv2d_depthwise_identity_pointwise_affine() throws {
  let N = 1
  let C = 3
  let H = 2
  let W = 2
  let x = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])

  var ds = DepthwiseSeparableConv2D(inC: C, outC: C, kH: 1, kW: 1, dataFormat: .nchw)

  // Depthwise = identity
  ds.depthwise.weight = Tensor(
    array: Array(repeating: 1.0, count: C), shape: [C, 1, 1, 1], dtype: .float64)
  ds.depthwise.bias = Tensor.zeros(shape: [C], dtype: .float64)

  // Pointwise = per-channel scale+bias
  let scales: [Double] = [2.0, -1.0, 0.5]
  ds.pointwise.weight = makeDiagPointwiseWeight(scales: scales, dtype: .float64)
  ds.pointwise.bias = Tensor(array: [0.5, -1.0, 2.0], shape: [C], dtype: .float64)

  let y = ds(x)
  let expected =
    x
    .multiplying(Tensor(array: scales, shape: [1, C, 1, 1], dtype: .float64))
    .adding(ds.pointwise.bias.reshaped([1, C, 1, 1]))
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// 2) Parity with a single Conv2D whose kernel W[o,i,:,:] = P[o,i]*D[i,:,:]
@Test("DSConv2D: parity with fused Conv2D (depthwise ∘ pointwise)")
func dsconv2d_parity_with_fused_conv() throws {
  let N = 1
  let inC = 2
  let outC = 3
  let H = 5
  let W = 5
  let kH = 3
  let kW = 3
  let x = Tensor.arange(0.0, to: Double(N * inC * H * W), step: 1, dtype: .float64)
    .reshaped([N, inC, H, W])

  // Build DS with explicit weights (no biases except pointwise).
  var ds = DepthwiseSeparableConv2D(
    inC: inC, outC: outC, kH: kH, kW: kW,
    stride: (1, 1), padding: .valid, dilation: (1, 1),
    dtype: .float64, device: .cpu, dataFormat: .nchw,
    activation: .identity)

  // Depthwise kernels D[i,:,:]
  let D = Tensor(
    array: [
      0.0, 0.0, 0.0,
      0.0, 2.0, 0.0,
      0.0, 0.0, 0.0,

      -1.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
    ],
    shape: [inC, 1, kH, kW],
    dtype: .float64)  // simple center impulses for clarity
  ds.depthwise.weight = D
  ds.depthwise.bias = Tensor.zeros(shape: [inC], dtype: .float64)

  // Pointwise mixing P[o,i]
  let P = Tensor(
    array: [0.5, -1.0, 2.0, 3.0, 0.0, 1.0],
    shape: [outC, inC, 1, 1],
    dtype: .float64)
  ds.pointwise.weight = P
  ds.pointwise.bias = Tensor(array: [0.1, -0.2, 0.3], shape: [outC], dtype: .float64)

  // Build a single Conv2D with W[o,i,:,:] = P[o,i] * D[i,:,:]; bias = pointwise.bias
  var fused = Conv2D.kaimingUniform(
    inC: inC, outC: outC, kH: kH, kW: kW,
    dtype: .float64, dataFormat: .nchw)
  let fusedWeight = P.reshaped([outC, inC, 1, 1])
    .multiplying(D.reshaped([1, inC, kH, kW]))
  fused.weight = fusedWeight
  fused.bias = ds.pointwise.bias

  let yDS = ds(x)
  let yFused = fused(x)

  #expect(yDS.isClose(to: yFused, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// 3) Gradient parity (input gradient) with fused conv under separable kernel
@Test("DSConv2D: input gradient equals fused Conv2D gradient (sum loss)")
func dsconv2d_input_grad_parity() throws {
  let N = 1
  let inC = 2
  let outC = 2
  let H = 3
  let W = 3
  let kH = 3
  let kW = 3
  let x = Tensor.arange(0.0, to: Double(N * inC * H * W), step: 1, dtype: .float64)
    .reshaped([N, inC, H, W])

  // Same construction as prior test, simpler weights.
  var ds = DepthwiseSeparableConv2D(
    inC: inC, outC: outC, kH: kH, kW: kW,
    stride: (1, 1), padding: .valid, dilation: (1, 1),
    dtype: .float64, device: .cpu, dataFormat: .nchw,
    activation: .identity)
  ds.depthwise.weight = Tensor(
    array: [
      0, 0, 0,
      0, 1, 0,
      0, 0, 0,
    ], shape: [1, 1, 3, 3], dtype: .float64
  ).broadcasted(to: [inC, 1, 3, 3])
  ds.depthwise.bias = Tensor.zeros(shape: [inC], dtype: .float64)
  ds.pointwise.weight = Tensor(
    array: [
      1, 0,
      0, 1,
    ], shape: [outC, inC, 1, 1], dtype: .float64)
  ds.pointwise.bias = Tensor.zeros(shape: [outC], dtype: .float64)

  var fused = Conv2D.kaimingUniform(
    inC: inC, outC: outC, kH: kH, kW: kW,
    dtype: .float64, dataFormat: .nchw)
  let fusedWeight = ds.pointwise.weight.reshaped([outC, inC, 1, 1])
    .multiplying(ds.depthwise.weight.reshaped([1, inC, kH, kW]))
  fused.weight = fusedWeight
  fused.bias = Tensor.zeros(shape: [outC], dtype: .float64)

  let (_, pbDS) = valueWithPullback(at: x, ds) { x, l in l(x).sum() }
  let (gradX_DS, _) = pbDS(Tensor(1.0, dtype: .float64))

  let (_, pbFused) = valueWithPullback(at: x, fused) { x, l in l(x).sum() }
  let (gradX_fused, _) = pbFused(Tensor(1.0, dtype: .float64))

  #expect(gradX_DS.isClose(to: gradX_fused, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// 4) NHWC parity mirrors NCHW
@Test("DSConv2D: NHWC path equals NCHW (parity)")
func dsconv2d_nhwc_parity() throws {
  let N = 1
  let C = 2
  let H = 2
  let W = 3
  let xNCHW = Tensor.arange(Double(0), to: Double(N * C * H * W), step: 1, dtype: .float64)
    .reshaped([N, C, H, W])
  let xNHWC = xNCHW.permuted([0, 2, 3, 1])

  var dsNCHW = DepthwiseSeparableConv2D(inC: C, outC: C, kH: 1, kW: 1, dataFormat: .nchw)
  var dsNHWC = DepthwiseSeparableConv2D(inC: C, outC: C, kH: 1, kW: 1, dataFormat: .nhwc)

  // Per-channel scale+bias via depthwise identity + pointwise diagonal
  dsNCHW.depthwise.weight = Tensor(array: [1.0, 1.0], shape: [C, 1, 1, 1], dtype: .float64)
  dsNCHW.pointwise.weight = Tensor(
    array: [1.5, 0.0, 0.0, -0.5], shape: [C, C, 1, 1], dtype: .float64)
  dsNCHW.depthwise.bias = Tensor.zeros(shape: [C], dtype: .float64)
  dsNCHW.pointwise.bias = Tensor(array: [0.25, -1.0], shape: [C], dtype: .float64)

  dsNHWC.depthwise.weight = dsNCHW.depthwise.weight
  dsNHWC.pointwise.weight = dsNCHW.pointwise.weight
  dsNHWC.depthwise.bias = dsNCHW.depthwise.bias
  dsNHWC.pointwise.bias = dsNCHW.pointwise.bias

  let yNCHW = dsNCHW(xNCHW)  // [N,C,H,W]
  let yNHWC = dsNHWC(xNHWC)  // [N,H,W,C]
  #expect(
    yNHWC.permuted([0, 3, 1, 2]).isClose(to: yNCHW, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// 5) Activation behavior (ReLU)
@Test("DSConv2D: activation(.relu) applied after pointwise")
func dsconv2d_activation_relu() throws {
  let x = Tensor(array: [-2.0, -1.0, 0.0, 3.0], shape: [1, 1, 2, 2], dtype: .float64)
  var ds = DepthwiseSeparableConv2D(inC: 1, outC: 1, kH: 1, kW: 1, dataFormat: .nchw)
  // Make layer the identity transformation, then rely on activation:
  ds.depthwise.weight = Tensor(array: [1.0], shape: [1, 1, 1, 1], dtype: .float64)
  ds.depthwise.bias = Tensor.zeros(shape: [1], dtype: .float64)
  ds.pointwise.weight = Tensor(array: [1.0], shape: [1, 1, 1, 1], dtype: .float64)
  ds.pointwise.bias = Tensor.zeros(shape: [1], dtype: .float64)
  ds.activation = .relu

  let y = ds(x)
  let expected = Tensor(array: [0.0, 0.0, 0.0, 3.0], shape: [1, 1, 2, 2], dtype: .float64)
  #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))
}
