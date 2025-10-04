//
// Sources/Torch/Modules/Layers/Conv2D.swift
//
// WHY
// - Unlocks vision models (CNNs) with a small, Swifty surface.
// - Mirrors PyTorch/ATen defaults (NCHW input, weight [outC, inC/groups, kH, kW]).
// - Offers NHWC compatibility to ease S4TF-style porting.
// - Parameters are exposed via static key paths, so your optimizers & Euclidean
//   algebra work out of the box.  (Layer → EuclideanModel → ParameterIterableModel).
//
// References in this repo: Layer.swift, Sequential.swift, Initializers.swift, Optimizers.swift.
//                                                                ⤷ traversal + AD + algebra
//                                                    ⤷ builder + combinators

// Sources/Torch/Modules/Layers/Conv2D.swift
import Foundation
import _Differentiation

public enum Padding {
  case valid, same
  case explicit(h: Int, w: Int)
}

public struct Conv2D: Layer {
  public var weight: Tensor
  public var bias: Tensor

  @noDerivative public var stride: (Int, Int)
  @noDerivative public var padding: Padding
  @noDerivative public var dilation: (Int, Int)
  @noDerivative public var groups: Int
  @noDerivative public var dataFormat: DataFormat

  public init(
    weight: Tensor, bias: Tensor,
    stride: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilation: (Int, Int) = (1, 1),
    groups: Int = 1,
    dataFormat: DataFormat = .nchw
  ) {
    self.weight = weight
    self.bias = bias
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.dataFormat = dataFormat
  }

  public static var parameterKeyPaths: [WritableKeyPath<Conv2D, Tensor>] {
    [\Conv2D.weight, \Conv2D.bias]
  }

  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
    bias.move(by: offset.bias)
  }

  // Conv2D.swift

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let xNCHW = (dataFormat == .nchw) ? x : _nhwcToNchw(x)

    // All shape/Int math must be outside AD:
    let kH = withoutDerivative(at: weight.shape[2])
    let kW = withoutDerivative(at: weight.shape[3])
    let (padH, padW) = withoutDerivative(
      at: _resolvePadding(forInput: withoutDerivative(at: xNCHW.shape), kernel: (kH, kW)))

    let stride64: [Int64] = [
      withoutDerivative(at: Int64(stride.0)), withoutDerivative(at: Int64(stride.1)),
    ]
    let padding64: [Int64] = [
      withoutDerivative(at: Int64(padH)), withoutDerivative(at: Int64(padW)),
    ]
    let dilation64: [Int64] = [
      withoutDerivative(at: Int64(dilation.0)), withoutDerivative(at: Int64(dilation.1)),
    ]
    let groups64: Int64 = withoutDerivative(at: Int64(groups))

    // ✅ Pass the bias directly to the single differentiable operation
    let yNCHW = xNCHW.conv2d(
      weight: weight,
      bias: self.bias,
      stride: stride64,
      padding: padding64,
      dilation: dilation64,
      groups: groups64
    )

    // The output is already biased. Just handle the data format.
    return (dataFormat == .nchw) ? yNCHW : _nchwToNhwc(yNCHW)
  }

  private func _resolvePadding(forInput inputShapeNCHW: [Int], kernel: (Int, Int)) -> (Int, Int) {
    switch padding {
    case .valid: return (0, 0)
    case .explicit(let h, let w): return (h, w)
    case .same:
      let effKH = (kernel.0 - 1) * dilation.0 + 1
      let effKW = (kernel.1 - 1) * dilation.1 + 1
      return (effKH / 2, effKW / 2)
    }
  }

  @differentiable(reverse) private func _nhwcToNchw(_ t: Tensor) -> Tensor {
    t.permuted([0, 3, 1, 2])
  }
  @differentiable(reverse) private func _nchwToNhwc(_ t: Tensor) -> Tensor {
    t.permuted([0, 2, 3, 1])
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var weight: Tensor
    public var bias: Tensor
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\TangentVector.weight, \TangentVector.bias]
    }
    public static var zero: TangentVector {
      .init(
        weight: Tensor.zeros(shape: [0, 0, 0, 0], dtype: .float32),
        bias: Tensor.zeros(shape: [0], dtype: .float32))
    }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(weight: l.weight + r.weight, bias: l.bias + r.bias)
    }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(weight: l.weight - r.weight, bias: l.bias - r.bias)
    }
  }
}

public extension Conv2D {
  /// Builds a `Conv2D` layer with Kaiming/He-uniform initialization, matching PyTorch.
  static func kaimingUniform(
    inC: Int,
    outC: Int,
    kH: Int,
    kW: Int,
    stride: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilation: (Int, Int) = (1, 1),
    groups: Int = 1,
    dtype: DType = .float32,
    device: Device = .cpu,
    dataFormat: DataFormat = .nchw
  ) -> Conv2D {
    precondition(groups > 0, "groups must be positive")
    precondition(inC % groups == 0, "inC must be divisible by groups")
    let channelsPerGroup = inC / groups
    let fanIn = channelsPerGroup * kH * kW
    precondition(fanIn > 0, "fanIn must be positive")

    let bound = Foundation.sqrt(6.0 / Double(fanIn))
    let weight = Tensor.uniform(
      low: -bound,
      high: bound,
      shape: [outC, channelsPerGroup, kH, kW],
      dtype: dtype,
      device: device
    )
    let biasTensor = Tensor.zeros(shape: [outC], dtype: dtype, device: device)

    return Conv2D(
      weight: weight,
      bias: biasTensor,
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups,
      dataFormat: dataFormat
    )
  }
}
