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

/// Padding strategies supported by `Conv2D`.
public enum Padding {
  /// No implicit padding (valid convolution).
  case valid, same
  /// Explicit padding supplied for height and width.
  case explicit(h: Int, w: Int)
}

/// A 2D convolutional layer that mirrors PyTorch's `Conv2d` semantics.
public struct Conv2D: Layer {
  /// Learnable convolution kernels with shape `[outChannels, inChannels/groups, kernelHeight, kernelWidth]`.
  public var weight: Tensor
  /// Learnable per-output-channel bias.
  public var bias: Tensor

  /// Vertical and horizontal stride.
  @noDerivative public var stride: (Int, Int)
  /// Padding policy applied to the input tensor.
  @noDerivative public var padding: Padding
  /// Dilation factors for height and width.
  @noDerivative public var dilation: (Int, Int)
  /// Number of groups that split input and output channels.
  @noDerivative public var groups: Int
  /// Layout of the input tensor (`.nchw` or `.nhwc`).
  @noDerivative public var dataFormat: DataFormat

  /// Creates a convolutional layer with the provided parameters.
  /// - Parameters:
  ///   - weight: Convolution kernel tensor.
  ///   - bias: Bias tensor broadcast across each output channel.
  ///   - stride: Step taken between receptive fields along height and width.
  ///   - padding: Padding policy for the spatial dimensions.
  ///   - dilation: Spacing inserted between kernel elements.
  ///   - groups: Number of channel groups processed independently.
  ///   - dataFormat: Input tensor layout.
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

  /// Writable key paths to the layer's trainable parameters.
  public static var parameterKeyPaths: [WritableKeyPath<Conv2D, Tensor>] {
    [\Conv2D.weight, \Conv2D.bias]
  }

  /// Updates the layer's parameters by applying the tangent `offset`.
  /// - Parameter offset: Tangent components that should be added to the parameters.
  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
    bias.move(by: offset.bias)
  }

  // Conv2D.swift

  /// Applies the convolution to `x`.
  /// - Parameter x: Input activations shaped according to `dataFormat`.
  /// - Returns: The convolved output tensor in the same layout as `x`.
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

  /// Resolves the amount of padding to apply given the selected policy.
  /// - Parameters:
  ///   - inputShapeNCHW: Input tensor shape expressed in NCHW form.
  ///   - kernel: Height and width of the convolution kernel.
  /// - Returns: Height and width padding values.
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

  /// Converts an NHWC tensor to NCHW.
  /// - Parameter t: Tensor in NHWC format.
  /// - Returns: Tensor in NCHW format.
  @differentiable(reverse) private func _nhwcToNchw(_ t: Tensor) -> Tensor {
    t.permuted([0, 3, 1, 2])
  }
  /// Converts an NCHW tensor to NHWC.
  /// - Parameter t: Tensor in NCHW format.
  /// - Returns: Tensor in NHWC format.
  @differentiable(reverse) private func _nchwToNhwc(_ t: Tensor) -> Tensor {
    t.permuted([0, 2, 3, 1])
  }

  /// Tangent representation for `Conv2D`.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Tangent for the convolution kernels.
    public var weight: Tensor
    /// Tangent for the bias parameter.
    public var bias: Tensor
    /// Writable key paths for the tangent components.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\TangentVector.weight, \TangentVector.bias]
    }
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector {
      .init(
        weight: Tensor.zeros(shape: [0, 0, 0, 0], dtype: .float32),
        bias: Tensor.zeros(shape: [0], dtype: .float32))
    }
    /// Adds two tangents element-wise.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(weight: l.weight + r.weight, bias: l.bias + r.bias)
    }
    /// Subtracts two tangents element-wise.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(weight: l.weight - r.weight, bias: l.bias - r.bias)
    }
  }
}

public extension Conv2D {
  /// Builds a `Conv2D` layer with Kaiming/He-uniform initialization, matching PyTorch.
  /// - Parameters:
  ///   - inC: Number of input channels.
  ///   - outC: Number of output channels.
  ///   - kH: Kernel height.
  ///   - kW: Kernel width.
  ///   - stride: Convolution stride.
  ///   - padding: Padding policy for the spatial dimensions.
  ///   - dilation: Kernel dilation factors.
  ///   - groups: Number of channel groups processed independently.
  ///   - dtype: Element dtype for the parameters.
  ///   - device: Device on which to allocate the tensors.
  ///   - dataFormat: Expected input layout.
  /// - Returns: A `Conv2D` instance whose parameters follow Kaiming-uniform initialization.
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
