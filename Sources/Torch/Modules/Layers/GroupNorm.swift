import ATenCXX
import _Differentiation

/// Group normalization layer that supports NCHW and NHWC layouts.
public struct GroupNorm: Layer {
  /// Learnable per-channel scaling factors.
  public var weight: Tensor
  /// Learnable per-channel offsets.
  public var bias: Tensor

  /// Number of groups into which channels are partitioned.
  @noDerivative public var numGroups: Int
  /// Numerical stability constant added to variances.
  @noDerivative public var epsilon: Double
  /// Input layout (NCHW or NHWC).
  @noDerivative public var dataFormat: DataFormat

  /// Creates a group-normalization layer.
  /// - Parameters:
  ///   - numGroups: Number of channel groups.
  ///   - numChannels: Total number of input channels.
  ///   - epsilon: Small constant added to variances.
  ///   - dataFormat: Layout of the input tensor.
  ///   - dtype: Element dtype for the parameters.
  ///   - device: Device on which to allocate the parameters.
  public init(
    numGroups: Int,
    numChannels: Int,
    epsilon: Double = 1e-5,
    dataFormat: DataFormat = .nchw,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    precondition(numGroups > 0 && numChannels % numGroups == 0,
                 "numGroups must be > 0 and divide numChannels")
    self.weight = Tensor.ones(shape: [numChannels], dtype: dtype, device: device)
    self.bias = Tensor.zeros(shape: [numChannels], dtype: dtype, device: device)
    self.numGroups = numGroups
    self.epsilon = epsilon
    self.dataFormat = dataFormat
  }

  /// Cached forward-pass intermediates used by the custom VJP.
  private struct ForwardCache {
    let inputNCHW: Tensor
    let mean: Tensor
    let rstd: Tensor
  }

  /// Performs the forward pass and returns intermediate statistics for backpropagation.
  /// - Parameter x: Input activations.
  /// - Returns: Normalized output and cached values needed for the backward pass.
  private func forwardWithCache(_ x: Tensor) -> (Tensor, ForwardCache) {
    let shape = withoutDerivative(at: x.shape)
    precondition(shape.count >= 2, "GroupNorm expects tensors with at least 2 dimensions")
    if dataFormat == .nhwc {
      precondition(
        shape.count == 4,
        "GroupNorm with DataFormat.nhwc expects rank-4 input shaped [N, H, W, C]; got \(shape)"
      )
    }

    let channelAxis = dataFormat == .nchw ? 1 : shape.count - 1
    precondition(
      shape.indices.contains(channelAxis),
      "GroupNorm channel axis out of range for shape \(shape)"
    )
    let channelCount = shape[channelAxis]
    let expectedChannels = withoutDerivative(at: weight.shape[0])
    precondition(
      channelCount == expectedChannels,
      "GroupNorm channel count (\(channelCount)) must match weight length (\(expectedChannels))"
    )
    let biasChannels = withoutDerivative(at: bias.shape[0])
    precondition(
      biasChannels == expectedChannels,
      "GroupNorm bias length (\(biasChannels)) must match weight length (\(expectedChannels))"
    )

    let xNCHW: Tensor
    switch dataFormat {
    case .nchw:
      xNCHW = x
    case .nhwc:
      xNCHW = x.permuted([0, 3, 1, 2])
    }

    var w = weight
    var b = bias
    let tuple = withUnsafePointer(to: &w._impl) { wPtr in
      withUnsafePointer(to: &b._impl) { bPtr in
        ATenCXX.TTSTensor._native_group_norm_forward(
          xNCHW._impl,
          Int64(numGroups),
          wPtr,
          bPtr,
          epsilon
        )
      }
    }

    let yNCHW = Tensor(ATenCXX.TTSTensor._native_group_norm_forward_get0(tuple))
    let mean = Tensor(ATenCXX.TTSTensor._native_group_norm_forward_get1(tuple))
    let rstd = Tensor(ATenCXX.TTSTensor._native_group_norm_forward_get2(tuple))

    let value: Tensor
    switch dataFormat {
    case .nchw:
      value = yNCHW
    case .nhwc:
      value = yNCHW.permuted([0, 2, 3, 1])
    }

    return (value, ForwardCache(inputNCHW: xNCHW, mean: mean, rstd: rstd))
  }

  /// Applies group normalization to `x`.
  /// - Parameter x: Input activations.
  /// - Returns: Normalized activations.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    forwardWithCache(x).0
  }

  /// Custom VJP that leverages ATen's fused group-norm backward kernel.
  /// - Parameter x: Input activations.
  /// - Returns: Layer output and a pullback that provides parameter and input gradients.
  @derivative(of: callAsFunction)
  @usableFromInline
  func vjpCallAsFunction(_ x: Tensor) -> (value: Tensor, pullback: (Tensor) -> (TangentVector, Tensor)) {
    let (value, cache) = forwardWithCache(x)
    return (
      value,
      { upstream in
        let upstreamNCHW: Tensor
        switch self.dataFormat {
        case .nchw:
          upstreamNCHW = upstream
        case .nhwc:
          upstreamNCHW = upstream.permuted([0, 3, 1, 2])
        }

        var w = self.weight
        let tuple = withUnsafePointer(to: &w._impl) { wPtr in
          ATenCXX.TTSTensor._native_group_norm_backward(
            upstreamNCHW._impl,
            cache.inputNCHW._impl,
            cache.mean._impl,
            cache.rstd._impl,
            Int64(self.numGroups),
            wPtr
          )
        }

        let gradInputNCHW = Tensor(ATenCXX.TTSTensor._native_group_norm_backward_get0(tuple))
        let gradWeight = Tensor(ATenCXX.TTSTensor._native_group_norm_backward_get1(tuple))
        let gradBias = Tensor(ATenCXX.TTSTensor._native_group_norm_backward_get2(tuple))

        let gradInput: Tensor
        switch self.dataFormat {
        case .nchw:
          gradInput = gradInputNCHW
        case .nhwc:
          gradInput = gradInputNCHW.permuted([0, 2, 3, 1])
        }

        var tangent = TangentVector.zero
        tangent.weight = gradWeight
        tangent.bias = gradBias
        return (tangent, gradInput)
      }
    )
  }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input activations.
  ///   - context: Forward context (unused).
  /// - Returns: Normalized activations.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  /// Applies the tangent `offset` to the layer's parameters.
  /// - Parameter offset: Tangent vector produced by differentiation.
  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
    bias.move(by: offset.bias)
  }

  /// Writable key paths for trainable parameters.
  public static var parameterKeyPaths: [WritableKeyPath<GroupNorm, Tensor>] {
    [\GroupNorm.weight, \GroupNorm.bias]
  }

  /// Tangent representation for `GroupNorm`.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Tangent for the scaling factors.
    public var weight: Tensor
    /// Tangent for the bias parameters.
    public var bias: Tensor

    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init(weight: .zero, bias: .zero) }

    /// Adds two tangent vectors element-wise.
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(weight: lhs.weight.adding(rhs.weight), bias: lhs.bias.adding(rhs.bias))
    }

    /// Subtracts two tangent vectors element-wise.
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(
        weight: lhs.weight.adding(rhs.weight.multiplying(-1)),
        bias: lhs.bias.adding(rhs.bias.multiplying(-1))
      )
    }

    /// Writable key paths for the tangent components.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\.weight, \.bias]
    }
  }
}
