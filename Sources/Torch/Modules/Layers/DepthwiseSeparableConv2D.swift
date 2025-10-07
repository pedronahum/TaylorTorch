import _Differentiation

/// Mobile-style separable convolution:
/// `y = activation( pointwise( depthwise(x) ) )`
///
/// - `depthwise`: kH×kW convolution with `groups = inC`, `outC = inC`
/// - `pointwise`: 1×1 convolution with `groups = 1`, `outC = desired`
///
/// Notes:
/// - Stride, padding and dilation apply to the depthwise stage (pointwise uses stride 1).
/// - Works with `.nchw` (default) and `.nhwc` just like `Conv2D`.
public struct DepthwiseSeparableConv2D: Layer {
  // MARK: Activation (closure-free to avoid toolchain issues)
  @frozen
  public enum ActivationKind: Sendable, Equatable {
    case identity
    case relu
    case tanh
    case sigmoid
    case exp
    case log
    case sqrt
    case sin
    case tan
    case asin
    case acos
    case atan
    case sinh
    case cosh
    case asinh
    case acosh
    case atanh
    case erf
    case erfc

  }

  // MARK: Stored state
  public var depthwise: Conv2D
  public var pointwise: Conv2D
  @noDerivative public var activation: ActivationKind

  // MARK: Inits

  /// Designated initializer from existing convs and an activation kind.
  @inlinable
  public init(depthwise: Conv2D, pointwise: Conv2D, activation: ActivationKind) {
    self.depthwise = depthwise
    self.pointwise = pointwise
    self.activation = activation
  }

  /// Convenience: identity activation.
  @inlinable
  public init(depthwise: Conv2D, pointwise: Conv2D) {
    self.init(depthwise: depthwise, pointwise: pointwise, activation: .identity)
  }

  /// Designated initializer that builds the depthwise & pointwise stages.
  /// No default-arguments here to avoid cross-module default-arg generators.
  @inlinable
  public init(
    inC: Int,
    outC: Int,
    kH: Int,
    kW: Int,
    stride: (Int, Int),
    padding: Padding,
    dilation: (Int, Int),
    dtype: DType,
    device: Device,
    dataFormat: DataFormat,
    activation: ActivationKind
  ) {
    // Depthwise: outC = inC, groups = inC
    self.depthwise = Conv2D.kaimingUniform(
      inC: inC,
      outC: inC,
      kH: kH,
      kW: kW,
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: inC,
      dtype: dtype,
      device: device,
      dataFormat: dataFormat
    )
    // Pointwise: 1×1, groups = 1, stride 1, valid padding
    self.pointwise = Conv2D.kaimingUniform(
      inC: inC,
      outC: outC,
      kH: 1,
      kW: 1,
      stride: (1, 1),
      padding: .valid,
      dilation: (1, 1),
      groups: 1,
      dtype: dtype,
      device: device,
      dataFormat: dataFormat
    )
    self.activation = activation
  }

  /// Convenience: defaults to `.float32/.cpu`, identity activation, `.nchw`.
  @inlinable
  public init(
    inC: Int,
    outC: Int,
    kH: Int,
    kW: Int,
    stride: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilation: (Int, Int) = (1, 1),
    dataFormat: DataFormat = .nchw
  ) {
    self.init(
      inC: inC,
      outC: outC,
      kH: kH,
      kW: kW,
      stride: stride,
      padding: padding,
      dilation: dilation,
      dtype: .float32,
      device: .cpu,
      dataFormat: dataFormat,
      activation: .identity
    )
  }

  // MARK: Forward

  @inlinable
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    applyActivation(pointwise(depthwise(x)))
  }

  @inlinable
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    applyActivation(pointwise.call(depthwise.call(x, context: context), context: context))
  }

  @inlinable
  @differentiable(reverse)
  func applyActivation(_ z: Tensor) -> Tensor {
    switch activation {
    case .identity: return z
    case .relu: return z.relu()
    case .log: return z.log()
    case .sqrt: return z.sqrt()
    case .sin: return z.sin()
    case .tan: return z.tan()
    case .asin: return z.asin()
    case .acos: return z.acos()
    case .atan: return z.atan()
    case .sinh: return z.sinh()
    case .cosh: return z.cosh()
    case .asinh: return z.asinh()
    case .acosh: return z.acosh()
    case .atanh: return z.atanh()
    case .erf: return z.erf()
    case .erfc: return z.erfc()
    case .tanh: return z.tanh()
    case .sigmoid: return z.sigmoid()
    case .exp: return z.exp()

    }
  }

  // MARK: Parameters & AD wiring

  @inlinable
  public mutating func move(by offset: TangentVector) {
    depthwise.move(by: offset.depthwise)
    pointwise.move(by: offset.pointwise)
  }

  @inlinable
  public static var parameterKeyPaths: [WritableKeyPath<DepthwiseSeparableConv2D, Tensor>] {
    var paths: [WritableKeyPath<DepthwiseSeparableConv2D, Tensor>] = []
    for kp in Conv2D.parameterKeyPaths {
      paths.append((\Self.depthwise).appending(kp))
    }
    for kp in Conv2D.parameterKeyPaths {
      paths.append((\Self.pointwise).appending(kp))
    }
    return paths
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var depthwise: Conv2D.TangentVector
    public var pointwise: Conv2D.TangentVector

    @inlinable public init(depthwise: Conv2D.TangentVector, pointwise: Conv2D.TangentVector) {
      self.depthwise = depthwise
      self.pointwise = pointwise
    }
    @inlinable public static var zero: TangentVector {
      .init(depthwise: .zero, pointwise: .zero)
    }
    @inlinable public static func + (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(depthwise: l.depthwise + r.depthwise, pointwise: l.pointwise + r.pointwise)
    }
    @inlinable public static func - (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(depthwise: l.depthwise - r.depthwise, pointwise: l.pointwise - r.pointwise)
    }
    @inlinable
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      var paths: [WritableKeyPath<TangentVector, Tensor>] = []
      for kp in Conv2D.TangentVector.parameterKeyPaths {
        paths.append((\Self.depthwise).appending(kp))
      }
      for kp in Conv2D.TangentVector.parameterKeyPaths {
        paths.append((\Self.pointwise).appending(kp))
      }
      return paths
    }
  }
}
