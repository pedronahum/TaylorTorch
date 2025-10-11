import Foundation
import _Differentiation

// MARK: - Conv2D
public struct Conv2D: Layer {
  // Trainable parameters
  public var weight: Tensor  // [Cout, Cin/groups, kH, kW]
  public var bias: Tensor  // [Cout]

  // Hyperparameters (non-diff)
  @noDerivative public let stride: (Int, Int)
  @noDerivative public let padding: (Int, Int)
  @noDerivative public let dilation: (Int, Int)
  @noDerivative public let groups: Int

  public typealias Input = Tensor
  public typealias Output = Tensor

  // Manual TangentVector to avoid nested synthesis issues.
  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var weight: Tensor
    public var bias: Tensor

    public init(weight: Tensor = Tensor(0), bias: Tensor = Tensor(0)) {
      self.weight = weight
      self.bias = bias
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight + rhs.weight, bias: lhs.bias + rhs.bias)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight - rhs.weight, bias: lhs.bias - rhs.bias)
    }
  }

  public mutating func move(by d: TangentVector) {
    weight += d.weight
    bias += d.bias
  }

  /// Kaiming/Glorot-style uniform init adapted to conv2d shapes.
  /// fanIn = Cin/groups * kH * kW; fanOut = Cout * kH * kW
  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: (Int, Int),
    stride: (Int, Int) = (1, 1),
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    groups: Int = 1,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    precondition(inChannels % groups == 0, "inChannels must be divisible by groups")
    let (kH, kW) = kernelSize
    let fanIn = inChannels / groups * kH * kW
    let fanOut = outChannels * kH * kW
    let a = Float(Foundation.sqrt(6.0 / Double(fanIn + fanOut)))

    self.weight = Tensor.uniform(
      low: -Double(a), high: Double(a),
      shape: [outChannels, inChannels / groups, kH, kW],
      dtype: dtype, device: device
    )
    self.bias = Tensor.zeros(shape: [outChannels], dtype: dtype, device: device)

    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let s = withoutDerivative(at: [Int64(stride.0), Int64(stride.1)])
    let p = withoutDerivative(at: [Int64(padding.0), Int64(padding.1)])
    let d = withoutDerivative(at: [Int64(dilation.0), Int64(dilation.1)])
    let g = withoutDerivative(at: Int64(groups))
    return x.conv2d(weight: weight, bias: bias, stride: s, padding: p, dilation: d, groups: g)
  }
}

// Avoid “curried self” path using a free closure in the VJP.
extension Conv2D {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector))
  {
    func primal(_ s: Conv2D, _ i: Tensor) -> Tensor {
      let str = withoutDerivative(at: [Int64(s.stride.0), Int64(s.stride.1)])
      let pad = withoutDerivative(at: [Int64(s.padding.0), Int64(s.padding.1)])
      let dil = withoutDerivative(at: [Int64(s.dilation.0), Int64(s.dilation.1)])
      let grp = withoutDerivative(at: Int64(s.groups))
      return i.conv2d(
        weight: s.weight, bias: s.bias, stride: str, padding: pad, dilation: dil, groups: grp)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (y, pb)
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - MaxPool2D
public struct MaxPool2D: ParameterlessLayer {
  @noDerivative public let kernelSize: (Int, Int)
  @noDerivative public let stride: (Int, Int)
  @noDerivative public let padding: (Int, Int)
  @noDerivative public let dilation: (Int, Int)
  @noDerivative public let ceilMode: Bool

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  public init(
    kernelSize: (Int, Int),
    stride: (Int, Int)? = nil,
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    ceilMode: Bool = false
  ) {
    self.kernelSize = kernelSize
    self.stride = stride ?? kernelSize
    self.padding = padding
    self.dilation = dilation
    self.ceilMode = ceilMode
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let ks = withoutDerivative(at: [Int64(kernelSize.0), Int64(kernelSize.1)])
    let st = withoutDerivative(at: [Int64(stride.0), Int64(stride.1)])
    let pd = withoutDerivative(at: [Int64(padding.0), Int64(padding.1)])
    let dl = withoutDerivative(at: [Int64(dilation.0), Int64(dilation.1)])
    let cm = withoutDerivative(at: ceilMode)
    return x.maxPool2d(kernelSize: ks, stride: st, padding: pd, dilation: dl, ceilMode: cm)
  }
}

// MARK: - AvgPool2D
public struct AvgPool2D: ParameterlessLayer {
  @noDerivative public let kernelSize: (Int, Int)
  @noDerivative public let stride: (Int, Int)
  @noDerivative public let padding: (Int, Int)
  @noDerivative public let ceilMode: Bool

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  public init(
    kernelSize: (Int, Int),
    stride: (Int, Int)? = nil,
    padding: (Int, Int) = (0, 0),
    ceilMode: Bool = false
  ) {
    self.kernelSize = kernelSize
    self.stride = stride ?? kernelSize
    self.padding = padding
    self.ceilMode = ceilMode
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let ks = withoutDerivative(at: [Int64(kernelSize.0), Int64(kernelSize.1)])
    let st = withoutDerivative(at: [Int64(stride.0), Int64(stride.1)])
    let pd = withoutDerivative(at: [Int64(padding.0), Int64(padding.1)])
    let cm = withoutDerivative(at: ceilMode)
    return x.avgPool2d(kernelSize: ks, stride: st, padding: pd, ceilMode: cm)
  }
}

// MARK: - MaxPool2D VJP (non-recursive)
extension MaxPool2D {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (EmptyTangentVector, Tensor))
  {
    // <-- do not call s(i) here
    func primal(_ s: MaxPool2D, _ i: Tensor) -> Tensor {
      let ks = withoutDerivative(at: [Int64(s.kernelSize.0), Int64(s.kernelSize.1)])
      let st = withoutDerivative(at: [Int64(s.stride.0), Int64(s.stride.1)])
      let pd = withoutDerivative(at: [Int64(s.padding.0), Int64(s.padding.1)])
      let dl = withoutDerivative(at: [Int64(s.dilation.0), Int64(s.dilation.1)])
      let cm = withoutDerivative(at: s.ceilMode)
      return i.maxPool2d(kernelSize: ks, stride: st, padding: pd, dilation: dl, ceilMode: cm)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (EmptyTangentVector(), dx)
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> EmptyTangentVector)
  {
    let (y, _) = _vjpCallAsFunction(x)
    return (y, { _ in EmptyTangentVector() })
  }
}

// MARK: - AvgPool2D VJP (non-recursive)
extension AvgPool2D {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (EmptyTangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: AvgPool2D, _ i: Tensor) -> Tensor {
      let ks = withoutDerivative(at: [Int64(s.kernelSize.0), Int64(s.kernelSize.1)])
      let st = withoutDerivative(at: [Int64(s.stride.0), Int64(s.stride.1)])
      let pd = withoutDerivative(at: [Int64(s.padding.0), Int64(s.padding.1)])
      let cm = withoutDerivative(at: s.ceilMode)
      return i.avgPool2d(kernelSize: ks, stride: st, padding: pd, ceilMode: cm)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (EmptyTangentVector(), dx)
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> EmptyTangentVector)
  {
    let (y, _) = _vjpCallAsFunction(x)
    return (y, { _ in EmptyTangentVector() })
  }
}

// MARK: - Flatten (handy utility)
public struct Flatten: ParameterlessLayer {
  @noDerivative public let startDim: Int
  @noDerivative public let endDim: Int

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  public init(startDim: Int = 1, endDim: Int = -1) {
    self.startDim = startDim
    self.endDim = endDim
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    x.flattened(startDim: startDim, endDim: endDim)
  }
}

extension Conv2D {
  /// Which fan to use in Kaiming initialization.
  public enum FanMode { case fanIn, fanOut }

  /// Nonlinearity hint for Kaiming gain.
  public enum KaimingNonlinearity: Equatable {
    case relu
    case leakyReLU(negativeSlope: Float)
    case linear  // treat as gain = 1
  }

  @inlinable
  internal static func _kaimingGain(_ nl: KaimingNonlinearity) -> Float {
    switch nl {
    case .relu:
      return Float(2).squareRoot()  // √2
    case .leakyReLU(let a):
      // √(2 / (1 + a^2))
      let denom = 1.0 + Double(a) * Double(a)
      return Float((2.0 / denom).squareRoot())
    case .linear:
      return 1
    }
  }

  /// Kaiming (He) **uniform** initializer for conv2d weights.
  ///
  /// - Parameters:
  ///   - inChannels:  Input channels (must be divisible by `groups`).
  ///   - outChannels: Output channels.
  ///   - kernelSize:  `(kH, kW)`.
  ///   - stride:      (default `(1,1)`).
  ///   - padding:     (default `(0,0)`).
  ///   - dilation:    (default `(1,1)`).
  ///   - groups:      Grouped conv (default `1`).
  ///   - mode:        `.fanIn` (default) or `.fanOut`.
  ///   - nonlinearity: Activation following this layer (defaults to `.relu`).
  ///   - dtype/device: Tensor storage settings.
  public init(
    kaimingUniformInChannels inChannels: Int,
    outChannels: Int,
    kernelSize: (Int, Int),
    stride: (Int, Int) = (1, 1),
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    groups: Int = 1,
    mode: FanMode = .fanIn,
    nonlinearity: KaimingNonlinearity = .relu,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    precondition(inChannels % groups == 0, "inChannels must be divisible by groups")

    let (kH, kW) = kernelSize
    let fanIn = (inChannels / groups) * kH * kW
    let fanOut = outChannels * kH * kW
    let fan = (mode == .fanIn) ? fanIn : fanOut

    // PyTorch-style Kaiming uniform: bound = √3 * std, std = gain / √fan
    let gain = Double(Self._kaimingGain(nonlinearity))
    let std = gain / Double(fan).squareRoot()
    let bound = Foundation.sqrt(3.0) * std

    self.weight = Tensor.uniform(
      low: -bound, high: bound,
      shape: [outChannels, inChannels / groups, kH, kW],
      dtype: dtype, device: device
    )

    // Common practice: bias ~ U[-1/√fan_in, 1/√fan_in]
    let bBound = 1.0 / Double(fanIn).squareRoot()
    self.bias = Tensor.uniform(
      low: -bBound, high: bBound, shape: [outChannels], dtype: dtype, device: device)

    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
  }
}
