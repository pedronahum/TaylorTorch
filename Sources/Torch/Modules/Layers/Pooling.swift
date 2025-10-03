//
// Sources/Torch/Modules/Layers/Pooling.swift
//
// WHY
// Small, stateless layers around ATen pooling ops. They make CNN blocks ergonomic,
// compose with Sequential/Builder, and are context‑agnostic (no parameters).
//
// Notes
// - Defaults to NCHW; NHWC supported via internal transposes (same as Conv2D).
//

// Sources/Torch/Modules/Layers/Pooling.swift
import _Differentiation

public struct MaxPool2D: Layer {
  @noDerivative public var kernel: (Int, Int)
  @noDerivative public var stride: (Int, Int)
  @noDerivative public var padding: (Int, Int)
  @noDerivative public var dilation: (Int, Int)
  @noDerivative public var ceilMode: Bool
  @noDerivative public var dataFormat: DataFormat

  public init(
    kernel: (Int, Int),
    stride: (Int, Int)? = nil,
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    ceilMode: Bool = false,
    dataFormat: DataFormat = .nchw
  ) {
    self.kernel = kernel
    self.stride = stride ?? kernel
    self.padding = padding
    self.dilation = dilation
    self.ceilMode = ceilMode
    self.dataFormat = dataFormat
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let (xNCHW, toOut) = _prep(x)
    let y = xNCHW.maxPool2d(
      kernelSize: [Int64(kernel.0), Int64(kernel.1)],
      stride: [Int64(stride.0), Int64(stride.1)],
      padding: [Int64(padding.0), Int64(padding.1)],
      dilation: [Int64(dilation.0), Int64(dilation.1)],
      ceilMode: ceilMode
    )
    return toOut(y)
  }

  private func _prep(_ x: Tensor) -> (Tensor, @differentiable(reverse) (Tensor) -> Tensor) {
    switch dataFormat {
    case .nchw: return (x, { $0 })
    case .nhwc:
      return (
        x.transposed(1, 3).transposed(2, 3),
        { $0.transposed(2, 3).transposed(1, 3) }
      )
    }
  }

  // --- Boilerplate ---
  public static var parameterKeyPaths: [WritableKeyPath<MaxPool2D, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    // ✅ FIX: Added the missing property required by ParameterIterable.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  public mutating func move(by offset: TangentVector) {}
}

public struct AvgPool2D: Layer {
  @noDerivative public var kernel: (Int, Int)
  @noDerivative public var stride: (Int, Int)
  @noDerivative public var padding: (Int, Int)
  @noDerivative public var ceilMode: Bool
  @noDerivative public var dataFormat: DataFormat

  public init(
    kernel: (Int, Int),
    stride: (Int, Int)? = nil,
    padding: (Int, Int) = (0, 0),
    ceilMode: Bool = false,
    dataFormat: DataFormat = .nchw
  ) {
    self.kernel = kernel
    self.stride = stride ?? kernel
    self.padding = padding
    self.ceilMode = ceilMode
    self.dataFormat = dataFormat
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let (xNCHW, toOut) = _prep(x)
    let y = xNCHW.avgPool2d(
      kernelSize: [Int64(kernel.0), Int64(kernel.1)],
      stride: [Int64(stride.0), Int64(stride.1)],
      padding: [Int64(padding.0), Int64(padding.1)],
      ceilMode: ceilMode
    )
    return toOut(y)
  }

  private func _prep(_ x: Tensor) -> (Tensor, @differentiable(reverse) (Tensor) -> Tensor) {
    switch dataFormat {
    case .nchw: return (x, { $0 })
    case .nhwc:
      return (
        x.transposed(1, 3).transposed(2, 3),
        { $0.transposed(2, 3).transposed(1, 3) }
      )
    }
  }

  // --- Boilerplate ---
  public static var parameterKeyPaths: [WritableKeyPath<AvgPool2D, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    // ✅ FIX: Added the missing property required by ParameterIterable.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  public mutating func move(by offset: TangentVector) {}
}
