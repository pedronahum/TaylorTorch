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

/// Max pooling layer for rank-4 tensors.
public struct MaxPool2D: Layer {
  @noDerivative public var kernel: (Int, Int)
  @noDerivative public var stride: (Int, Int)
  @noDerivative public var padding: (Int, Int)
  @noDerivative public var dilation: (Int, Int)
  @noDerivative public var ceilMode: Bool
  @noDerivative public var dataFormat: DataFormat

  /// Creates a 2D max-pooling layer.
  /// - Parameters:
  ///   - kernel: Spatial extent of the pooling window.
  ///   - stride: Step between pooling windows. Defaults to `kernel`.
  ///   - padding: Implicit zero padding applied to height and width.
  ///   - dilation: Spacing inside the pooling window.
  ///   - ceilMode: Whether to use ceil when computing the output size.
  ///   - dataFormat: Input tensor layout.
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

  /// Applies max pooling to `x`.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor containing the pooled features.
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

  /// Normalizes layout and produces a closure to restore the original format.
  /// - Parameter x: Input tensor.
  /// - Returns: Tuple containing the NCHW tensor and a closure converting back to `dataFormat`.
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
  /// MaxPool2D exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<MaxPool2D, Tensor>] { [] }
  /// Tangent representation for `MaxPool2D`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    // ✅ FIX: Added the missing property required by ParameterIterable.
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  /// MaxPool2D has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
}

/// Average pooling layer for rank-4 tensors.
public struct AvgPool2D: Layer {
  @noDerivative public var kernel: (Int, Int)
  @noDerivative public var stride: (Int, Int)
  @noDerivative public var padding: (Int, Int)
  @noDerivative public var ceilMode: Bool
  @noDerivative public var dataFormat: DataFormat

  /// Creates a 2D average-pooling layer.
  /// - Parameters:
  ///   - kernel: Spatial extent of the pooling window.
  ///   - stride: Step between pooling windows. Defaults to `kernel`.
  ///   - padding: Implicit zero padding applied to height and width.
  ///   - ceilMode: Whether to use ceil when computing the output size.
  ///   - dataFormat: Input tensor layout.
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

  /// Applies average pooling to `x`.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor containing the pooled features.
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

  /// Normalizes layout and produces a closure to restore the original format.
  /// - Parameter x: Input tensor.
  /// - Returns: Tuple containing the NCHW tensor and a closure converting back to `dataFormat`.
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
  /// AvgPool2D exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<AvgPool2D, Tensor>] { [] }
  /// Tangent representation for `AvgPool2D`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    // ✅ FIX: Added the missing property required by ParameterIterable.
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  /// AvgPool2D has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
}
