@preconcurrency import ATenCXX

extension Tensor {
  /// Draw samples from a uniform distribution on `[0, 1)` with the requested shape.
  /// - Parameters:
  ///   - shape: Extents for each dimension of the resulting tensor.
  ///   - dtype: Element dtype for the output (defaults to `.float32`).
  ///   - device: Execution device on which to allocate storage.
  public static func rand(shape: [Int], dtype: DType = .float32, device: Device = .cpu) -> Tensor {
    var s64 = shape.map(Int64.init)
    return Tensor(TTSTensor.rand(&s64, s64.count, dtype._c10, device._c10))
  }

  /// Draw samples from a standard normal distribution `N(0, 1)` with the requested shape.
  /// - Parameters:
  ///   - shape: Extents for each dimension of the resulting tensor.
  ///   - dtype: Element dtype for the output (defaults to `.float32`).
  ///   - device: Execution device on which to allocate storage.
  public static func randn(shape: [Int], dtype: DType = .float32, device: Device = .cpu) -> Tensor {
    var s64 = shape.map(Int64.init)
    return Tensor(TTSTensor.randn(&s64, s64.count, dtype._c10, device._c10))
  }

  /// Create an integer range tensor using the semantics of `torch.arange`.
  /// - Parameters:
  ///   - start: Inclusive starting value.
  ///   - end: Exclusive upper bound.
  ///   - step: Increment applied between values.
  ///   - dtype: Optional override for the tensor dtype; defaults to the scalar type of `T`.
  ///   - device: Execution device for the resulting tensor.
  public static func arange<T: TorchInteger & TorchArithmetic & ExpressibleByIntegerLiteral>(
    _ start: T, to end: T, step: T, dtype: DType? = nil, device: Device = .cpu
  ) -> Tensor {
    let dt = (dtype ?? T.torchDType)._c10
    return Tensor(
      TTSTensor.arange(start._cxxScalar, end._cxxScalar, step._cxxScalar, dt, device._c10))
  }

  /// Create a floating-point range tensor using the semantics of `torch.arange`.
  /// - Parameters:
  ///   - start: Inclusive starting value.
  ///   - end: Exclusive upper bound.
  ///   - step: Increment applied between values.
  ///   - dtype: Optional override for the tensor dtype; defaults to the scalar type of `T`.
  ///   - device: Execution device for the resulting tensor.
  public static func arange<T: TorchFloating & TorchArithmetic & ExpressibleByFloatLiteral>(
    _ start: T, to end: T, step: T, dtype: DType? = nil, device: Device = .cpu
  ) -> Tensor {
    let dt = (dtype ?? T.torchDType)._c10
    return Tensor(
      TTSTensor.arange(start._cxxScalar, end._cxxScalar, step._cxxScalar, dt, device._c10))
  }

  /// Create an evenly spaced grid between `start` and `end` (inclusive).
  /// - Parameters:
  ///   - start: Starting value for the range.
  ///   - end: Final value for the range (included in the result).
  ///   - steps: Number of samples to generate (must be > 0).
  ///   - dtype: Element dtype for the output (defaults to `.float32`).
  ///   - device: Execution device on which to allocate storage.
  public static func linspace<T: BinaryFloatingPoint>(
    start: T, end: T, steps: Int, dtype: DType = .float32, device: Device = .cpu
  ) -> Tensor {
    Tensor(TTSTensor.linspace(Double(start), Double(end), Int64(steps), dtype._c10, device._c10))
  }

  /// Draw samples from a uniform distribution on `[low, high)`.
  /// - Parameters:
  ///   - low: The lower bound of the distribution.
  ///   - high: The upper bound of the distribution.
  ///   - shape: Extents for each dimension of the resulting tensor.
  ///   - dtype: Element dtype for the output (defaults to `.float32`).
  ///   - device: Execution device on which to allocate storage.
  public static func uniform(
    low: Double,
    high: Double,
    shape: [Int],
    dtype: DType = .float32,
    device: Device = .cpu
  ) -> Tensor {
    // Generate a tensor with values in [0, 1)
    let r = Tensor.rand(shape: shape, dtype: dtype, device: device)
    // Scale and shift to the desired [low, high) range
    return r.multiplying(high - low).adding(low)
  }

}
