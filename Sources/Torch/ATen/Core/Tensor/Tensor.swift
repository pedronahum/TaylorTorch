@preconcurrency import ATenCXX  // ✅ Fix #1: Mark the import as safe for concurrency

/// Swift-idiomatic façade over the C++ `TTSTensor` wrapper.
/// Provides value semantics and convenience APIs while delegating heavy lifting to ATen.
public struct Tensor: Sendable {
  /// Owning handle to the underlying C++ tensor implementation.
  @usableFromInline
  var _impl: TTSTensor

  /// Wraps an existing `TTSTensor` produced by the C++ layer.
  @inlinable public init(_ impl: TTSTensor) { self._impl = impl }
}

extension Tensor {
  // MARK: Factories

  /// Creates an uninitialized tensor with the given shape, dtype, and device.
  /// - Parameters:
  ///   - shape: The size of each tensor dimension in row-major order.
  ///   - dtype: Torch dtype that determines storage precision.
  ///   - device: Execution device; defaults to `.cpu`.
  public static func empty(shape: [Int], dtype: DType, device: Device = .cpu) -> Tensor {
    var sizes64 = shape.map { Int64($0) }
    return Tensor(TTSTensor.empty(&sizes64, sizes64.count, dtype._c10, device._c10))
  }

  /// Creates a tensor filled with zeros using the provided shape, dtype, and device.
  public static func zeros(shape: [Int], dtype: DType, device: Device = .cpu) -> Tensor {
    var sizes64 = shape.map { Int64($0) }
    return Tensor(TTSTensor.zeros(&sizes64, sizes64.count, dtype._c10, device._c10))
  }

  /// Creates a tensor filled with ones using the provided shape, dtype, and device.
  public static func ones(shape: [Int], dtype: DType, device: Device = .cpu) -> Tensor {
    var sizes64 = shape.map { Int64($0) }
    return Tensor(TTSTensor.ones(&sizes64, sizes64.count, dtype._c10, device._c10))
  }

  /// Creates a tensor filled with a single scalar value.
  /// - Parameters:
  ///   - value: Scalar value to broadcast across the tensor.
  ///   - shape: Desired dimensions for the result tensor.
  ///   - device: Execution device; defaults to `.cpu`.
  public static func full<T: TorchArithmetic>(
    _ value: T, shape: [Int], device: Device = .cpu
  ) -> Tensor {
    var sizes64 = shape.map { Int64($0) }
    // ✅ Back to using the property, which will now work
    return Tensor(
      TTSTensor.full(value._cxxScalar, &sizes64, sizes64.count, T.torchDType._c10, device._c10))
  }

  /// Creates a rank-0 tensor that stores the provided scalar on the target device.
  public init<T: TorchArithmetic>(_ scalar: T, device: Device = .cpu) {
    // ✅ Back to using the property
    self._impl = TTSTensor.fromScalar(scalar._cxxScalar, T.torchDType._c10, device._c10)
  }

  /// Creates a scalar tensor with the requested dtype rather than the scalar's default.
  public init<T: TorchArithmetic>(_ scalar: T, dtype: DType, device: Device = .cpu) {
    self.init(scalar, device: device)
    if let current = self.dtype, current != dtype {
      self = self.to(dtype: dtype)
    }
  }
}

extension Tensor {
  // MARK: Queries

  /// Number of logical dimensions tracked by the tensor.
  public var rank: Int { Int(_impl.dim()) }

  /// Sizes of each dimension, expressed with Swift `Int` values.
  public var shape: [Int] {
    let d = Int(_impl.dim())
    return (0..<d).map { Int(_impl.sizeAt(Int64($0))) }
  }

  /// Torch dtype describing the tensor's element type, or `nil` if it is unsupported.
  public var dtype: DType? { DType(_impl.dtype()) }

  /// Device on which the tensor's storage currently resides.
  public var device: Device {
    let dev = _impl.device()
    switch dev.type() {
    case c10.DeviceType.CPU: return .cpu
    // ✅ Fix #3: dev.index() is Int8, which now matches .cuda(Int8)
    case c10.DeviceType.CUDA: return .cuda(dev.index())
    case c10.DeviceType.MPS: return .mps
    default: return .cpu
    }
  }
}

extension Tensor {
  // MARK: Conversions

  /// Returns a copy of the tensor backed by the requested dtype.
  public func to(dtype: DType) -> Tensor {
    Tensor(_impl.toDType(dtype._c10))
  }

  /// Returns a copy of the tensor materialized on the target device.
  public func to(device: Device) -> Tensor {
    Tensor(_impl.toDevice(device._c10))
  }
}

// In Tensor.swift

extension Tensor {
  // MARK: Arithmetic (minimal)

  /// Returns the element-wise sum of `self` and `other`, scaling `other` by `alpha`.
  public func adding(_ other: Tensor, alpha: Scalar = .int64(1)) -> Tensor {
    Tensor(_impl.add(other._impl, alpha._cxxScalar))
  }

  // Before:
  // func adding(_ scalar: Scalar) -> Tensor { ... }

  // After (✅ Make it generic):
  /// Returns the element-wise sum of `self` and a scalar broadcast across every element.
  public func adding<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.addScalar(scalar._cxxScalar))
  }
}
