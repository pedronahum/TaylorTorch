@preconcurrency import ATenCXX  // ✅ Fix #1: Mark the import as safe for concurrency

/// A multi-dimensional array of elements with automatic differentiation support.
///
/// `Tensor` is the fundamental data structure in TaylorTorch, representing n-dimensional arrays
/// that can store numerical data and track gradients for automatic differentiation. Tensors provide
/// efficient storage and computation on CPUs, CUDA GPUs, and Metal devices (MPS).
///
/// ## Creating Tensors
///
/// Create tensors using factory methods or initializers:
///
/// ```swift
/// // From a shape with zeros
/// let zeros = Tensor.zeros(shape: [2, 3], dtype: .float32)
///
/// // Random initialization
/// let random = Tensor.randn(shape: [10, 20], dtype: .float32)
///
/// // From a scalar
/// let scalar = Tensor(3.14, dtype: .float32)
///
/// // From an array
/// let array = Tensor([[1, 2, 3], [4, 5, 6]], dtype: .float32)
/// ```
///
/// ## Tensor Properties
///
/// Access tensor metadata through properties:
///
/// ```swift
/// let t = Tensor.randn([2, 3, 4])
/// print(t.shape)   // [2, 3, 4]
/// print(t.rank)    // 3
/// print(t.dtype)   // .float32
/// print(t.device)  // .cpu
/// ```
///
/// ## Element-wise Operations
///
/// Tensors support standard arithmetic operations with automatic broadcasting:
///
/// ```swift
/// let x = Tensor.ones([2, 3])
/// let y = Tensor.full(2.0, shape: [2, 3])
///
/// let sum = x + y      // [3.0, 3.0, ...]
/// let product = x * y  // [2.0, 2.0, ...]
/// let scaled = x * 5.0 // Broadcasting
/// ```
///
/// ## Automatic Differentiation
///
/// Tensors integrate seamlessly with Swift's automatic differentiation:
///
/// ```swift
/// @differentiable
/// func model(_ input: Tensor) -> Tensor {
///     let w = Tensor.randn([784, 10])
///     return input.matmul(w)
/// }
///
/// let input = Tensor.randn([32, 784])
/// let (output, pullback) = valueWithPullback(at: input) { x in model(x) }
/// let gradient = pullback(Tensor.ones(output.shape))
/// ```
///
/// ## Device Management
///
/// Move tensors between devices for optimal performance:
///
/// ```swift
/// let cpu = Tensor.ones([1000, 1000], device: .cpu)
/// let gpu = cpu.to(device: .cuda(0))  // Move to GPU
/// let mps = cpu.to(device: .mps)      // Metal on macOS
/// ```
///
/// ## Topics
///
/// ### Creating Tensors
///
/// - ``Tensor/empty(shape:dtype:device:)``
/// - ``Tensor/zeros(shape:dtype:device:)``
/// - ``Tensor/ones(shape:dtype:device:)``
/// - ``Tensor/full(_:shape:device:)``
/// - ``Tensor/init(_:device:)``
/// - ``Tensor/init(_:dtype:device:)``
///
/// ### Tensor Properties
///
/// - ``Tensor/shape``
/// - ``Tensor/rank``
/// - ``Tensor/dtype``
/// - ``Tensor/device``
///
/// ### Device Operations
///
/// - ``Tensor/to(dtype:)``
/// - ``Tensor/to(device:)``
///
/// ### Arithmetic Operations
///
/// - ``Tensor/adding(_:alpha:)``
/// - ``Tensor/adding(_:)-5n7g4``
///
/// ## See Also
///
/// - ``DType``
/// - ``Device``
/// - ``Layer``
public struct Tensor: Sendable {
  /// Owning handle to the underlying C++ tensor implementation.
  @usableFromInline
  var _impl: TTSTensor

  /// Wraps an existing `TTSTensor` produced by the C++ layer.
  @inlinable public init(_ impl: TTSTensor) { self._impl = impl }
}

extension Tensor {
  // MARK: Factories

  /// Creates an uninitialized tensor with the specified shape, data type, and device.
  ///
  /// The tensor's values are uninitialized and contain arbitrary data. Use this method when you plan
  /// to immediately fill the tensor with values, as it's slightly more efficient than allocating
  /// and initializing with zeros.
  ///
  /// - Parameters:
  ///   - shape: The size of each dimension in row-major order. For example, `[2, 3]` creates
  ///            a 2×3 matrix.
  ///   - dtype: The data type for elements. Common values include `.float32`, `.float64`, `.int32`.
  ///   - device: The device where the tensor should be allocated. Defaults to `.cpu`.
  ///
  /// - Returns: A new tensor with uninitialized values.
  ///
  /// ```swift
  /// // Create a 3x4 matrix (uninitialized)
  /// var t = Tensor.empty(shape: [3, 4], dtype: .float32)
  ///
  /// // Fill it with values
  /// for i in 0..<3 {
  ///     for j in 0..<4 {
  ///         t[i, j] = Tensor(Float(i * 4 + j))
  ///     }
  /// }
  /// ```
  ///
  /// - Note: Values in an empty tensor are unpredictable. Always initialize before reading.
  ///
  /// ## See Also
  /// - ``zeros(shape:dtype:device:)``
  /// - ``ones(shape:dtype:device:)``
  public static func empty(shape: [Int], dtype: DType, device: Device = .cpu) -> Tensor {
    var sizes64 = shape.map { Int64($0) }
    return Tensor(TTSTensor.empty(&sizes64, sizes64.count, dtype._c10, device._c10))
  }

  /// Creates a tensor filled with zeros.
  ///
  /// This is one of the most commonly used tensor creation methods. All elements are initialized
  /// to zero.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor. For example, `[2, 3, 4]` creates a 2×3×4 tensor.
  ///   - dtype: The data type for elements. Defaults to `.float32`.
  ///   - device: The device where the tensor resides. Defaults to `.cpu`.
  ///
  /// - Returns: A new tensor with all elements set to zero.
  ///
  /// ```swift
  /// // Create a 3x3 zero matrix
  /// let zeros = Tensor.zeros(shape: [3, 3], dtype: .float32)
  /// // [[0.0, 0.0, 0.0],
  /// //  [0.0, 0.0, 0.0],
  /// //  [0.0, 0.0, 0.0]]
  ///
  /// // Create a batch of zero vectors on GPU
  /// let batch = Tensor.zeros(shape: [32, 128], dtype: .float32, device: .cuda(0))
  /// ```
  ///
  /// - Note: Commonly used for initializing biases or creating mask tensors.
  ///
  /// ## See Also
  /// - ``ones(shape:dtype:device:)``
  /// - ``full(_:shape:device:)``
  public static func zeros(shape: [Int], dtype: DType, device: Device = .cpu) -> Tensor {
    var sizes64 = shape.map { Int64($0) }
    return Tensor(TTSTensor.zeros(&sizes64, sizes64.count, dtype._c10, device._c10))
  }

  /// Creates a tensor filled with ones.
  ///
  /// All elements are initialized to the value 1.
  ///
  /// - Parameters:
  ///   - shape: The dimensions of the tensor.
  ///   - dtype: The data type for elements. Defaults to `.float32`.
  ///   - device: The device where the tensor resides. Defaults to `.cpu`.
  ///
  /// - Returns: A new tensor with all elements set to one.
  ///
  /// ```swift
  /// // Create a vector of ones
  /// let ones = Tensor.ones(shape: [5], dtype: .float32)
  /// // [1.0, 1.0, 1.0, 1.0, 1.0]
  ///
  /// // Create a batch of ones for masking
  /// let mask = Tensor.ones(shape: [32, 100], dtype: .float32)
  /// ```
  ///
  /// - Note: Useful for initialization and creating constant tensors.
  ///
  /// ## See Also
  /// - ``zeros(shape:dtype:device:)``
  /// - ``full(_:shape:device:)``
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
    if self.count == 0 { return other }
    if other.count == 0 { return self }
    return Tensor(_impl.add(other._impl, alpha._cxxScalar))
  }

  // Before:
  // func adding(_ scalar: Scalar) -> Tensor { ... }

  // After (✅ Make it generic):
  /// Returns the element-wise sum of `self` and a scalar broadcast across every element.
  public func adding<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.addScalar(scalar._cxxScalar))
  }
}
