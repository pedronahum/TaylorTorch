import ATenCXX

// MARK: - Bool

/// Makes `Bool` available as a tensor scalar and bridges it to the Torch dtype catalog.
extension Bool: TorchBoolScalar, TorchTensorScalar {
  public static var torchDType: DType { .bool }
}

// MARK: - Signed integers

/// Registers `Int8` as a signed integer scalar with Torch interop support.
extension Int8: TorchSignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic,
  TorchTensorScalar
{
  public static var torchDType: DType { .int8 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

/// Registers `Int16` as a signed integer scalar with Torch interop support.
extension Int16: TorchSignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic,
  TorchSliceIndex, TorchTensorScalar
{
  public static var torchDType: DType { .int16 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

/// Registers `Int32` as a signed integer scalar with Torch interop support.
extension Int32: TorchSignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic, TorchIndex,
  TorchSliceIndex, TorchFloatingOrIndex, TorchTensorScalar
{
  public static var torchDType: DType { .int32 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

/// Registers `Int64` as a signed integer scalar with Torch interop support.
extension Int64: TorchSignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic, TorchIndex,
  TorchSliceIndex, TorchInt64OrString, TorchFloatingOrIndex, TorchTensorScalar
{
  public static var torchDType: DType { .int64 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  /// On Linux, Int64 can be ambiguous (long vs long long), so we explicitly cast to CLongLong
  public var _cxxScalar: c10.Scalar { c10.Scalar(CLongLong(self)) }
}

/*
/// Registers `Int` as a signed integer scalar with Torch interop support.
extension Int: TorchSliceIndex { // Note: Removed FixedWidthInteger, it's redundant
    public static var torchDType: DType { .int64 }
}
*/

extension Int: TorchSignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic, TorchSliceIndex {
  public static var torchDType: DType { .int64 }
  /// On Linux, Int64 can be ambiguous (long vs long long), so we explicitly cast to CLongLong
  public var _cxxScalar: c10.Scalar { c10.Scalar(CLongLong(Int64(self))) }
}

// MARK: - Unsigned integers

/// Registers `UInt8` as an unsigned integer scalar with Torch interop support.
extension UInt8: TorchUnsignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic,
  TorchTensorScalar
{
  public static var torchDType: DType { .uint8 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

/// Registers `UInt16` as an unsigned integer scalar with Torch interop support.
extension UInt16: TorchUnsignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic,
  TorchTensorScalar
{
  public static var torchDType: DType { .uint16 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

/// Registers `UInt32` as an unsigned integer scalar with Torch interop support.
extension UInt32: TorchUnsignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic,
  TorchTensorScalar
{
  public static var torchDType: DType { .uint32 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

/// Registers `UInt64` as an unsigned integer scalar with Torch interop support.
extension UInt64: TorchUnsignedInteger, TorchInteger, TorchRealNumber, TorchArithmetic,
  TorchTensorScalar
{
  public static var torchDType: DType { .uint64 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

// MARK: - IEEE floats

/// Registers `Float16` as a Torch floating-point scalar while promoting to `Float` when bridging to C++.
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: TorchFloating, TorchRealNumber, TorchArithmetic,
  TorchFloatOrHalf, TorchFloatingNoDouble, TorchFloatHalfOrDouble,
  TorchFloatingOrIndex, TorchAnyNumericOrQuantized, TorchTensorScalar
{
  public static var torchDType: DType { .float16 }
  /// Promotes the value to `Float` before constructing the Torch scalar.
  public var _cxxScalar: c10.Scalar {
    // Promote Float16 to Float and initialize the Scalar directly.
    return c10.Scalar(Float(self))
  }
}

/// Registers `Float` as a Torch floating-point scalar with seamless C++ bridging.
extension Float: TorchFloating, TorchRealNumber, TorchArithmetic,
  TorchFloatOrDouble, TorchFloatingOrComplex, TorchFloatOrHalf,
  TorchFloatingNoDouble, TorchFloatHalfOrDouble,
  TorchFloatingOrIndex, TorchFloatOrInt64OrString,
  TorchAnyNumericOrQuantized, TorchTensorScalar
{
  public static var torchDType: DType { .float32 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}

/// Registers `Double` as a Torch floating-point scalar with seamless C++ bridging.
extension Double: TorchFloating, TorchRealNumber, TorchArithmetic,
  TorchFloatOrDouble, TorchFloatingOrComplex,
  TorchFloatHalfOrDouble,
  TorchAnyNumericOrQuantized, TorchTensorScalar
{
  public static var torchDType: DType { .float64 }
  /// Converts the Swift value into a Torch `c10::Scalar`.
  public var _cxxScalar: c10.Scalar { c10.Scalar(self) }
}
