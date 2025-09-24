/// Mirrors the set of Torch scalar dtypes that can flow through the Swift API.
/// The raw value is kept stable to match the canonical ATen identifiers where possible.
public enum DType: UInt16, Sendable, Hashable, Codable {
  // Booleans & text
  case bool
  case string
  case variant

  // Signed integers
  case int8, int16, int32, int64

  // Unsigned integers
  case uint8, uint16, uint32, uint64

  // IEEE-like floats
  case float16   // IEEE fp16
  case bfloat16  // Brain floating point
  case float32
  case float64

  // Complex
  case complex64    // (Float, Float)
  case complex128   // (Double, Double)

  // Quantized integers
  case qint8, qint16, qint32
  case quint8, quint16
}

public extension DType {
  /// Returns `true` when this dtype represents any fixed-width integer.
  var isInteger: Bool {
    switch self {
    case .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64: return true
    default: return false
    }
  }

  /// Returns `true` when this dtype represents an unsigned integer.
  var isUnsigned: Bool {
    switch self {
    case .uint8, .uint16, .uint32, .uint64: return true
    default: return false
    }
  }

  /// Returns `true` when this dtype represents a floating-point value.
  var isFloating: Bool {
    switch self {
    case .float16, .bfloat16, .float32, .float64: return true
    default: return false
    }
  }

  /// Returns `true` when this dtype represents a complex number payload.
  var isComplex: Bool {
    self == .complex64 || self == .complex128
  }

  /// Returns `true` when this dtype represents one of the quantized integer formats.
  var isQuantized: Bool {
    switch self {
    case .qint8, .qint16, .qint32, .quint8, .quint16: return true
    default: return false
    }
  }
}
