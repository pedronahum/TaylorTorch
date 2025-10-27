import ATenCXX

/// Swift-side wrapper around the subset of Torch scalars that can flow through
/// the Swift bindings. Each case stores the strongly typed payload while still
/// bridging back to `c10::Scalar` when needed.
public enum Scalar: Sendable {
    /// 8-bit signed integer scalar.
    case int8(Int8)
    /// 16-bit signed integer scalar.
    case int16(Int16)
    /// 32-bit signed integer scalar.
    case int32(Int32)
    /// 64-bit signed integer scalar.
    case int64(Int64)

    /// 32-bit floating point scalar.
    case float(Float)
    /// 64-bit floating point scalar.
    case double(Double)

    /// Boolean scalar backed by a single byte.
    case bool(Bool)

    // MARK: - Type Checking

    /// Returns `true` when the scalar stores one of the integer payloads.
    public var isInteger: Bool {
        switch self {
        case .int8, .int16, .int32, .int64:
            return true
        default:
            return false
        }
    }

    /// Returns `true` when the scalar stores a floating-point payload.
    public var isFloatingPoint: Bool {
        switch self {
        case .float, .double:
            return true
        default:
            return false
        }
    }

    /// Returns `true` when the scalar stores a Boolean payload.
    public var isBool: Bool {
        if case .bool = self { return true }
        return false
    }
}

internal extension Scalar {
    /// Bridges the Swift scalar back to the underlying `c10::Scalar` value.
     @usableFromInline
    var _cxxScalar: c10.Scalar {
        switch self {
        case .int8(let value):  return c10.Scalar(value)
        case .int16(let value): return c10.Scalar(value)
        case .int32(let value): return c10.Scalar(value)
        case .int64(let value):
            // Use explicit helper to avoid C++ overload ambiguity on Linux
            return make_scalar_int64(value)
        case .float(let value): return c10.Scalar(value)
        case .double(let value): return c10.Scalar(value)
        case .bool(let value):
            return c10.Scalar(value ? CUnsignedChar(1) : CUnsignedChar(0))
        }
    }

}

internal extension Scalar {
    /// Creates a Swift `Scalar` from a `c10::Scalar` when the payload type is supported.
    init?(_ cxxScalar: c10.Scalar) {
        // Use the underlying `ScalarType` for a robust conversion.
        let scalarType = cxxScalar.type()

        // ✅ Corrected: Use the 'c10' namespace
        switch scalarType {
        case c10.ScalarType.Byte: // UInt8 in PyTorch
            self = .int8(Int8(bitPattern: cxxScalar.toByte()))
        case c10.ScalarType.Char: // Int8 in PyTorch
            self = .int8(cxxScalar.toChar())
        case c10.ScalarType.Short:
            self = .int16(cxxScalar.toShort())
        case c10.ScalarType.Int:
            self = .int32(cxxScalar.toInt())
        case c10.ScalarType.Long:
            self = .int64(cxxScalar.toLong())
        case c10.ScalarType.Half: // Promote Float16 to Float
            self = .float(cxxScalar.toFloat())
        case c10.ScalarType.Float:
            self = .float(cxxScalar.toFloat())
        case c10.ScalarType.Double:
            self = .double(cxxScalar.toDouble())
        case c10.ScalarType.Bool:
            self = .bool(cxxScalar.toBool())
        default:
            // This C++ scalar holds a type we don't yet support (e.g., Complex)
            return nil
        }
    }
}


extension Scalar: ExpressibleByIntegerLiteral {
    /// Creates a scalar from an integer literal, defaulting to `Int64` storage.
    public init(integerLiteral value: Int64) {
        self = .int64(value)
    }
}

extension Scalar: ExpressibleByFloatLiteral {
    /// Creates a scalar from a floating-point literal, defaulting to `Double` storage.
    public init(floatLiteral value: Double) {
        self = .double(value)
    }
}

extension Scalar: ExpressibleByBooleanLiteral {
    /// Creates a scalar from a boolean literal.
    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}

// You can add other convenience initializers if needed
public extension Scalar {
    /// Shorthand for creating a scalar from a `Float`.
    init(_ value: Float) { self = .float(value) }
    /// Shorthand for creating a scalar from an `Int`, stored as `Int64`.
    init(_ value: Int) { self = .int64(Int64(value)) }
    /// Shorthand for creating a scalar from an `Int32`.
    init(_ value: Int32) { self = .int32(value) }
    /// Shorthand for creating a scalar from an `Int16`.
    init(_ value: Int16) { self = .int16(value) }
    /// Shorthand for creating a scalar from an `Int8`.
    init(_ value: Int8) { self = .int8(value) }
    /// Shorthand for creating a scalar from a `Double`.
    init(_ value: Double) { self = .double(value) }
    /// Shorthand for creating a scalar from a `Bool`.
    init(_ value: Bool) { self = .bool(value) }
}
