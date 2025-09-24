import ATenCXX

public extension DType {
  /// Returns the `c10::ScalarType` that corresponds to this Swift dtype.
  var _c10: c10.ScalarType {
    switch self {
    case .bool:     return c10.ScalarType.Bool
    case .int8:     return c10.ScalarType.Char
    case .uint8:    return c10.ScalarType.Byte
    case .int16:    return c10.ScalarType.Short
    case .int32:    return c10.ScalarType.Int
    case .int64:    return c10.ScalarType.Long
    case .float16:  return c10.ScalarType.Half
    case .bfloat16: return c10.ScalarType.BFloat16
    case .float32:  return c10.ScalarType.Float
    case .float64:  return c10.ScalarType.Double
    // TODO: complex & quantized (map as needed)
    default:
      fatalError("DType \(self) not yet bridged to c10.ScalarType")
    }
  }

  /// Initializes a dtype from a Torch `c10::ScalarType` when a mapping exists.
  init?(_ s: c10.ScalarType) {
    switch s {
    case c10.ScalarType.Bool:     self = .bool
    case c10.ScalarType.Char:     self = .int8
    case c10.ScalarType.Byte:     self = .uint8
    case c10.ScalarType.Short:    self = .int16
    case c10.ScalarType.Int:      self = .int32
    case c10.ScalarType.Long:     self = .int64
    case c10.ScalarType.Half:     self = .float16
    case c10.ScalarType.BFloat16: self = .bfloat16
    case c10.ScalarType.Float:    self = .float32
    case c10.ScalarType.Double:   self = .float64
    default: return nil
    }
  }
}
