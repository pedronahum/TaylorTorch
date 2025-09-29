import Foundation

/// Codable representation of a tensor's metadata and raw storage bytes.
public struct CodableTensor: Codable {
  /// Logical shape of the tensor in row-major order.
  public var shape: [Int]
  /// Element type for the encoded tensor payload.
  public var dtype: DType
  /// Device on which the tensor was originally materialized.
  public var device: Device
  /// Raw bytes containing the tensor elements in row-major order.
  public var bytes: Data
}

public extension CodableTensor {
  /// Serializes a tensor by capturing its metadata and copying its elements into `Data`.
  init(_ t: Tensor) {
    self.shape = t.shape
    self.dtype = t.dtype ?? .float32
    self.device = t.device

    switch self.dtype {
    // ✅ Use the safe, scoped 'withUnsafeBufferPointer' pattern
    case .float32:
      let array = t.toArray(as: Float.self)
      self.bytes = array.withUnsafeBufferPointer { Data(buffer: $0) }
    case .float64:
      let array = t.toArray(as: Double.self)
      self.bytes = array.withUnsafeBufferPointer { Data(buffer: $0) }
    case .int64:
      let array = t.toArray(as: Int64.self)
      self.bytes = array.withUnsafeBufferPointer { Data(buffer: $0) }
    case .int32:
      let array = t.toArray(as: Int32.self)
      self.bytes = array.withUnsafeBufferPointer { Data(buffer: $0) }
    
    // For simple byte arrays, direct initialization is safe
    case .uint8:
      self.bytes = Data(t.toArray(as: UInt8.self))
    case .bool:
      self.bytes = Data(t.toArray(as: Bool.self).map { $0 ? 1 : 0 })
      
    default:
      fatalError("CodableTensor: dtype \(self.dtype) not yet supported")
    }
  }

  /// Reconstructs a `Tensor` from the encoded representation.
  func makeTensor() -> Tensor {
    switch dtype {
    case .float32: return bytes.withUnsafeBytes { Tensor(array: Array($0.bindMemory(to: Float.self)), shape: shape, device: device) }
    case .float64: return bytes.withUnsafeBytes { Tensor(array: Array($0.bindMemory(to: Double.self)), shape: shape, device: device) }
    case .int64:   return bytes.withUnsafeBytes { Tensor(array: Array($0.bindMemory(to: Int64.self)), shape: shape, device: device) }
    case .int32:   return bytes.withUnsafeBytes { Tensor(array: Array($0.bindMemory(to: Int32.self)), shape: shape, device: device) }
    case .uint8:   return bytes.withUnsafeBytes { Tensor(array: Array($0.bindMemory(to: UInt8.self)), shape: shape, device: device) }
    case .bool:    return Tensor(array: bytes.map { $0 != 0 }, shape: shape, device: device)
    default:       fatalError("CodableTensor: dtype \(dtype) not yet supported")
    }
  }
}
