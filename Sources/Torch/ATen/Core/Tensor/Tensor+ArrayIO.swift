@preconcurrency import ATenCXX
import Foundation

// Utilities
/// Computes the product of the integers in `xs` using &* to preserve overflow semantics.
@inline(__always) private func _product(_ xs: [Int]) -> Int {
  var p = 1
  for x in xs { p &*= x }
  return p
}

public extension Tensor {
  /// Create a tensor by copying from a Swift array.
  /// - Parameters:
  ///   - array: Flat, row-major values.
  ///   - shape: Logical shape for the tensor (product must equal `array.count`).
  ///   - device: Destination device (`.cpu`, `.cuda(_:)`, `.mps`).
  init<T: TorchTensorScalar>(array: [T], shape: [Int], device: Device = .cpu) {
    precondition(_product(shape) == array.count, "shape product must equal element count")
    var sizes64 = shape.map { Int64($0) }

    // Map Swift DType -> c10.ScalarType
    let dt = T.torchDType._c10  // uses your DType bridge :contentReference[oaicite:1]{index=1}

    // Special-case Bool: normalize to 0/1 bytes to avoid relying on ABI of Swift.Bool
    if T.self == Bool.self {
      let bytes = (array as! [Bool]).map { $0 ? UInt8(1) : UInt8(0) }
      self._impl = bytes.withUnsafeBytes { raw in
        TTSTensor.fromHostBuffer(
          raw.baseAddress!, bytes.count,
          &sizes64, sizes64.count,
          dt, device._c10 // device bridge :contentReference[oaicite:2]{index=2}
        )
      }
      return
    }

    // Generic path for numeric types with stable layout (Int8/16/32/64, UInt8, Float, Double)
    self._impl = array.withUnsafeBytes { raw in
      TTSTensor.fromHostBuffer(
        raw.baseAddress!, array.count,
        &sizes64, sizes64.count,
        dt, device._c10
      )
    }
  }

  /// Create a tensor from a Swift array while explicitly choosing the dtype.
  init<T: TorchTensorScalar>(
    array: [T],
    shape: [Int],
    dtype: DType,
    device: Device = .cpu
  ) {
    self.init(array: array, shape: shape, device: device)
    if let current = self.dtype, current != dtype {
      self = self.to(dtype: dtype)
    }
  }

  /// Copy tensor contents into a Swift array of the requested element type.
  /// If the tensor's dtype or device differ, data are converted and moved on the C++ side.
  /// - Parameter type: Element type for the resulting array (defaults to `T`).
  func toArray<T: TorchTensorScalar>(as type: T.Type = T.self) -> [T] {
    let n = Int(_impl.numel())
    let dt = T.torchDType._c10  // uses your DType bridge :contentReference[oaicite:3]{index=3}

    // Bool: retrieve as 0/1 bytes then map to Bool
    if T.self == Bool.self {
      var tmp = Array<UInt8>(repeating: 0, count: n)
      tmp.withUnsafeMutableBytes { raw in
        _ = _impl.toHostBuffer(raw.baseAddress!, n, dt)
      }
      let out = tmp.map { $0 != 0 }
      return out as! [T]
    }

    // Numeric types: write directly
    return Array<T>(unsafeUninitializedCapacity: n) { buf, initializedCount in
      let ok = _impl.toHostBuffer(UnsafeMutableRawPointer(buf.baseAddress!), n, dt)
      precondition(ok, "toArray failed: type not supported or buffer too small")
      initializedCount = n
    }
  }
}
