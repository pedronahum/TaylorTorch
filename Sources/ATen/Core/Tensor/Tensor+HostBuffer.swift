// Sources/ATen/Tensor+HostBuffer.swift
public extension Tensor {
  /// Exposes a temporary host buffer view over tensor data. When the tensor is a
  /// contiguous CPU tensor with a matching dtype, this attempts to borrow memory;
  /// otherwise it materializes a copy before invoking `body`.
  func withHostBuffer<T: TorchTensorScalar, R>(
    as: T.Type, _ body: (UnsafeBufferPointer<T>) throws -> R
  ) rethrows -> R {
    if device == .cpu, dtype == T.torchDType, isContiguous {
      // ⚠️ If you add a shim for data_ptr<T>(), use it here.
      let array: [T] = toArray(as: T.self)  // fallback copy until data_ptr<T> shim exists
      return try array.withUnsafeBufferPointer(body)
    } else {
      let array: [T] = toArray(as: T.self)  // staged copy (CPU+cast)
      return try array.withUnsafeBufferPointer(body)
    }
  }

  /// Provides mutable access to the tensor elements via a Swift array copy and
  /// returns both the closure result and a rebuilt tensor using the mutated buffer.
  func withMutableHostBuffer<T: TorchTensorScalar, R>(
    as: T.Type, _ body: (inout [T]) throws -> R
  ) rethrows -> (result: R, tensor: Tensor) {
    var array: [T] = toArray(as: T.self)
    let r = try body(&array)
    let newT = Tensor(array: array, shape: shape, device: device)
    return (r, newT)
  }
}
