// Sources/ATen/Tensor+FactoriesVariadic.swift
public extension Tensor {
  /// Variadic convenience wrapper around `Tensor.empty(shape:dtype:device:)`.
  static func empty(_ shape: Int..., dtype: DType, device: Device = .cpu) -> Tensor {
    empty(shape: shape, dtype: dtype, device: device)    // uses your existing factory
  }
  /// Variadic convenience wrapper that creates a zero-filled tensor.
  static func zeros(_ shape: Int..., dtype: DType = .float32, device: Device = .cpu) -> Tensor {
    zeros(shape: shape, dtype: dtype, device: device)
  }
  /// Variadic convenience wrapper that creates a one-filled tensor.
  static func ones(_ shape: Int..., dtype: DType = .float32, device: Device = .cpu) -> Tensor {
    ones(shape: shape, dtype: dtype, device: device)
  }
  /// Variadic convenience wrapper that creates a tensor filled with `value`.
  static func full<T: TorchArithmetic>(_ value: T, _ shape: Int..., device: Device = .cpu) -> Tensor {
    full(value, shape: shape, device: device)
  }
}
