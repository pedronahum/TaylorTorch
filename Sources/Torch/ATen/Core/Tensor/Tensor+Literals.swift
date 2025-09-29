// Sources/ATen/Tensor+Literals.swift

/// Convenience literal helper that creates a tensor from `values` and an explicit `shape` array.
@inlinable public func tensor<T: TorchTensorScalar>(_ values: [T], shape: [Int], device: Device = .cpu) -> Tensor {
  Tensor(array: values, shape: shape, device: device)
}
/// Variadic overload for writing tensor literals without allocating an intermediate shape array.
@inlinable public func tensor<T: TorchTensorScalar>(_ values: [T], _ shape: Int..., device: Device = .cpu) -> Tensor {
  Tensor(array: values, shape: shape, device: device)
}
