// Sources/ATen/Tensor+Builders.swift

/// Result builder that collects scalar elements into a 1-D tensor literal.
@resultBuilder public struct Tensor1DBuilder<T: TorchTensorScalar> {
  public static func buildBlock(_ elements: T...) -> [T] { elements }
}

/// Result builder that gathers row literals into a 2-D tensor literal.
@resultBuilder public struct Tensor2DBuilder<T: TorchTensorScalar> {
  public static func buildBlock(_ rows: [T]...) -> [[T]] { rows }
}

/// Convenient API for materializing a 1-D tensor using Swift's builder syntax.
@inlinable public func tensor1D<T: TorchTensorScalar>(
  device: Device = .cpu,
  @Tensor1DBuilder<T> _ content: () -> [T]
) -> Tensor {
  let a = content()
  return Tensor(array: a, shape: [a.count], device: device)
}

/// Convenient API for materializing a 2-D tensor using Swift's builder syntax.
@inlinable public func tensor2D<T: TorchTensorScalar>(
  device: Device = .cpu,
  @Tensor2DBuilder<T> _ content: () -> [[T]]
) -> Tensor {
  let rows = content()
  precondition(!rows.isEmpty, "empty literal")
  let cols = rows[0].count
  precondition(rows.allSatisfy { $0.count == cols }, "ragged rows in tensor literal")
  let flat = rows.flatMap { $0 }
  return Tensor(array: flat, shape: [rows.count, cols], device: device)
}
