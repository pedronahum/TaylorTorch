// Sources/ATen/Core/Tensor/Tensor+ShapeSugar.swift
extension Tensor {
  /// Convenience: reshape to the shape of another tensor.
  @inlinable
  public func reshaped(as other: Tensor) -> Tensor { reshaped(other.shape) }
}
