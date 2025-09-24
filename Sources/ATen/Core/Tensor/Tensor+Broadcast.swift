@preconcurrency import ATenCXX

public extension Tensor {
  /// Expand singleton dimensions to match the supplied shape. Returns a view when possible.
  /// - Parameters:
  ///   - shape: Target shape compatible with current broadcast semantics.
  ///   - implicit: Pass `true` for PyTorch-style implicit broadcasting checks.
  @inlinable
  func expanded(to shape: [Int], implicit: Bool = false) -> Tensor {
    var s64 = shape.map(Int64.init)
    return Tensor(_impl.expand(&s64, s64.count, implicit))
  }

  /// Expand singleton dimensions to match the layout of another tensor. Returns a view when possible.
  /// - Parameter other: Tensor whose shape should be adopted.
  @inlinable
  func expanded(as other: Tensor) -> Tensor {
    Tensor(_impl.expandAs(other._impl))
  }

  /// Broadcast the tensor to an explicit shape, materializing a view when feasible.
  /// - Parameter shape: Target broadcast shape.
  @inlinable
  func broadcasted(to shape: [Int]) -> Tensor {
    var s64 = shape.map(Int64.init)
    return Tensor(_impl.broadcastTo(&s64, s64.count))
  }
}
