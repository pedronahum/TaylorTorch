@preconcurrency import ATenCXX

extension Tensor: Equatable {
  /// Returns `true` when two tensors have identical shapes and equal contents.
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    lhs._impl.equal(rhs._impl)
  }
}

extension Tensor {
  /// Numeric closeness test matching the semantics of `torch.allclose`.
  /// - Parameters:
  ///   - other: Tensor to compare against.
  ///   - rtol: Relative tolerance.
  ///   - atol: Absolute tolerance.
  ///   - equalNan: When `true`, `NaN` values are treated as equal.
  public func isClose(
    to other: Tensor, rtol: Double = 1e-5, atol: Double = 1e-8, equalNan: Bool = false
  ) -> Bool {
    _impl.allclose(other._impl, rtol, atol, equalNan)
  }
}

// Value equality (all elements equal after promotion/broadcast).
extension Tensor {
  /// Returns `true` when the tensors are broadcast-compatible and every element matches.
  public func elementsEqual(_ other: Tensor) -> Bool {
    // elementwise eq → Bool tensor → reduce 'all'
    let allEqual = self.eq(other).all()
    let vals: [Bool] = allEqual.toArray()
    return vals.count == 1 && vals[0]
  }

  /// Alias for `elementsEqual(_:)`.
  /// Returns `true` when the tensors are broadcast-compatible and every element matches.
  public func equal(_ other: Tensor) -> Bool {
    self.elementsEqual(other)
  }
}
