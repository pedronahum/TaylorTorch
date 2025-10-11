#if canImport(_Differentiation)
  import _Differentiation
  import ATenCXX

  // ✅ Step 1: Add AdditiveArithmetic conformance
  extension Tensor: AdditiveArithmetic {
    /// Additive identity represented as a scalar float tensor (matches most model parameters).
    public static var zero: Tensor { Tensor(0, dtype: .float32) }
  }

  // ✅ Step 2: Differentiable conformance
  extension Tensor: Differentiable {
    public typealias TangentVector = Tensor

    // The `move(by:)` function is required for the conformance.
    /// Updates the tensor by adding the provided tangent direction.
    public mutating func move(by direction: Tensor) {
      self = self + direction
    }
  }

#endif
