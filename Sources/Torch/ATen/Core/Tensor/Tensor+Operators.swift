// Sources/ATen/Tensor+Operators.swift

// ✅ Global operators MUST be at the top level of the file, not in an extension.
// MARK: Tensor ⊗ Tensor
/// Element-wise sum of two tensors.
@inlinable public func + (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.adding(rhs) }
/// Element-wise difference between two tensors.
@inlinable public func - (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.subtracting(rhs) }
/// Element-wise product of two tensors.
@inlinable public func * (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.multiplying(rhs) }
/// Element-wise quotient of two tensors.
@inlinable public func / (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.dividing(rhs) }

// MARK: Tensor ⊗ Scalar
/// Element-wise sum of a tensor and a broadcast scalar.
@inlinable public func + <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor { lhs.adding(rhs) }
/// Element-wise difference between a tensor and a broadcast scalar.
@inlinable public func - <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor {
  lhs.subtracting(rhs)
}
/// Element-wise product of a tensor and a broadcast scalar.
@inlinable public func * <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor {
  lhs.multiplying(rhs)
}
/// Element-wise quotient of a tensor and a broadcast scalar.
@inlinable public func / <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor { lhs.dividing(rhs) }

// MARK: Scalar ⊗ Tensor (flip)
/// Element-wise sum with the scalar operand appearing on the left.
@inlinable public func + <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor { rhs.adding(lhs) }
/// Element-wise difference with the scalar operand appearing on the left.
@inlinable public func - <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor {
  rhs.negated().adding(lhs)
}
/// Element-wise product with the scalar operand appearing on the left.
@inlinable public func * <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor {
  rhs.multiplying(lhs)
}
/// Element-wise quotient with the scalar operand appearing on the left.
@inlinable public func / <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor { Tensor(lhs) / rhs }

// ✅ The extension should only contain members of the Tensor type.
// MARK: Compound assignment
extension Tensor {
  /// In-place element-wise addition.
  public static func += (lhs: inout Tensor, rhs: Tensor) { lhs = lhs + rhs }
  /// In-place element-wise subtraction.
  public static func -= (lhs: inout Tensor, rhs: Tensor) { lhs = lhs - rhs }
  /// In-place element-wise multiplication.
  public static func *= (lhs: inout Tensor, rhs: Tensor) { lhs = lhs * rhs }
  /// In-place element-wise division.
  public static func /= (lhs: inout Tensor, rhs: Tensor) { lhs = lhs / rhs }

  /// In-place element-wise addition with a broadcast scalar.
  public static func += <T: TorchArithmetic>(lhs: inout Tensor, rhs: T) { lhs = lhs + rhs }
  /// In-place element-wise subtraction with a broadcast scalar.
  public static func -= <T: TorchArithmetic>(lhs: inout Tensor, rhs: T) { lhs = lhs - rhs }
  /// In-place element-wise multiplication with a broadcast scalar.
  public static func *= <T: TorchArithmetic>(lhs: inout Tensor, rhs: T) { lhs = lhs * rhs }
  /// In-place element-wise division with a broadcast scalar.
  public static func /= <T: TorchArithmetic>(lhs: inout Tensor, rhs: T) { lhs = lhs / rhs }
}
