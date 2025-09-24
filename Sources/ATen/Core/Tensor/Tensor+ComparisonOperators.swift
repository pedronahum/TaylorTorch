
@preconcurrency import ATenCXX

/// Element-wise comparison operators that mirror PyTorch's tensor overloads but avoid
/// clashing with Swift's built-in numeric behavior.
infix operator .== : ComparisonPrecedence
infix operator .<  : ComparisonPrecedence
infix operator .<= : ComparisonPrecedence
infix operator .>  : ComparisonPrecedence
infix operator .>= : ComparisonPrecedence

// Tensor ⊗ Tensor
/// Element-wise equality comparison between two tensors.
@inlinable public func .== (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.eq(rhs) }
/// Element-wise strictly-less-than comparison between two tensors.
@inlinable public func .<  (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.lt(rhs) }
/// Element-wise less-than-or-equal comparison between two tensors.
@inlinable public func .<= (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.le(rhs) }
/// Element-wise strictly-greater-than comparison between two tensors.
@inlinable public func .>  (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.gt(rhs) }
/// Element-wise greater-than-or-equal comparison between two tensors.
@inlinable public func .>= (lhs: Tensor, rhs: Tensor) -> Tensor { lhs.ge(rhs) }

// Tensor ⊗ Scalar
/// Element-wise equality comparison between a tensor and scalar (broadcast across the tensor).
@inlinable public func .== <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor { lhs.eq(rhs) }
/// Element-wise strictly-less-than comparison with a scalar broadcast across the tensor.
@inlinable public func .<  <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor { lhs.lt(rhs) }
/// Element-wise less-than-or-equal comparison with a scalar broadcast across the tensor.
@inlinable public func .<= <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor { lhs.le(rhs) }
/// Element-wise strictly-greater-than comparison with a scalar broadcast across the tensor.
@inlinable public func .>  <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor { lhs.gt(rhs) }
/// Element-wise greater-than-or-equal comparison with a scalar broadcast across the tensor.
@inlinable public func .>= <T: TorchArithmetic>(lhs: Tensor, rhs: T) -> Tensor { lhs.ge(rhs) }

// Scalar ⊗ Tensor (flip)
/// Element-wise equality comparison where the scalar operand appears on the left.
@inlinable public func .== <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor { rhs.eq(lhs) }
/// Element-wise strictly-less-than comparison where the scalar operand appears on the left.
@inlinable public func .<  <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor { rhs.gt(lhs) }
/// Element-wise less-than-or-equal comparison where the scalar operand appears on the left.
@inlinable public func .<= <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor { rhs.ge(lhs) }
/// Element-wise strictly-greater-than comparison where the scalar operand appears on the left.
@inlinable public func .>  <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor { rhs.lt(lhs) }
/// Element-wise greater-than-or-equal comparison where the scalar operand appears on the left.
@inlinable public func .>= <T: TorchArithmetic>(lhs: T, rhs: Tensor) -> Tensor { rhs.le(lhs) }
