@preconcurrency import ATenCXX
import _Differentiation

// MARK: - Unary
extension Tensor {
  /// Returns the element-wise additive inverse of the tensor.
  @inlinable
  public func negated() -> Tensor { Tensor(_impl.neg()) }

  /// Returns the element-wise absolute value of the tensor.
  @inlinable
  public func abs() -> Tensor { Tensor(_impl.abs_()) }

  /// Applies the ReLU activation (`max(0, x)`) element-wise.
  @inlinable
  public func relu() -> Tensor { Tensor(_impl.relu()) }

  /// Returns the element-wise exponential of the tensor.
  @inlinable
  public func exp() -> Tensor { Tensor(_impl.exp_()) }

  /// Returns the element-wise natural logarithm of the tensor.
  @inlinable
  public func log() -> Tensor { Tensor(_impl.log_()) }

  /// Returns the element-wise square root of the tensor.
  @inlinable
  public func sqrt() -> Tensor { Tensor(_impl.sqrt_()) }

  /// Returns the element-wise hyperbolic tangent of the tensor.
  @inlinable
  public func tanh() -> Tensor { Tensor(_impl.tanh_()) }

  /// Returns the element-wise sigmoid of the tensor.
  @inlinable
  public func sigmoid() -> Tensor { Tensor(_impl.sigmoid_()) }

  /// Returns the element-wise sine of the tensor.
  @inlinable
  public func sin() -> Tensor { Tensor(_impl.sin_()) }

  /// Returns the element-wise cosine of the tensor.
  @inlinable
  public func cos() -> Tensor { Tensor(_impl.cos_()) }

  /// Returns the element-wise tangent of the tensor.
  @inlinable
  public func tan() -> Tensor { Tensor(_impl.tan_()) }

  /// Returns the element-wise arcsine of the tensor.
  @inlinable
  public func asin() -> Tensor { Tensor(_impl.asin_()) }

  /// Returns the element-wise arccosine of the tensor.
  @inlinable
  public func acos() -> Tensor { Tensor(_impl.acos_()) }

  /// Returns the element-wise arctangent of the tensor.
  @inlinable
  public func atan() -> Tensor { Tensor(_impl.atan_()) }

  /// Returns the element-wise hyperbolic sine of the tensor.
  @inlinable
  public func sinh() -> Tensor { Tensor(_impl.sinh_()) }

  /// Returns the element-wise hyperbolic cosine of the tensor.
  @inlinable
  public func cosh() -> Tensor { Tensor(_impl.cosh_()) }

  /// Returns the element-wise inverse hyperbolic sine of the tensor.
  @inlinable
  public func asinh() -> Tensor { Tensor(_impl.asinh_()) }

  /// Returns the element-wise inverse hyperbolic cosine of the tensor.
  @inlinable
  public func acosh() -> Tensor { Tensor(_impl.acosh_()) }

  /// Returns the element-wise inverse hyperbolic tangent of the tensor.
  @inlinable
  public func atanh() -> Tensor { Tensor(_impl.atanh_()) }

  /// Returns the element-wise Gauss error function of the tensor.
  @inlinable
  public func erf() -> Tensor { Tensor(_impl.erf_()) }

  /// Returns the element-wise complementary error function of the tensor.
  @inlinable
  public func erfc() -> Tensor { Tensor(_impl.erfc_()) }

}

// MARK: - Binary (tensor ⊗ tensor)
extension Tensor {
  /// Returns the element-wise difference `self - alpha * other`.
  @inlinable public func subtracting(_ other: Tensor, alpha: Scalar = .int64(1)) -> Tensor {
    Tensor(_impl.sub(other._impl, alpha._cxxScalar))
  }
  /// Returns the element-wise product of the two tensors.
  @inlinable public func multiplying(_ other: Tensor) -> Tensor { Tensor(_impl.mul(other._impl)) }
  /// Returns the element-wise quotient `self / other`.
  @inlinable public func dividing(_ other: Tensor) -> Tensor { Tensor(_impl.div(other._impl)) }
}

// MARK: - Binary (tensor ⊗ scalar)
extension Tensor {
  /// Returns the element-wise difference `self - scalar`.
  @inlinable public func subtracting<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.subScalar(scalar._cxxScalar))
  }
  /// Returns the element-wise product `self * scalar`.
  @inlinable public func multiplying<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.mulScalar(scalar._cxxScalar))
  }
  /// Returns the element-wise quotient `self / scalar`.
  @inlinable public func dividing<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.divScalar(scalar._cxxScalar))
  }
}

// MARK: - Power
extension Tensor {
  /// Raises each element of the tensor to a scalar power.
  @inlinable public func pow<T: TorchArithmetic>(_ power: T) -> Tensor {
    Tensor(_impl.powScalar(power._cxxScalar))
  }
  /// Raises each element of the tensor to the power given by the corresponding element of `other`.
  @inlinable public func pow(_ other: Tensor) -> Tensor {
    Tensor(_impl.powTensor(other._impl))
  }
}

// MARK: - Clamp
extension Tensor {
  /// Clamps each element to the inclusive range `[min, max]`.
  @inlinable public func clamp<T: TorchArithmetic>(min: T, max: T) -> Tensor {
    Tensor(_impl.clamp(min._cxxScalar, max._cxxScalar))
  }
}

// MARK: - Reductions
extension Tensor {
  /// Returns the sum of all tensor elements.
  @inlinable public func sum() -> Tensor { Tensor(_impl.sumAll()) }
  /// Returns the mean of all tensor elements.
  @inlinable public func mean() -> Tensor { Tensor(_impl.meanAll()) }

  /// Returns the sum along `dim`, optionally keeping reduced dimensions.
  @inlinable public func sum(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.sumDim(Int64(dim), keepdim))
  }
  /// Returns the mean along `dim`, optionally keeping reduced dimensions.
  @inlinable public func mean(dim: Int, keepdim: Bool = false) -> Tensor {
    Tensor(_impl.meanDim(Int64(dim), keepdim))
  }
}

// MARK: - Linalg
extension Tensor {
  /// Performs matrix multiplication following PyTorch's broadcasting semantics.
  @inlinable public func matmul(_ other: Tensor) -> Tensor { Tensor(_impl.matmul(other._impl)) }

  /// 1-D dot product (returns a rank-0 tensor).
  @inlinable public func dot(_ other: Tensor) -> Tensor { Tensor(_impl.dot(other._impl)) }
}

// MARK: - Comparisons (tensor ⊗ tensor)
extension Tensor {
  /// Element-wise equality comparison with another tensor.
  @inlinable public func eq(_ other: Tensor) -> Tensor { Tensor(_impl.eq(other._impl)) }
  /// Element-wise strictly-less-than comparison with another tensor.
  @inlinable public func lt(_ other: Tensor) -> Tensor { Tensor(_impl.lt(other._impl)) }
  /// Element-wise less-than-or-equal comparison with another tensor.
  @inlinable public func le(_ other: Tensor) -> Tensor { Tensor(_impl.le(other._impl)) }
  /// Element-wise strictly-greater-than comparison with another tensor.
  @inlinable public func gt(_ other: Tensor) -> Tensor { Tensor(_impl.gt(other._impl)) }
  /// Element-wise greater-than-or-equal comparison with another tensor.
  @inlinable public func ge(_ other: Tensor) -> Tensor { Tensor(_impl.ge(other._impl)) }
}

// MARK: - Comparisons (tensor ⊗ scalar)
extension Tensor {
  /// Element-wise equality comparison with a scalar broadcast across the tensor.
  @inlinable public func eq<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.eqScalar(scalar._cxxScalar))
  }
  /// Element-wise strictly-less-than comparison with a scalar broadcast across the tensor.
  @inlinable public func lt<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.ltScalar(scalar._cxxScalar))
  }
  /// Element-wise less-than-or-equal comparison with a scalar broadcast across the tensor.
  @inlinable public func le<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.leScalar(scalar._cxxScalar))
  }
  /// Element-wise strictly-greater-than comparison with a scalar broadcast across the tensor.
  @inlinable public func gt<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.gtScalar(scalar._cxxScalar))
  }
  /// Element-wise greater-than-or-equal comparison with a scalar broadcast across the tensor.
  @inlinable public func ge<T: TorchArithmetic>(_ scalar: T) -> Tensor {
    Tensor(_impl.geScalar(scalar._cxxScalar))
  }
}

// MARK: - Where (ternary)
public enum TorchWhere {
  /// Return a tensor whose elements select between `a` and `b` based on `condition` (broadcasting applies).
  /// - Parameters:
  ///   - condition: Boolean tensor that chooses between `a` and `b`.
  ///   - a: Tensor providing values where the condition is `true`.
  ///   - b: Tensor providing values where the condition is `false`.
  @inlinable
  public static func select(condition: Tensor, _ a: Tensor, _ b: Tensor) -> Tensor {
    Tensor(TTSTensor.where3(condition._impl, a._impl, b._impl))
  }
}
