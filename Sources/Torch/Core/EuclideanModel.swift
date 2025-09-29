import _Differentiation

/// A model that has a Euclidean "parameter subspace".
///
/// Mirrors the spirit of S4TF's `EuclideanDifferentiable`: many real models
/// contain both trainable tensors and other non-trainable fields. The "vectorView"
/// projects the model into the Euclidean parameter space as a TangentVector,
/// which is what optimizers use for weight decay, dot products, norms, etc.
///
/// Default implementation builds the view from your `parameterKeyPaths`.
public protocol EuclideanModel: ParameterIterableModel {
  /// The Euclidean parameter view of `self` as a tangent vector.
  /// Defaults to copying the model's trainable tensors in `parameterKeyPaths`
  /// and zero elsewhere (if present in the tangent).
  var vectorView: TangentVector { get }

  /// (Optional) Return alternative Euclidean views, e.g. excluding biases.
  /// Default implementation uses `parameterKeyPaths` and a predicate to zero-out
  /// some leaves in the view.
  func vectorView(
    excluding predicate: (WritableKeyPath<Self, Tensor>) -> Bool
  ) -> TangentVector
}

// MARK: - Default implementations

extension EuclideanModel {
  @inlinable
  public var vectorView: TangentVector {
    // Use the generic helper you already have.
    self.asTangentVector()
  }

  /// Build a masked vector view by zeroing any leaves for which `predicate` is true.
  @inlinable
  public func vectorView(
    excluding predicate: (WritableKeyPath<Self, Tensor>) -> Bool
  ) -> TangentVector {
    var tv = TangentVector.zero
    // Pre-create zero tensors with correct dtype/device per parameter as needed.
    for (mKP, tKP) in Self.zippedParameterPaths() {
      let p = self[keyPath: mKP]
      tv[keyPath: tKP] =
        predicate(mKP)
        ? Tensor.zeros(shape: p.shape, dtype: p.dtype!, device: p.device)
        : p
    }
    return tv
  }
}

// MARK: - Euclidean algebra on TangentVectors

extension EuclideanModel {
  /// ⟨a, b⟩ = Σ_i sum(a_i * b_i)
  @inlinable
  public static func euclideanDot(_ a: TangentVector, _ b: TangentVector) -> Tensor {
    var acc = Tensor(0.0)
    for kp in TangentVector.parameterKeyPaths {
      let prod = a[keyPath: kp].multiplying(b[keyPath: kp])
      acc = acc.adding(prod.sum())
    }
    return acc
  }

  /// ||g||^2 = Σ_i sum(g_i^2)
  @inlinable
  public static func euclideanSquaredNorm(_ g: TangentVector) -> Tensor {
    var acc = Tensor(0.0)
    for kp in TangentVector.parameterKeyPaths {
      let t = g[keyPath: kp]
      acc = acc.adding(t.multiplying(t).sum())
    }
    return acc
  }

  /// ||g|| = sqrt(Σ_i sum(g_i^2))
  @inlinable
  public static func euclideanNorm(_ g: TangentVector) -> Tensor {
    euclideanSquaredNorm(g).sqrt()
  }
}
