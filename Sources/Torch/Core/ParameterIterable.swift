
import _Differentiation

/// List the writable key paths to all *trainable* Tensor leaves of a model.
/// You provide this statically; no reflection or private SPI involved.
/// Later we can replace this with a macro that synthesizes the list.
public protocol ParameterIterable {
  static var parameterKeyPaths: [WritableKeyPath<Self, Tensor>] { get }
}

/// Handy sugar to compose parent.child key paths in a typesafe way.
extension WritableKeyPath {
  @inlinable
  public func appending(_ child: WritableKeyPath<Value, Tensor>) -> WritableKeyPath<Root, Tensor> {
    self.appending(path: child)
  }
}

/// A model whose stored `Tensor` parameters are enumerable via key paths,
/// and whose `TangentVector` mirrors that structure (also enumerable).
public protocol ParameterIterableModel:
  Differentiable & ParameterIterable
where TangentVector: ParameterIterable {}

extension ParameterIterable {
  /// In‑place iteration over all parameters.
  @inlinable
  public mutating func forEachParameter(_ body: (inout Tensor) -> Void) {
    for kp in Self.parameterKeyPaths { body(&self[keyPath: kp]) }
  }

  /// Read‑only map; returns a new instance with transformed parameters.
  @inlinable
  public func mapParameters(_ body: (Tensor) -> Tensor) -> Self {
    var out = self
    for kp in Self.parameterKeyPaths { out[keyPath: kp] = body(out[keyPath: kp]) }
    return out
  }

  /// Collects parameters (useful for checkpointing or averaging).
  @inlinable
  public func flattenedParameters() -> [Tensor] {
    Self.parameterKeyPaths.map { self[keyPath: $0] }
  }

  /// Assigns parameters back in a deterministic order.
  @inlinable
  public mutating func assignFlattenedParameters(_ values: [Tensor]) {
    precondition(values.count == Self.parameterKeyPaths.count, "Parameter count mismatch")
    for (i, kp) in Self.parameterKeyPaths.enumerated() { self[keyPath: kp] = values[i] }
  }
}

extension ParameterIterableModel {
  /// Zips model key paths with its tangent key paths (1:1 order).
  @inlinable
  public static func zippedParameterPaths()
    -> [(WritableKeyPath<Self, Tensor>, WritableKeyPath<TangentVector, Tensor>)]
  {
    precondition(
      Self.parameterKeyPaths.count == TangentVector.parameterKeyPaths.count,
      "Model and TangentVector must list parameters in the same order"
    )
    return Array(zip(Self.parameterKeyPaths, TangentVector.parameterKeyPaths))
  }

  /// Build a tangent vector by copying the model's *current* parameter values.
  /// Useful for L2 regularization, EMA, or AdamW's decoupled weight decay.
  @inlinable
  public func asTangentVector() -> TangentVector {
    var tv = TangentVector.zero  // structure
    for (mKP, tKP) in Self.zippedParameterPaths() {
      tv[keyPath: tKP] = self[keyPath: mKP]
    }
    return tv
  }

  /// Create a tangent filled with per‑leaf zeros that match shapes/dtypes/devices of model params.
  @inlinable
  public func zerosLikeParameters() -> TangentVector {
    var tv = TangentVector.zero
    for (mKP, tKP) in Self.zippedParameterPaths() {
      let p = self[keyPath: mKP]
      tv[keyPath: tKP] = Tensor.zeros(shape: p.shape, dtype: p.dtype!, device: p.device)
    }
    return tv
  }
}
