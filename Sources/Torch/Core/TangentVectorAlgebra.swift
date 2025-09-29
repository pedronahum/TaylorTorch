
import _Differentiation

extension ParameterIterableModel {
  // MARK: - Elementwise maps over the tangent vector

  /// out[i] = f(a[i])
  @inlinable
  public static func map(_ a: TangentVector, _ f: (Tensor) -> Tensor) -> TangentVector {
    var out = TangentVector.zero
    for kp in TangentVector.parameterKeyPaths { out[keyPath: kp] = f(a[keyPath: kp]) }
    return out
  }

  /// out[i] = f(a[i], b[i])
  @inlinable
  public static func zipMap(
    _ a: TangentVector, _ b: TangentVector,
    _ f: (Tensor, Tensor) -> Tensor
  ) -> TangentVector {
    var out = TangentVector.zero
    let paths = TangentVector.parameterKeyPaths
    for i in paths.indices {
      let kp = paths[i]
      out[keyPath: kp] = f(a[keyPath: kp], b[keyPath: kp])
    }
    return out
  }

  // MARK: - Vector-like ops used by optimizers

  @inlinable public static func add(_ a: TangentVector, _ b: TangentVector) -> TangentVector {
    zipMap(a, b) { $0.adding($1) }  // tensor + tensor
  }

  /// a - b
  @inlinable public static func sub(_ a: TangentVector, _ b: TangentVector) -> TangentVector {
    zipMap(a, b) { $0.adding($1.multiplying(-1.0)) }
  }

  /// s * a (scalar scaling).  Uses your scalar lifting semantics.  ✅ Tested all over your suite. :contentReference[oaicite:5]{index=5}
  @inlinable public static func scale(_ a: TangentVector, by s: Double) -> TangentVector {
    map(a) { $0.multiplying(s) }
  }

  /// Elementwise product: a .* b
  @inlinable public static func hadamard(_ a: TangentVector, _ b: TangentVector) -> TangentVector {
    zipMap(a, b) { $0.multiplying($1) }
  }

  /// Elementwise division: a ./ b
  @inlinable public static func ewiseDiv(_ a: TangentVector, _ b: TangentVector) -> TangentVector {
    zipMap(a, b) { $0.dividing($1) }
  }

  /// Elementwise sqrt
  @inlinable public static func sqrt(_ a: TangentVector) -> TangentVector {
    map(a) { $0.sqrt() }  // sqrt is in your math + VJP tests. :contentReference[oaicite:6]{index=6}
  }

  /// Add scalar epsilon to each leaf (for AdamW denom).
  @inlinable public static func addEpsilon(_ a: TangentVector, _ eps: Double) -> TangentVector {
    map(a) { $0.adding(eps) }  // tensor + scalar; tested in multiple files. :contentReference[oaicite:7]{index=7}
  }

  /// L2 global norm of a tangent: sqrt(Σ_i ||gi||^2).
  @inlinable public static func globalNorm(of g: TangentVector) -> Tensor {
    var sqSum = Tensor(0.0)  // scalar; broadcasts correctly in your ops. :contentReference[oaicite:8]{index=8}
    for kp in TangentVector.parameterKeyPaths {
      let t = g[keyPath: kp]
      sqSum = sqSum.adding(t.multiplying(t).sum())  // reductions are part of your core & tested. :contentReference[oaicite:9]{index=9}
    }
    return sqSum.sqrt()
  }
}
