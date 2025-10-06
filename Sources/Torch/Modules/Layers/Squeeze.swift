// Sources/Torch/Modules/Shape/Squeeze.swift
//
// WHY
// Small utilities to drop or insert size‑1 dimensions as layers,
// keeping model definitions declarative (helpful when porting checkpoints).
//
// Wraps `Tensor.squeezed` and `Tensor.unsqueezed`.
import _Differentiation

/// Removes dimensions of size one from the tensor shape.
public struct Squeeze: Layer {
  /// If nil, squeezes all size‑1 dims. Otherwise squeezes only the given dim.
  @noDerivative public var dim: Int?

  /// Creates a squeeze layer.
  /// - Parameter dim: Specific dimension to remove, or `nil` to remove all singleton axes.
  public init(_ dim: Int? = nil) { self.dim = dim }

  /// Removes singleton dimensions according to `dim`.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor with singleton axes removed.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    if let raw = dim {
      let rank = withoutDerivative(at: x.rank)
      let d = raw >= 0 ? raw : raw + rank
      precondition(d >= 0 && d < rank, "Squeeze dim out of range")
      return x.squeezed(dim: d)
    } else {
      return x.squeezed()
    }
  }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor with singleton axes removed.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  /// Squeeze exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<Squeeze, Tensor>] { [] }

  /// Tangent representation for `Squeeze`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  /// Squeeze has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
}

/// Inserts a singleton dimension at the specified index.
public struct Unsqueeze: Layer {
  @noDerivative public var dim: Int

  /// Creates an unsqueeze layer.
  /// - Parameter dim: Dimension at which to insert a size-one axis (supports negatives).
  public init(_ dim: Int) { self.dim = dim }

  /// Inserts a singleton dimension according to `dim`.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor with an additional dimension of length one.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // PyTorch-style indexing: negative dims count into the *new* rank.
    // newRank = x.rank + 1
    let newRank = withoutDerivative(at: x.rank &+ 1)
    let d = dim >= 0 ? dim : dim + newRank
    precondition(d >= 0 && d <= newRank, "Unsqueeze dim out of range")
    return x.unsqueezed(dim: d)
  }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor with an additional dimension of length one.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  /// Unsqueeze exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<Unsqueeze, Tensor>] { [] }

  /// Tangent representation for `Unsqueeze`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  /// Unsqueeze has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
}
