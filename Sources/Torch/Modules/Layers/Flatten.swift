// Sources/Torch/Modules/Layers/Flatten.swift
//
// WHY
// - Make shape changes first-class Layers so they compose in Sequential/Builder.
// - Typical usage: after Embedding [B,T,D] → Flatten(startDim: 1) = [B, T*D],
//   or Conv2D [B,C,H,W] → Flatten(startDim: 1) = [B, C*H*W].
// - Stateless, context-agnostic. Plays nicely with optimizers & Euclidean views.
//
// References: Layer.swift (context + EuclideanModel), Sequential/Builder, Identity.
//             See repository files for the same patterns.
//             (Layer)        (Seq/Builder)             (stateless TV)
//                ⤷           ⤷                        ⤷
//                 Torch/Modules/Layer.swift           Torch/Modules/Combinators/Identity.swift
//                                                    Torch/Modules/Layers/Pooling.swift
import _Differentiation

/// Flattens adjacent tensor dimensions into a single dimension.
public struct Flatten: Layer {
  /// First dimension to flatten (can be negative; normalized at runtime).
  @noDerivative public var startDim: Int
  /// Last dimension (inclusive) to flatten (can be negative).
  @noDerivative public var endDim: Int

  /// Default flattens all but the batch: [B, *] → [B, -1].
  /// - Parameters:
  ///   - startDim: First dimension to collapse.
  ///   - endDim: Last dimension (inclusive) to collapse.
  public init(startDim: Int = 1, endDim: Int = -1) {
    self.startDim = startDim
    self.endDim = endDim
  }

  /// Collapses the specified range of dimensions into one.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor with the specified dimensions flattened.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let rank = withoutDerivative(at: x.rank)
    let s = _normalize(startDim, rank)
    let e = _normalize(endDim, rank)
    precondition(rank > 0 && s >= 0 && e >= s && e < rank, "Invalid flatten dims")
    var shape = withoutDerivative(at: x.shape)
    let flatCount = shape[s...e].reduce(1, *)
    shape.replaceSubrange(s...e, with: [flatCount])
    return x.reshaped(shape)
  }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward-context structure (unused).
  /// - Returns: Tensor with the specified dimensions flattened.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  // --- Stateless-parameter boilerplate (matches Identity/Pooling) ---
  /// Flatten has no trainable parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Flatten exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<Flatten, Tensor>] { [] }
  /// Tangent representation for `Flatten`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

/// Normalizes possibly-negative dimension indices given the tensor rank.
@inline(__always)
private func _normalize(_ dim: Int, _ rank: Int) -> Int { dim >= 0 ? dim : (dim + rank) }
