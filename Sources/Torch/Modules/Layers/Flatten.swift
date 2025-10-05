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

public struct Flatten: Layer {
  /// First dimension to flatten (can be negative; normalized at runtime).
  @noDerivative public var startDim: Int
  /// Last dimension (inclusive) to flatten (can be negative).
  @noDerivative public var endDim: Int

  /// Default flattens all but the batch: [B, *] → [B, -1].
  public init(startDim: Int = 1, endDim: Int = -1) {
    self.startDim = startDim
    self.endDim = endDim
  }

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

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  // --- Stateless-parameter boilerplate (matches Identity/Pooling) ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<Flatten, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

@inline(__always)
private func _normalize(_ dim: Int, _ rank: Int) -> Int { dim >= 0 ? dim : (dim + rank) }
