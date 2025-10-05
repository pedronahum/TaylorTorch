// Sources/Torch/Modules/Shape/Squeeze.swift
//
// WHY
// Small utilities to drop or insert size‑1 dimensions as layers,
// keeping model definitions declarative (helpful when porting checkpoints).
//
// Wraps `Tensor.squeezed` and `Tensor.unsqueezed`.
import _Differentiation

public struct Squeeze: Layer {
  /// If nil, squeezes all size‑1 dims. Otherwise squeezes only the given dim.
  @noDerivative public var dim: Int?

  public init(_ dim: Int? = nil) { self.dim = dim }

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

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  public static var parameterKeyPaths: [WritableKeyPath<Squeeze, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  public mutating func move(by offset: TangentVector) {}
}

public struct Unsqueeze: Layer {
  @noDerivative public var dim: Int

  public init(_ dim: Int) { self.dim = dim }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // PyTorch-style indexing: negative dims count into the *new* rank.
    // newRank = x.rank + 1
    let newRank = withoutDerivative(at: x.rank &+ 1)
    let d = dim >= 0 ? dim : dim + newRank
    precondition(d >= 0 && d <= newRank, "Unsqueeze dim out of range")
    return x.unsqueezed(dim: d)
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  public static var parameterKeyPaths: [WritableKeyPath<Unsqueeze, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
  public mutating func move(by offset: TangentVector) {}
}
