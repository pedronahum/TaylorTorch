// Sources/Torch/Modules/Layers/Permute.swift
//
// WHY
// - Reorders axes as a first-class Layer (e.g. [B,T,D] → [B,D,T]).
// - Common after Embedding for "channels-first" blocks, or before 1D/2D convolutions.
//
// Guarantees
// - Validates `axes` is a permutation of 0..<rank (after negative-index normalization).
// - Stateless, differentiable (VJP is just an inverse permutation).
//
// References: Layer protocol & context, Sequential/Builder, Identity stateless pattern.
import _Differentiation

public struct Permute: Layer {
  /// Target axis order; negative values are normalized w.r.t. input rank.
  @noDerivative public var axes: [Int]

  public init(_ axes: [Int]) { self.axes = axes }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let rank = withoutDerivative(at: x.rank)
    let perm = withoutDerivative(at: _normalizePermutation(axes, rank: rank))
    return x.permuted(perm)
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  // --- Stateless parameter boilerplate ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<Permute, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Helpers

@inline(__always)
private func _normalize(_ dim: Int, rank: Int) -> Int {
  let d = dim >= 0 ? dim : dim + rank
  precondition(d >= 0 && d < rank, "Axis \(dim) out of range for rank \(rank)")
  return d
}

private func _normalizePermutation(_ axes: [Int], rank: Int) -> [Int] {
  precondition(!axes.isEmpty, "Permutation must not be empty")
  precondition(axes.count == rank, "Permutation length must equal tensor rank")

  let perm = axes.map { _normalize($0, rank: rank) }
  var seen = [Bool](repeating: false, count: rank)
  for a in perm {
    precondition(!seen[a], "Axes must be a permutation (duplicate \(a))")
    seen[a] = true
  }
  return perm
}
