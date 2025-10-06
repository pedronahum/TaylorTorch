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

/// Reorders tensor axes according to a supplied permutation.
public struct Permute: Layer {
  /// Target axis order; negative values are normalized w.r.t. input rank.
  @noDerivative public var axes: [Int]

  /// Creates a permutation layer.
  /// - Parameter axes: Desired axis order, possibly containing negative indices.
  public init(_ axes: [Int]) { self.axes = axes }

  /// Applies the axis permutation.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor with axes reordered to match `axes`.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let rank = withoutDerivative(at: x.rank)
    let perm = withoutDerivative(at: _normalizePermutation(axes, rank: rank))
    return x.permuted(perm)
  }

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor with axes reordered to match `axes`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  // --- Stateless parameter boilerplate ---
  /// Permute has no trainable parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Permute exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<Permute, Tensor>] { [] }
  /// Tangent representation for `Permute`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}

// MARK: - Helpers

/// Normalizes a potentially-negative axis index.
@inline(__always)
private func _normalize(_ dim: Int, rank: Int) -> Int {
  let d = dim >= 0 ? dim : dim + rank
  precondition(d >= 0 && d < rank, "Axis \(dim) out of range for rank \(rank)")
  return d
}

/// Validates and normalizes a permutation of axes.
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
