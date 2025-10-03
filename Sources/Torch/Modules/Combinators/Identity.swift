// Sources/Torch/Modules/Combinators/Identity.swift
//
// WHY: Useful “no‑op” building block when conditionally composing graphs,
// and as the neutral element for `Sequential` chains.

import _Differentiation

public struct Identity: Layer {
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { x }

  // Add the missing move(by:) function
  public mutating func move(by offset: TangentVector) {}

  // No trainable parameters.
  public static var parameterKeyPaths: [WritableKeyPath<Identity, Tensor>] { [] }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { TangentVector() }
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector { .init() }
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
