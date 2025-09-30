// Sources/Torch/Modules/Builders/LayerBuilder.swift
//
// WHY: Lets users write typed stacks ergonomically:
//
//   let model = SequentialBlock {
//     Linear.glorot(inFeatures: 784, outFeatures: 256)
//     ReLU()
//     Linear.glorot(inFeatures: 256, outFeatures: 10)
//   }
//
// Under the hood, this lowers to nested `Sequential<_,_>` types—so it stays
// fully generic and zero‑cost (no type erasure). It composes with your
// ParameterIterable traversal and optimizers.
//
// Works with your existing `Sequential<L1,L2>` implementation. :contentReference[oaicite:5]{index=5}

import _Differentiation

@resultBuilder
public enum LayerBuilder {
  public static func buildBlock<L: Layer>(_ l: L) -> L { l }

  public static func buildBlock<L1: Layer, L2: Layer>(_ l1: L1, _ l2: L2)
    -> Sequential<L1, L2>
  { Sequential(l1, l2) }

  public static func buildBlock<L1: Layer, L2: Layer, L3: Layer>(
    _ l1: L1, _ l2: L2, _ l3: L3
  ) -> Sequential<Sequential<L1, L2>, L3> {
    Sequential(Sequential(l1, l2), l3)
  }

  public static func buildBlock<L1: Layer, L2: Layer, L3: Layer, L4: Layer>(
    _ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4
  ) -> Sequential<Sequential<Sequential<L1, L2>, L3>, L4> {
    Sequential(Sequential(Sequential(l1, l2), l3), l4)
  }

  public static func buildBlock<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer>(
    _ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5
  ) -> Sequential<Sequential<Sequential<Sequential<L1, L2>, L3>, L4>, L5> {
    Sequential(Sequential(Sequential(Sequential(l1, l2), l3), l4), l5)
  }

  public static func buildBlock<L1: Layer, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer>(
    _ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6
  ) -> Sequential<
    Sequential<Sequential<Sequential<Sequential<L1, L2>, L3>, L4>, L5>, L6
  > {
    Sequential(Sequential(Sequential(Sequential(Sequential(l1, l2), l3), l4), l5), l6)
  }

  // Extend with more overloads if you routinely build deeper stacks.
}

/// A thin wrapper that hides the nested type in API surfaces.
/// Still zero‑cost: it stores the fully‑typed body and forwards everything.
public struct SequentialBlock<Body: Layer>: Layer {
  public var body: Body

  public init(@LayerBuilder _ make: () -> Body) { self.body = make() }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { body(x) }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    body.call(x, context: context)
  }

  public mutating func move(by offset: TangentVector) { body.move(by: offset.body) }

  // Parameter traversal forwards to `Body` by composing key paths via `\.body`.
  public static var parameterKeyPaths: [WritableKeyPath<SequentialBlock, Tensor>] {
    var paths: [WritableKeyPath<SequentialBlock, Tensor>] = []
    for kp in Body.parameterKeyPaths {
      paths.append((\SequentialBlock.body).appending(kp))
    }
    return paths
  }

  /// Mirror the tangent structure so optimizers stay shape‑aligned.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var body: Body.TangentVector

    public static var zero: TangentVector { TangentVector(body: .zero) }
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(body: lhs.body + rhs.body)
    }
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(body: lhs.body - rhs.body)
    }

    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      var paths: [WritableKeyPath<TangentVector, Tensor>] = []
      for kp in Body.TangentVector.parameterKeyPaths {
        paths.append((\TangentVector.body).appending(kp))
      }
      return paths
    }
  }
}
