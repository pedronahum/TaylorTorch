//
// Sources/Torch/Modules/Builders/LayerBuilder.swift
//
// Unlimited-depth result builder that folds statements into nested
// `Sequential<_, _>` types. Conservative about inlining/resilience.
//

import _Differentiation

// MARK: - Result builder with unbounded depth

@resultBuilder
public enum LayerBuilder {
  // Stream the first statement into the accumulator.
  public static func buildPartialBlock<L: Layer>(first l: L) -> L { l }

  // Fold each next statement by nesting into `Sequential<Acc, Next>`.
  public static func buildPartialBlock<Acc: Layer, Next: Layer>(
    accumulated acc: Acc, next: Next
  ) -> Sequential<Acc, Next> {
    Sequential(acc, next)
  }

  // Typed conditionals: allow `if/else` when both branches produce the same `L`.
  public static func buildEither<L: Layer>(first l: L) -> L { l }
  public static func buildEither<L: Layer>(second l: L) -> L { l }

  // Allow empty builder blocks to compile as a no‑op.
  public static func buildBlock() -> Identity { Identity() }
}

// MARK: - Wrapper to hide the nested type while keeping it zero‑cost.

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

  public static var parameterKeyPaths: [WritableKeyPath<SequentialBlock, Tensor>] {
    var paths: [WritableKeyPath<SequentialBlock, Tensor>] = []
    for kp in Body.parameterKeyPaths {
      // Use the same helper you use elsewhere.
      paths.append((\SequentialBlock.body).appending(kp))
    }
    return paths
  }

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
