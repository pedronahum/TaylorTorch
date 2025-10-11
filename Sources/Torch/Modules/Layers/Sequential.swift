import Foundation
import _Differentiation

// MARK: - Identity (parameterless pass-through)

/// A parameterless layer that returns its input unchanged.
public struct Identity<IO: Differentiable>: ParameterlessLayer {
  public typealias Input = IO
  public typealias Output = IO
  public typealias TangentVector = EmptyTangentVector

  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: IO) -> IO { x }

}

extension Sequential {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector) -> (TangentVector, Input.TangentVector)
    )
  {
    // Differentiate through the stored `body` directly.
    let (y, bodyPB) = body.appliedForBackpropagation(to: x)
    return (
      y,
      { v in
        let (dBody, dX) = bodyPB(v)  // dBody : Body.TangentVector
        return (
          TangentVector(body: dBody),  // wrap into Sequential.TangentVector
          dX
        )
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector) -> TangentVector
    )
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - Chain (two-layer composition with explicit tangent)

/// Compose two layers `First` → `Second` such that `First.Output == Second.Input`.
public struct Chain<First: Layer, Second: Layer>: Layer where First.Output == Second.Input {
  public var first: First
  public var second: Second

  public init(_ first: First, _ second: Second) {
    self.first = first
    self.second = second
  }

  public typealias Input = First.Input
  public typealias Output = Second.Output

  // Manual TangentVector to avoid synthesis pitfalls and guarantee AdditiveArithmetic witnesses.
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // explicit zero/+/- to avoid solver corner cases
    KeyPathIterable,  // required by Module
    VectorProtocol,  // required by Module
    PointwiseMultiplicative  // required by Module
  {
    public typealias VectorSpaceScalar = Float
    public var first: First.TangentVector
    public var second: Second.TangentVector

    public init(first: First.TangentVector = .zero, second: Second.TangentVector = .zero) {
      self.first = first
      self.second = second
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(first: lhs.first + rhs.first, second: lhs.second + rhs.second)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(first: lhs.first - rhs.first, second: lhs.second - rhs.second)
    }
  }

  public mutating func move(by d: TangentVector) {
    first.move(by: d.first)
    second.move(by: d.second)
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Input) -> Output {
    let y = first(x)
    return second(y)
  }

  // Manual VJPs to avoid “curried self” solver path.
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector) -> (TangentVector, Input.TangentVector)
    )
  {
    // Forward through first, then second.
    let y1 = first(x)
    let (y, pbSecond) = second.appliedForBackpropagation(to: y1)
    return (
      y,
      { v in
        // Backprop through second, then first.
        let (dSecond, dY1) = pbSecond(v)
        let (_, pbFirst) = first.appliedForBackpropagation(to: x)
        let (dFirst, dX) = pbFirst(dY1)
        return (TangentVector(first: dFirst, second: dSecond), dX)
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Input)
    -> (value: Output, pullback: (Output.TangentVector) -> TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - Result builder for sequential models

@resultBuilder
public enum SequentialBuilder {
  /// First element starts the chain as-is.
  public static func buildPartialBlock<First: Layer>(first: First) -> First { first }

  /// Append the next layer, ensuring the types line up: `Accum.Output == Next.Input`.
  public static func buildPartialBlock<Accum: Layer, Next: Layer>(
    accumulated: Accum, next: Next
  ) -> Chain<Accum, Next> where Accum.Output == Next.Input {
    Chain(accumulated, next)
  }

  // If you later want `if`/`else` in builders, add:
  // public static func buildEither<TrueBranch: Layer, FalseBranch: Layer>(
  //   first: TrueBranch
  // ) -> TrueBranch { first }
  // public static func buildEither<TrueBranch: Layer, FalseBranch: Layer>(
  //   second: FalseBranch
  // ) -> FalseBranch { second }
  //
  // And possibly buildOptional/buildArray with an Identity wrapper.
}

// MARK: - Sequential (thin, differentiable wrapper over the built chain)

/// A “Swifty” sequential container built via `@SequentialBuilder`.
///
/// Usage:
/// ```swift
/// let model = Sequential {
///   Linear(inFeatures: 128, outFeatures: 256)
///   ReLU()
///   Linear(inFeatures: 256, outFeatures: 10)
/// }
/// let logits = model(x)
/// ```
public struct Sequential<Body: Layer>: Layer {
  public var body: Body

  public init(@SequentialBuilder _ layers: () -> Body) {
    self.body = layers()
  }

  public typealias Input = Body.Input
  public typealias Output = Body.Output

  // Manual TangentVector: a simple pass-through to Body’s tangent.
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var body: Body.TangentVector

    public init(body: Body.TangentVector = .zero) { self.body = body }

    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self { .init(body: lhs.body + rhs.body) }
    public static func - (lhs: Self, rhs: Self) -> Self { .init(body: lhs.body - rhs.body) }
  }

  public mutating func move(by d: TangentVector) { body.move(by: d.body) }

  @differentiable(reverse)
  public func callAsFunction(_ x: Input) -> Output { body(x) }

}
