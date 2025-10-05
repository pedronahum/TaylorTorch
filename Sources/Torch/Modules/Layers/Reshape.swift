// Sources/Torch/Modules/Layers/Reshape.swift
//
// WHY
// - Safe, differentiable shape changes with optional one-dimension inference (-1).
// - Keeps "shape logic" out of training code; composes as a Layer in your DSL.
//
// Notes
// - One `-1` in the target infers that dimension from the input element-count.
// - Validates that the element counts match.
//
// References: Layer.swift (context), Sequential/Builder, Identity stateless pattern.
import _Differentiation

public struct Reshape: Layer {
  /// Target shape specification; may contain at most one `-1` (infer).
  @noDerivative public var target: [Int]

  public init(_ target: [Int]) {
    self.target = target
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let inShape = withoutDerivative(at: x.shape)
    let inCount = withoutDerivative(at: inShape.reduce(1, *))
    var out = withoutDerivative(at: target)

    // Compute inferred dim if any.
    var inferIndex: Int? = nil
    var knownProd = 1
    for (i, s) in out.enumerated() {
      if s == -1 {
        precondition(inferIndex == nil, "Only one -1 (inferred) dimension is allowed")
        inferIndex = i
      } else {
        precondition(s > 0, "Reshape dims must be positive (or -1 for a single inferred dim)")
        knownProd *= s
      }
    }

    if let idx = inferIndex {
      precondition(knownProd != 0 && inCount % knownProd == 0, "Element count mismatch")
      out[idx] = inCount / knownProd
    }

    let outCount = withoutDerivative(at: out.reduce(1, *))
    precondition(outCount == inCount, "Element count mismatch in Reshape")
    return x.reshaped(out)
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  // --- Stateless parameter boilerplate ---
  public mutating func move(by offset: TangentVector) {}
  public static var parameterKeyPaths: [WritableKeyPath<Reshape, Tensor>] { [] }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public init() {}
    public static var zero: TangentVector { .init() }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
