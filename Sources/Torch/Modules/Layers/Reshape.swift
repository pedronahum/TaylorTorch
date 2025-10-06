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

/// Reshapes tensors to a target shape while optionally inferring one dimension.
public struct Reshape: Layer {
  /// Target shape specification; may contain at most one `-1` (infer).
  @noDerivative public var target: [Int]

  /// Creates a reshape layer.
  /// - Parameter target: Desired output shape with at most one inferred dimension (`-1`).
  public init(_ target: [Int]) {
    self.target = target
  }

  /// Reshapes the input tensor to the target shape.
  /// - Parameter x: Input tensor.
  /// - Returns: Tensor with the new shape.
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

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context (unused).
  /// - Returns: Tensor with the new shape.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  // --- Stateless parameter boilerplate ---
  /// Reshape has no parameters, so applying `offset` is a no-op.
  public mutating func move(by offset: TangentVector) {}
  /// Reshape exposes no trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<Reshape, Tensor>] { [] }
  /// Tangent representation for `Reshape`, which is always empty.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Creates an empty tangent vector.
    public init() {}
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init() }
    /// No parameter key paths are exposed.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [] }
  }
}
