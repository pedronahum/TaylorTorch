// Sources/Torch/Modules/Layer.swift
//
// WHY: A minimal common surface for trainable modules.
// - Sits on top of EuclideanModel (so optimizers can use vector/tangent views).
// - Adds an optional "contextful" call to future‑proof stochastic/stateful layers
//   (Dropout, BatchNorm) without changing the primary call signature.
//
// Fits the existing stack: Linear/Sequential already conform naturally. See:
//  - ParameterIterable/EuclideanModel/Tangent algebra used by optimizers.
//  - Optimizers expect Model: ParameterIterableModel.
//                                └> Layer : EuclideanModel : ParameterIterableModel
//
// References: ParameterIterable & Euclidean implementations and optimizers.
//   ParameterIterable.swift, EuclideanModel.swift, Optimizers.swift

import _Differentiation

public protocol Layer: EuclideanModel {
  /// Pure, differentiable forward pass.
  @differentiable(reverse)
  func callAsFunction(_ input: Tensor) -> Tensor

  /// Contextual entry point. Default forwards to `callAsFunction`.
  /// Use this in layers that behave differently in training vs inference.
  @differentiable(reverse)
  func call(_ input: Tensor, context: ForwardContext) -> Tensor
}

extension Layer {
  @differentiable(reverse)
  public func call(_ input: Tensor, context: ForwardContext) -> Tensor {
    // Default: ignore context, behave like the pure call.
    callAsFunction(input)
  }
}
