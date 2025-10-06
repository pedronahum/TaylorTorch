// Sources/Torch/Modules/Layers/Sequential+Context.swift
//
// WHY: Thread `ForwardContext` through the existing Sequential<L1,L2> without
// touching the file. Keeps typed composition and prepares for dropout/batchnorm.

import _Differentiation

extension Sequential {
  /// Applies the composed layers while threading through a `ForwardContext`.
  /// - Parameters:
  ///   - x: Input tensor.
  ///   - context: Forward context shared across both layers.
  /// - Returns: Output of `l2(l1(x, context), context)`.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    l2.call(l1.call(x, context: context), context: context)
  }
}
