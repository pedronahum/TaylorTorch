// Sources/Torch/Modules/Context/ForwardContext.swift
//
// WHY: Plumbs information like `training` (and later RNG) through a model
// without changing every layer’s signature. Layers that don’t care can ignore it.
// Adds no dependency to optimizers or parameters.

import _Differentiation

public struct ForwardContext {
  /// True during training (dropout on, batchnorm updates running stats).
  public var training: Bool

  public init(training: Bool = false) {
    self.training = training
  }
}
