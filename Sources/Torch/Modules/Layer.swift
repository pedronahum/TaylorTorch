// Sources/Torch/Modules/Layer.swift
import _Differentiation

/// A trainable component mapping a Tensor to a Tensor.
/// Keeps the surface minimal and Swifty; build up from here.
public protocol Layer: EuclideanModel {
  @differentiable(reverse)
  func callAsFunction(_ input: Tensor) -> Tensor
}
