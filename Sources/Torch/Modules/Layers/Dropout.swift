import Foundation
import _Differentiation

/// Inverted dropout: during training, randomly zeroes elements with probability `probability`,
/// scaling remaining activations by `1 / (1 - probability)`. At inference, returns the input.
public struct Dropout: ParameterlessLayer {
  // Hyperparameters (non-diff)
  @noDerivative public let probability: Float

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  /// - Parameter probability: Drop probability `p` in [0, 1).
  public init(probability p: Float = 0.5) {
    precondition(p >= 0 && p < 1, "Dropout probability must be in [0, 1).")
    self.probability = p
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Identity at inference or p == 0
    let phase = withoutDerivative(at: Context.local.learningPhase)
    let p = withoutDerivative(at: probability)
    if phase == .inference || p == 0 { return x }

    // Generate a Bernoulli(keepProb) mask via uniform thresholding.
    let keepProb = 1.0 - p
    let dtype = withoutDerivative(at: x.dtype ?? .float32)
    let dev = withoutDerivative(at: x.device)
    let u = Tensor.uniform(low: 0.0, high: 1.0, shape: x.shape, dtype: dtype, device: dev)
    let keepMask = withoutDerivative(at: u.gt(p))  // boolean

    // Inverted dropout: y = keep ? x / keepProb : 0
    let pos = x.dividing(keepProb)
    let zero = Tensor.zeros(shape: [], dtype: dtype, device: dev)
    return TorchWhere.select(condition: keepMask, pos, zero)
  }
}

// Manual VJPs that do not re-enter `callAsFunction`.
extension Dropout {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (EmptyTangentVector, Tensor))
  {
    let phase = withoutDerivative(at: Context.local.learningPhase)
    let p = withoutDerivative(at: probability)

    // Inference or p == 0 => identity and identity gradient.
    if phase == .inference || p == 0 {
      return (x, { upstream in (EmptyTangentVector(), upstream) })
    }

    let keepProb = 1.0 - p
    let dtype = withoutDerivative(at: x.dtype ?? .float32)
    let dev = withoutDerivative(at: x.device)

    // Make the mask here and capture it for the pullback.
    let u = Tensor.uniform(low: 0.0, high: 1.0, shape: x.shape, dtype: dtype, device: dev)
    let keepMask = withoutDerivative(at: u.gt(p))  // Bool tensor

    let pos = x.dividing(keepProb)
    let zero = Tensor.zeros(shape: [], dtype: dtype, device: dev)
    let y = TorchWhere.select(condition: keepMask, pos, zero)

    // Pullback: ∂L/∂x = upstream * keepMask / keepProb
    let scale = Tensor(1.0 / keepProb, dtype: dtype, device: dev)
    let maskF = keepMask.to(dtype: dtype)

    let pb: (Tensor) -> (EmptyTangentVector, Tensor) = { upstream in
      let dx = upstream.multiplying(maskF).multiplying(scale)
      return (EmptyTangentVector(), dx)
    }
    return (y, pb)
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> EmptyTangentVector)
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
