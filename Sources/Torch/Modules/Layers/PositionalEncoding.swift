import Foundation
import _Differentiation

/// Sinusoidal positional encoding added to token embeddings.
/// Returns a broadcastable [1, L, C] that is added to [N, L, C].
public struct PositionalEncoding: ParameterlessLayer {
  @noDerivative public let maxLength: Int
  @noDerivative public let embedSize: Int
  @noDerivative public let dtype: DType
  @noDerivative public let device: Device

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  public init(maxLength: Int, embedSize: Int, dtype: DType = .float32, device: Device = .cpu) {
    self.maxLength = maxLength
    self.embedSize = embedSize
    self.dtype = dtype
    self.device = device
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // x: [N, L, C]  → y: [N, L, C]  (adds [1, L, C])
    precondition(x.rank == 3 && x.shape[2] == embedSize, "PosEnc expects [N, L, \(embedSize)].")
    let inputLength = x.shape[1]
    let configuredMax = withoutDerivative(at: maxLength)
    precondition(
      inputLength <= configuredMax,
      "PosEnc maxLength \(configuredMax) is smaller than sequence length \(inputLength).")

    let inputDevice = withoutDerivative(at: x.device)
    let configuredDevice = withoutDerivative(at: device)
    precondition(
      inputDevice == configuredDevice,
      "PosEnc configured for \(configuredDevice) but received tensor on \(inputDevice).")

    let l = x.shape[1]
    let c = embedSize
    precondition(c.isMultiple(of: 2), "PosEnc embedSize must be even to pair sin/cos terms.")
    let pos = Tensor.arange(
      Double(0), to: Double(l), step: 1.0, dtype: .float64, device: configuredDevice)
      .unsqueezed(dim: 1)  // [L, 1]
    let i = Tensor.arange(
      Double(0), to: Double(c / 2), step: 1.0, dtype: .float64, device: configuredDevice)
    let div = (i * 2.0).dividing(Double(c))  // [C/2]
    let denom = Tensor(Double(10000.0), dtype: .float64, device: configuredDevice)
      .pow(div)
      .unsqueezed(dim: 0)  // [1, C/2]

    let arg = pos.dividing(denom)  // [L, C/2]
    let peSin = arg.sin()
    let peCos = arg.cos()
    let targetDType = withoutDerivative(at: x.dtype ?? dtype)
    let evenIdx = Tensor.arange(
      Int64(0), to: Int64(c), step: Int64(2), dtype: .int64, device: configuredDevice)
    let oddIdx = Tensor.arange(
      Int64(1), to: Int64(c), step: Int64(2), dtype: .int64, device: configuredDevice)
    var base = Tensor.zeros(shape: [l, c], dtype: targetDType, device: configuredDevice)
    let sinTyped = peSin.to(dtype: targetDType).to(device: configuredDevice)
    let cosTyped = peCos.to(dtype: targetDType).to(device: configuredDevice)
    base = base.indexCopy(dim: 1, index: evenIdx, source: sinTyped)
    base = base.indexCopy(dim: 1, index: oddIdx, source: cosTyped)
    let encoding = withoutDerivative(
      at: base
        .unsqueezed(dim: 0)  // [1, L, C]
        .broadcasted(to: [x.shape[0], l, c])  // [N, L, C]
    )
    return x + encoding
  }

  @derivative(of: callAsFunction)
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector)
    )
  {
    // Positional encodings are constants; gradient flows through identity.
    let y = callAsFunction(x)
    return (
      y,
      { v in
        (EmptyTangentVector(), v)
      }
    )
  }
}
