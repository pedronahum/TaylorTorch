import Foundation
import _Differentiation

/// Simple lookup embedding: maps token indices [N, L] → embeddings [N, L, C].
public struct Embedding: Layer {
  public var weight: Tensor  // [vocab, embed]
  @noDerivative public let vocabSize: Int
  @noDerivative public let embedSize: Int

  public typealias Input = Tensor  // Int64 indices, shape [N, L]
  public typealias Output = Tensor  // Float, shape [N, L, C]

  // Manual TangentVector to avoid nested synthesis pitfalls in large models.
  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var weight: Tensor
    public init(weight: Tensor = Tensor(0)) { self.weight = weight }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self { .init(weight: lhs.weight + rhs.weight) }
    public static func - (lhs: Self, rhs: Self) -> Self { .init(weight: lhs.weight - rhs.weight) }
  }

  public mutating func move(by d: TangentVector) { weight += d.weight }

  public init(vocabSize: Int, embedSize: Int, dtype: DType = .float32, device: Device = .cpu) {
    self.vocabSize = vocabSize
    self.embedSize = embedSize
    // Small uniform init.
    let scale = 0.02
    self.weight = Tensor.uniform(
      low: -scale, high: scale, shape: [vocabSize, embedSize], dtype: dtype, device: device)
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    precondition(x.rank == 2, "Embedding expects [N, L] indices.")
    // Gather rows and reshape back to [N, L, C].
    let n = withoutDerivative(at: x.shape[0])
    let l = withoutDerivative(at: x.shape[1])
    let c = withoutDerivative(at: weight.shape[1])
    let idx: [Int64] = withoutDerivative(at: x.toArray(as: Int64.self))
    let gathered = withoutDerivative(
      at: weight.indexSelect(dim: 0, indices: idx)  // [N*L, C]
    )
    return gathered.reshaped([n, l, c])
  }

  /// Explicit VJP: gather in forward; scatter-add rows in backward.
  @derivative(of: callAsFunction)
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (TangentVector, Tensor.TangentVector))
  {
    let y = callAsFunction(x)
    // Cache shapes + flat indices for scatter.
    let n = withoutDerivative(at: x.shape[0])
    let l = withoutDerivative(at: x.shape[1])
    let c = withoutDerivative(at: weight.shape[1])
    let flatCount = n * l
    let idx: [Int64] = withoutDerivative(at: x.toArray(as: Int64.self))
    let scatterDevice = withoutDerivative(at: weight.device)

    func pb(_ v: Tensor) -> (TangentVector, Tensor.TangentVector) {
      // v: [N, L, C] → [N*L, C] then scatter-add by row ids into dW:[V, C].
      let v2d = v.reshaped([flatCount, c])
      let zerosDType = withoutDerivative(at: weight.dtype ?? .float32)
      let idxTensor = Tensor(
        array: idx,
        shape: [flatCount],
        dtype: .int64,
        device: scatterDevice)
      var dW = Tensor.zeros(shape: weight.shape, dtype: zerosDType, device: scatterDevice)
      dW = dW.indexAdd(dim: 0, index: idxTensor, source: v2d)  // uses your scatter API
      return (TangentVector(weight: dW), .zero)
    }
    return (y, pb)
  }
}
