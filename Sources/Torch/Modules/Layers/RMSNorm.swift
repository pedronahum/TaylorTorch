import ATenCXX
import _Differentiation

public struct RMSNorm: Layer {
  public var weight: Tensor
  public var bias: Tensor
  @noDerivative public var epsilon: Double

  private struct ForwardCache {
    let normalized: Tensor
    let rms: Tensor
    let gammaBroadcast: Tensor
    let featureAxis: Int
    let featureCount: Int
    let reduceAxes: [Int]
  }

  public init(
    features: Int,
    epsilon: Double = 1e-5,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.weight = Tensor.ones(shape: [features], dtype: dtype, device: device)
    self.bias = Tensor.zeros(shape: [features], dtype: dtype, device: device)
    self.epsilon = epsilon
  }

  private func forwardWithCache(_ x: Tensor) -> (Tensor, ForwardCache) {
    let last = withoutDerivative(at: x.rank - 1)
    let x2 = x.multiplying(x)
    let epsTensor = withoutDerivative(at: Tensor(epsilon, dtype: x2.dtype ?? x.dtype!, device: x.device))
    let rms = x2.mean(dim: last, keepdim: true).adding(epsTensor).sqrt()

    let normalized = x.dividing(rms)

    let shape = withoutDerivative(at: x.shape)
    let broadcast = withoutDerivative(at: broadcastShape(for: shape, featureAxis: last))
    let gamma = weight.reshaped(broadcast)
    let beta = bias.reshaped(broadcast)

    let value = normalized.multiplying(gamma).adding(beta)

    let reduceAxes = withoutDerivative(at: (0..<x.rank).filter { $0 != last })
    let cache = ForwardCache(
      normalized: normalized,
      rms: rms,
      gammaBroadcast: gamma,
      featureAxis: last,
      featureCount: shape[last],
      reduceAxes: reduceAxes
    )

    return (value, cache)
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    forwardWithCache(x).0
  }

  @derivative(of: callAsFunction)
  @usableFromInline
  func vjpCallAsFunction(_ x: Tensor) -> (value: Tensor, pullback: (Tensor) -> (TangentVector, Tensor)) {
    let (value, cache) = forwardWithCache(x)
    return (
      value,
      { upstream in
        let gamma = cache.gammaBroadcast
        let normalized = cache.normalized
        let rms = cache.rms
        let featureAxis = cache.featureAxis
        let featureCount = Double(cache.featureCount)
        let reduceAxes = cache.reduceAxes

        let upstreamNorm = upstream.multiplying(gamma)
        let sumGX = upstreamNorm.multiplying(x).sum(dim: featureAxis, keepdim: true)
        let rmsCube = rms.multiplying(rms).multiplying(rms)
        let featureCountTensor = Tensor(featureCount, dtype: rms.dtype ?? upstream.dtype ?? .float32, device: rms.device)

        let gradInput = upstreamNorm.dividing(rms)
          .subtracting(x.multiplying(sumGX).dividing(rmsCube).dividing(featureCountTensor))

        var gradWeight = upstream.multiplying(normalized)
        for axis in reduceAxes.reversed() {
          gradWeight = gradWeight.sum(dim: axis)
        }

        var gradBias = upstream
        for axis in reduceAxes.reversed() {
          gradBias = gradBias.sum(dim: axis)
        }

        var tangent = TangentVector.zero
        tangent.weight = gradWeight
        tangent.bias = gradBias
        return (tangent, gradInput)
      }
    )
  }

  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
    bias.move(by: offset.bias)
  }

  public static var parameterKeyPaths: [WritableKeyPath<RMSNorm, Tensor>] {
    [\RMSNorm.weight, \RMSNorm.bias]
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var weight: Tensor
    public var bias: Tensor
    public static var zero: TangentVector { .init(weight: .zero, bias: .zero) }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(weight: l.weight.adding(r.weight), bias: l.bias.adding(r.bias))
    }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(
        weight: l.weight.adding(r.weight.multiplying(-1)),
        bias: l.bias.adding(r.bias.multiplying(-1)))
    }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\.weight, \.bias]
    }
  }

  private func broadcastShape(for input: [Int], featureAxis: Int) -> [Int] {
    var out = [Int](repeating: 1, count: input.count)
    out[featureAxis] = input[featureAxis]
    return out
  }
}
