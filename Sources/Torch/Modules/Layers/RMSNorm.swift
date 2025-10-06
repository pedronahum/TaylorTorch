import ATenCXX
import _Differentiation

/// Root-mean-square normalization initialized with per-feature scale and bias.
public struct RMSNorm: Layer {
  /// Learnable scaling factors.
  public var weight: Tensor
  /// Learnable offsets.
  public var bias: Tensor
  /// Numerical stability constant added to the RMS computation.
  @noDerivative public var epsilon: Double

  /// Cached forward-pass intermediates used by the custom VJP.
  private struct ForwardCache {
    let normalized: Tensor
    let rms: Tensor
    let gammaBroadcast: Tensor
    let featureAxis: Int
    let featureCount: Int
    let reduceAxes: [Int]
  }

  /// Creates an RMS normalization layer.
  /// - Parameters:
  ///   - features: Number of features on the final axis.
  ///   - epsilon: Small constant added to the RMS for stability.
  ///   - dtype: Element dtype for the parameters.
  ///   - device: Device on which to allocate the parameters.
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

  /// Performs the forward pass and returns cached tensors for the backward pass.
  /// - Parameter x: Input activations.
  /// - Returns: Tuple containing the normalized output and cached intermediates.
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

  /// Applies RMS normalization to `x`.
  /// - Parameter x: Input activations.
  /// - Returns: Normalized activations.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    forwardWithCache(x).0
  }

  /// Custom VJP that leverages cached intermediates for the backward pass.
  /// - Parameter x: Input activations.
  /// - Returns: Layer output and a pullback producing parameter and input gradients.
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

  /// Contextual forward that proxies to `callAsFunction`.
  /// - Parameters:
  ///   - x: Input activations.
  ///   - context: Forward context (unused).
  /// - Returns: Normalized activations.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(x)
  }

  /// Applies the tangent `offset` to the layer's parameters.
  /// - Parameter offset: Tangent vector from differentiation.
  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
    bias.move(by: offset.bias)
  }

  /// Writable key paths for trainable parameters.
  public static var parameterKeyPaths: [WritableKeyPath<RMSNorm, Tensor>] {
    [\RMSNorm.weight, \RMSNorm.bias]
  }

  /// Tangent representation for `RMSNorm`.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Tangent for the scaling factors.
    public var weight: Tensor
    /// Tangent for the bias parameters.
    public var bias: Tensor
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init(weight: .zero, bias: .zero) }
    /// Adds two tangent vectors element-wise.
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(weight: l.weight.adding(r.weight), bias: l.bias.adding(r.bias))
    }
    /// Subtracts two tangent vectors element-wise.
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(
        weight: l.weight.adding(r.weight.multiplying(-1)),
        bias: l.bias.adding(r.bias.multiplying(-1)))
    }
    /// Writable key paths for the tangent components.
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\.weight, \.bias]
    }
  }

  /// Computes the broadcast shape for per-feature parameters.
  private func broadcastShape(for input: [Int], featureAxis: Int) -> [Int] {
    var out = [Int](repeating: 1, count: input.count)
    out[featureAxis] = input[featureAxis]
    return out
  }
}
