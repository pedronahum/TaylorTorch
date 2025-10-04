// Sources/Torch/Modules/Layers/LayerNorm.swift
//
// WHY
// Layer Normalization (Ba et al., 2016) normalizes activations *per sample* over
// the last k feature axes. It is batch-size independent (unlike BatchNorm),
// numerically stable (epsilon in the denominator), and widely used in
// transformers and RNNs.
//
// Design
// - Conforms to the shared `Layer` surface (pure call + contextual call). No
//   running stats; train/inference behavior is identical.
// - Parameters are `gamma` (scale) and `beta` (offset). They broadcast across the
//   normalized axes and are discoverable via `parameterKeyPaths`, so optimizers
//   and Euclidean algebra work with no extra code.
//
// References
// - S4TF `Normalization.swift` (LayerNorm semantics).
// - Keras / TF docs for axis, epsilon, and broadcasting behavior.
//
// See also: Layer.swift, ParameterIterable.swift, EuclideanModel.swift.

import _Differentiation

public struct LayerNorm: Layer {
  // Trainable parameters spanning the normalized trailing shape.
  public var gamma: Tensor
  public var beta: Tensor

  // How many *trailing* axes to normalize over (e.g., 1 = last axis).
  @noDerivative public var normalizedRank: Int
  // Numerical stability constant added inside the square root.
  @noDerivative public var epsilon: Double

  // MARK: - Inits

  /// Normalize across the last `normalizedRank` axes whose combined shape is `normalizedShape`.
  /// `gamma` and `beta` are initialized to ones/zeros with that trailing shape.
  public init(
    normalizedShape: [Int],
    epsilon: Double = 1e-5,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    precondition(!normalizedShape.isEmpty, "normalizedShape must have at least one dimension")
    self.gamma = Tensor.ones(shape: normalizedShape, dtype: dtype, device: device)
    self.beta = Tensor.zeros(shape: normalizedShape, dtype: dtype, device: device)
    self.normalizedRank = normalizedShape.count
    self.epsilon = epsilon
  }

  /// Convenience: normalize along the last axis (feature dimension).
  public init(
    featureCount: Int,
    epsilon: Double = 1e-5,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.init(normalizedShape: [featureCount], epsilon: epsilon, dtype: dtype, device: device)
  }

  // MARK: - Forward

  /// y = ((x - μ) / sqrt(σ² + ε)) * γ + β
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    _layerNormForward(x).value
  }

  @usableFromInline
  func _layerNormForward(_ x: Tensor) -> (
    value: Tensor,
    normalized: Tensor,
    centered: Tensor,
    denom: Tensor,
    axes: [Int],
    broadcastShape: [Int],
    leadingRank: Int,
    normalizedShape: [Int],
    gammaB: Tensor
  ) {
    let r = withoutDerivative(at: x.rank)
    precondition(
      normalizedRank > 0 && normalizedRank <= r,
      "normalizedRank (\(normalizedRank)) must be in 1...\(r)"
    )

    let axes = withoutDerivative(at: Array((r - normalizedRank)..<r))

    var mean = x
    for d in axes { mean = mean.mean(dim: d, keepdim: true) }

    let centered = x.subtracting(mean)
    var variance = centered.multiplying(centered)
    for d in axes { variance = variance.mean(dim: d, keepdim: true) }

    let epsTensor = withoutDerivative(at: Tensor(
      epsilon,
      dtype: variance.dtype ?? x.dtype!,
      device: variance.device
    ))
    let denom = variance.adding(epsTensor).sqrt()
    let normalized = centered.dividing(denom)

    let normalizedShape = withoutDerivative(at: gamma.shape)
    let leadingRank = withoutDerivative(at: r - normalizedRank)
    let leadingOnes = withoutDerivative(at: Array(repeating: 1, count: max(leadingRank, 0)))
    let broadcastShape = withoutDerivative(at: leadingOnes + normalizedShape)
    let gammaB = gamma.reshaped(broadcastShape)
    let betaB = beta.reshaped(broadcastShape)
    let value = normalized.multiplying(gammaB).adding(betaB)

    return (
      value: value,
      normalized: normalized,
      centered: centered,
      denom: denom,
      axes: axes,
      broadcastShape: broadcastShape,
      leadingRank: leadingRank,
      normalizedShape: normalizedShape,
      gammaB: gammaB
    )
  }

  /// Contextual entry point. LayerNorm is stateless (no running stats), so this
  /// simply forwards to the pure call.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor { callAsFunction(x) }

  // MARK: - Parameter traversal & AD plumbing

  public mutating func move(by offset: TangentVector) {
    gamma.move(by: offset.gamma)
    beta.move(by: offset.beta)
  }

  public static var parameterKeyPaths: [WritableKeyPath<LayerNorm, Tensor>] {
    [\LayerNorm.gamma, \LayerNorm.beta]
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var gamma: Tensor
    public var beta: Tensor

    public static var zero: TangentVector { .init(gamma: .zero, beta: .zero) }
    public static func + (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(gamma: l.gamma.adding(r.gamma), beta: l.beta.adding(r.beta))
    }
    public static func - (l: TangentVector, r: TangentVector) -> TangentVector {
      .init(
        gamma: l.gamma.adding(r.gamma.multiplying(-1)),
        beta: l.beta.adding(r.beta.multiplying(-1))
      )
    }

    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      [\TangentVector.gamma, \TangentVector.beta]
    }
  }
}

extension LayerNorm {
  @usableFromInline
  @derivative(of: callAsFunction)
  func vjpCallAsFunction(_ x: Tensor) -> (value: Tensor, pullback: (Tensor) -> (TangentVector, Tensor)) {
    let cache = _layerNormForward(x)
    let axesSorted = withoutDerivative(at: cache.axes.sorted(by: >))
    func reduceOverAxes(_ tensor: Tensor, keepdim: Bool) -> Tensor {
      var result = tensor
      for dim in axesSorted {
        result = result.sum(dim: dim, keepdim: keepdim)
      }
      return result
    }

    let leadingRank = cache.leadingRank
    let leadingAxesSorted = withoutDerivative(at: Array(0..<max(leadingRank, 0)).sorted(by: >))
    func reduceLeading(_ tensor: Tensor) -> Tensor {
      var result = tensor
      for dim in leadingAxesSorted {
        result = result.sum(dim: dim, keepdim: false)
      }
      return result
    }

    let normalizedShape = cache.normalizedShape
    let elementsPerSample = withoutDerivative(at: normalizedShape.reduce(1, *))
    let dtype = cache.normalized.dtype ?? cache.gammaB.dtype ?? x.dtype ?? .float32
    let device = cache.normalized.device
    let nTensor = withoutDerivative(at: Tensor(Double(elementsPerSample), dtype: dtype, device: device))
    let invNTensor = withoutDerivative(at: Tensor(1.0 / Double(elementsPerSample), dtype: dtype, device: device))

    return (
      value: cache.value,
      pullback: { upstream in
        let dBeta = reduceLeading(upstream)
        let dGamma = reduceLeading(upstream.multiplying(cache.normalized))

        let sumDy = reduceOverAxes(upstream, keepdim: true)
        let sumDyNormalized = reduceOverAxes(upstream.multiplying(cache.normalized), keepdim: true)

        let numerator = upstream.multiplying(nTensor)
          .subtracting(sumDy)
          .subtracting(cache.normalized.multiplying(sumDyNormalized))
        let dx = cache.gammaB
          .dividing(cache.denom)
          .multiplying(numerator)
          .multiplying(invNTensor)

        var tangent = TangentVector.zero
        tangent.gamma = dGamma
        tangent.beta = dBeta
        return (tangent, dx)
      }
    )
  }
}
