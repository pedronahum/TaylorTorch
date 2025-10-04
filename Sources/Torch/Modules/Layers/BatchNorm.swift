// Sources/Torch/Modules/Layers/BatchNorm.swift
//
// BatchNorm1D / BatchNorm2D with context-driven training/inference behavior.
// - Train: normalize with batch stats; update running stats (momentum).
// - Eval : normalize with running stats.
// - Affine scale/shift are trainable parameters; running stats are not.
//
// Integrates with:
//  - Layer + ForwardContext (training gate).               (see: Layer.swift, ForwardContext.swift)
//  - ParameterIterable / EuclideanModel for optimizers.    (see: ParameterIterable.swift, EuclideanModel.swift)
//  - DataFormat for 2D (NCHW/NHWC).                        (see: DataFormat.swift)

import _Differentiation

// Small reference wrapper to allow in-place state updates from a non-mutating call.
public final class _TensorBox {
  public var value: Tensor
  public init(_ value: Tensor) { self.value = value }
}

// MARK: - BatchNorm1D  (expects [N, F] — features last)

public struct BatchNorm1D: Layer {
  // Trainable affine params
  public var weight: Tensor   // gamma [F]
  public var bias: Tensor     // beta  [F]

  // Non-trainable running stats (per-feature)
  @noDerivative public var runningMean: _TensorBox // [F]
  @noDerivative public var runningVar: _TensorBox  // [F]

  // Hyper-parameters
  @noDerivative public var momentum: Double   // smoothing factor (≈ PyTorch's 0.1)
  @noDerivative public var epsilon: Double    // numerical stability (e.g. 1e-5)

  public init(
    numFeatures: Int,
    momentum: Double = 0.1,
    epsilon: Double = 1e-5,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.weight = Tensor.ones(shape: [numFeatures], dtype: dtype, device: device)
    self.bias   = Tensor.zeros(shape: [numFeatures], dtype: dtype, device: device)
    self.runningMean = _TensorBox(Tensor.zeros(shape: [numFeatures], dtype: dtype, device: device))
    self.runningVar  = _TensorBox(Tensor.ones(shape: [numFeatures], dtype: dtype, device: device))
    self.momentum = momentum
    self.epsilon = epsilon
  }

  // Inference path (pure): use running stats
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let m = runningMean.value
    let v = runningVar.value
    return _batchNorm1D(
      x, mean: m, variance: v, weight: weight, bias: bias, eps: epsilon)
  }

  // Training/Eval path with context
  @differentiable(reverse, wrt: (self, x))
  public func call(_ x: Tensor, context: @noDerivative ForwardContext) -> Tensor {
    guard context.training else { return self(x) }

    // Batch stats along the batch axis (dim 0)
    let batchMean = x.mean(dim: 0)
    let featureCount1D = withoutDerivative(at: batchMean.shape[0])
    let batchBroadcastShape = withoutDerivative(at: [1, featureCount1D])
    let centered = x.subtracting(batchMean.reshaped(batchBroadcastShape))
    let batchVar = centered.multiplying(centered).mean(dim: 0)

    // Update running stats (non-differentiable state)
    let (batchMeanND, batchVarND) = withoutDerivative(at: (batchMean, batchVar))
    let mom = Tensor(momentum).to(dtype: batchMeanND.dtype!).to(device: batchMeanND.device)
    let oneMinus = Tensor(1.0, dtype: mom.dtype!, device: mom.device).subtracting(mom)
    runningMean.value = runningMean.value.multiplying(oneMinus).adding(batchMeanND.multiplying(mom))
    runningVar.value = runningVar.value.multiplying(oneMinus).adding(batchVarND.multiplying(mom))

    return _batchNorm1D(
      x, mean: batchMean, variance: batchVar, weight: weight, bias: bias, eps: epsilon)
  }

  // Shared compute
  @differentiable(reverse)
  private func _batchNorm1D(
    _ x: Tensor, mean: Tensor, variance: Tensor, weight: Tensor, bias: Tensor, eps: Double
  ) -> Tensor {
    let epsTensor = withoutDerivative(at: Tensor(
      eps,
      dtype: variance.dtype ?? x.dtype!,
      device: variance.device
    ))
    let featureCount = withoutDerivative(at: weight.shape[0])
    let broadcastShape = withoutDerivative(at: [1, featureCount])
    let denom = variance.adding(epsTensor).sqrt().reshaped(broadcastShape)
    let meanB = mean.reshaped(broadcastShape)
    let norm = x.subtracting(meanB).dividing(denom)
    let weightB = weight.reshaped(broadcastShape)
    let biasB = bias.reshaped(broadcastShape)
    return norm.multiplying(weightB).adding(biasB)
  }

  // --- Layer plumbing ---
  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
    bias.move(by: offset.bias)
  }
  public static var parameterKeyPaths: [WritableKeyPath<BatchNorm1D, Tensor>] {
    [\BatchNorm1D.weight, \BatchNorm1D.bias]
  }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var weight: Tensor
    public var bias: Tensor
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [\.weight, \.bias] }
    public static var zero: TangentVector { .init(weight: .zero, bias: .zero) }
    public static func + (l: Self, r: Self) -> Self { .init(weight: l.weight + r.weight, bias: l.bias + r.bias) }
    public static func - (l: Self, r: Self) -> Self { .init(weight: l.weight - r.weight, bias: l.bias - r.bias) }
  }
}

extension BatchNorm1D {
  @usableFromInline
  @derivative(of: call(_:context:))
  func vjpCall(_ x: Tensor, context: @noDerivative ForwardContext)
    -> (value: Tensor, pullback: (Tensor) -> (TangentVector, Tensor))
  {
    if !context.training {
      let mean = runningMean.value
      let variance = runningVar.value
      let value = _batchNorm1D(
        x,
        mean: mean,
        variance: variance,
        weight: weight,
        bias: bias,
        eps: epsilon)

      let featureCount = withoutDerivative(at: weight.shape[0])
      let reshape = withoutDerivative(at: [1, featureCount])
      let epsTensor = Tensor(epsilon, dtype: variance.dtype ?? x.dtype!, device: variance.device)
      let denomVec = variance.adding(epsTensor).sqrt()
      let denom = denomVec.reshaped(reshape)
      let centered = x.subtracting(mean.reshaped(reshape))
      let normalized = centered.dividing(denom)
      let weightB = weight.reshaped(reshape)

      return (value, { upstream in
        let dBeta = upstream.sum(dim: 0)
        let dGamma = upstream.multiplying(normalized).sum(dim: 0)
        let dx = upstream.multiplying(weightB).dividing(denom)
        var tangent = TangentVector.zero
        tangent.weight = dGamma
        tangent.bias = dBeta
        return (tangent, dx)
      })
    }

    let batchMean = x.mean(dim: 0)
    let featureCount1D = withoutDerivative(at: batchMean.shape[0])
    let broadcastShape = withoutDerivative(at: [1, featureCount1D])
    let centered = x.subtracting(batchMean.reshaped(broadcastShape))
    let batchVar = centered.multiplying(centered).mean(dim: 0)
    let value = _batchNorm1D(
      x,
      mean: batchMean,
      variance: batchVar,
      weight: weight,
      bias: bias,
      eps: epsilon)

    let epsTensor = Tensor(epsilon, dtype: batchVar.dtype ?? x.dtype!, device: batchVar.device)
    let varPlusEps = batchVar.adding(epsTensor)
    let denomVec = varPlusEps.sqrt()
    let invStdVec = varPlusEps.pow(-0.5)
    let invStdCubeVec = varPlusEps.pow(-1.5)
    let denom = denomVec.reshaped(broadcastShape)
    let invStd = invStdVec.reshaped(broadcastShape)
    let normalized = centered.dividing(denom)
    let weightB = weight.reshaped(broadcastShape)
    let n = Double(x.shape[0])

    return (value, { upstream in
      let dBeta = upstream.sum(dim: 0)
      let dGamma = upstream.multiplying(normalized).sum(dim: 0)
      let dxHat = upstream.multiplying(weightB)
      let dVar = dxHat.multiplying(centered).sum(dim: 0)
        .multiplying(-0.5)
        .multiplying(invStdCubeVec)
      let dMean = dxHat.multiplying(invStd).negated().sum(dim: 0)
        .adding(dVar.multiplying(centered.sum(dim: 0)).multiplying(-2.0 / n))
      let dx = dxHat.multiplying(invStd)
        .adding(centered.multiplying(dVar.reshaped(broadcastShape)).multiplying(2.0 / n))
        .adding(dMean.reshaped(broadcastShape).dividing(n))

      var tangent = TangentVector.zero
      tangent.weight = dGamma
      tangent.bias = dBeta
      return (tangent, dx)
    })
  }
}

// MARK: - BatchNorm2D  (supports NCHW/NHWC)

public struct BatchNorm2D: Layer {
  public var weight: Tensor   // gamma [C]
  public var bias: Tensor     // beta  [C]

  @noDerivative public var runningMean: _TensorBox // [C]
  @noDerivative public var runningVar: _TensorBox  // [C]

  @noDerivative public var momentum: Double
  @noDerivative public var epsilon: Double
  @noDerivative public var dataFormat: DataFormat  // NCHW or NHWC

  public init(
    numFeatures: Int,
    momentum: Double = 0.1,
    epsilon: Double = 1e-5,
    dataFormat: DataFormat = .nchw,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.weight = Tensor.ones(shape: [numFeatures], dtype: dtype, device: device)
    self.bias   = Tensor.zeros(shape: [numFeatures], dtype: dtype, device: device)
    self.runningMean = _TensorBox(Tensor.zeros(shape: [numFeatures], dtype: dtype, device: device))
    self.runningVar  = _TensorBox(Tensor.ones(shape: [numFeatures], dtype: dtype, device: device))
    self.momentum = momentum
    self.epsilon = epsilon
    self.dataFormat = dataFormat
  }

  // Inference: running stats
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let m = runningMean.value
    let v = runningVar.value
    return _batchNorm2D(
      x, mean: m, variance: v, weight: weight, bias: bias, eps: epsilon, format: dataFormat)
  }

  // Training: batch stats + running updates
  @differentiable(reverse, wrt: (self, x))
  public func call(_ x: Tensor, context: @noDerivative ForwardContext) -> Tensor {
    guard context.training else { return self(x) }

    let shape = withoutDerivative(at: x.shape)
    let (reduceDims, broadcastShape) = withoutDerivative(at: _layoutInfo(shape: shape, format: dataFormat))
    // mean over N,H,W (reduceDims)
    var mean = x
    for d in reduceDims.sorted(by: >) { mean = mean.mean(dim: d) }  // [C]
    // var = E[(x - mean)^2]
    let centered = x.subtracting(mean.reshaped(broadcastShape))
    var varT = centered.multiplying(centered)
    for d in reduceDims.sorted(by: >) { varT = varT.mean(dim: d) }  // [C]

    let (meanND, varND) = withoutDerivative(at: (mean, varT))
    let mom = Tensor(momentum).to(dtype: meanND.dtype!).to(device: meanND.device)
    let oneMinus = Tensor(1.0, dtype: mom.dtype!, device: mom.device).subtracting(mom)
    runningMean.value = runningMean.value.multiplying(oneMinus).adding(meanND.multiplying(mom))
    runningVar.value = runningVar.value.multiplying(oneMinus).adding(varND.multiplying(mom))

    return _batchNorm2D(
      x, mean: mean, variance: varT, weight: weight, bias: bias, eps: epsilon, format: dataFormat)
  }

  // Shared compute
  @differentiable(reverse)
  private func _batchNorm2D(
    _ x: Tensor, mean: Tensor, variance: Tensor,
    weight: Tensor, bias: Tensor, eps: Double, format: DataFormat
  ) -> Tensor {
    let shape = withoutDerivative(at: x.shape)
    let (_, broadcastShape) = withoutDerivative(at: _layoutInfo(shape: shape, format: format))
    let epsTensor = withoutDerivative(at: Tensor(
      eps,
      dtype: variance.dtype ?? x.dtype!,
      device: variance.device
    ))
    let denom = variance.adding(epsTensor).sqrt().reshaped(broadcastShape)
    let meanB = mean.reshaped(broadcastShape)
    let weightB = weight.reshaped(broadcastShape)
    let biasB = bias.reshaped(broadcastShape)
    let norm = x.subtracting(meanB).dividing(denom)
    return norm.multiplying(weightB).adding(biasB)
  }

  // Layout helpers
  private func _layoutInfo(shape: [Int], format: DataFormat)
    -> (reduceDims: [Int], broadcastShape: [Int])
  {
    switch format {
    case .nchw:
      // x: [N, C, H, W]; reduce over N,H,W => dims [0,2,3]; broadcast [1,C,1,1]
      return (
        [0, 2, 3],
        [1, shape[1], 1, 1]
      )
    case .nhwc:
      // x: [N, H, W, C]; reduce over N,H,W => dims [0,1,2]; broadcast [1,1,1,C]
      return (
        [0, 1, 2],
        [1, 1, 1, shape[3]]
      )
    }
  }

  // --- Layer plumbing ---
  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
    bias.move(by: offset.bias)
  }
  public static var parameterKeyPaths: [WritableKeyPath<BatchNorm2D, Tensor>] {
    [\BatchNorm2D.weight, \BatchNorm2D.bias]
  }
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var weight: Tensor
    public var bias: Tensor
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [\.weight, \.bias] }
    public static var zero: TangentVector { .init(weight: .zero, bias: .zero) }
    public static func + (l: Self, r: Self) -> Self { .init(weight: l.weight + r.weight, bias: l.bias + r.bias) }
    public static func - (l: Self, r: Self) -> Self { .init(weight: l.weight - r.weight, bias: l.bias - r.bias) }
  }
}


extension BatchNorm2D {
  @usableFromInline
  @derivative(of: call(_:context:))
  func vjpCall(_ x: Tensor, context: @noDerivative ForwardContext)
    -> (value: Tensor, pullback: (Tensor) -> (TangentVector, Tensor))
  {
    let shape = withoutDerivative(at: x.shape)
    let (reduceDims, broadcastShape) = withoutDerivative(at: _layoutInfo(shape: shape, format: dataFormat))
    let reduceOrder = withoutDerivative(at: reduceDims.sorted(by: >))
    let elementsPerChannel = withoutDerivative(at: reduceDims.reduce(1) { $0 * shape[$1] })
    let count = Double(elementsPerChannel)

    func reducedSum(_ tensor: Tensor) -> Tensor {
      var result = tensor
      for dim in reduceOrder { result = result.sum(dim: dim) }
      return result
    }

    if !context.training {
      let mean = runningMean.value
      let variance = runningVar.value
      let value = _batchNorm2D(
        x,
        mean: mean,
        variance: variance,
        weight: weight,
        bias: bias,
        eps: epsilon,
        format: dataFormat)

      let epsTensor = Tensor(epsilon, dtype: variance.dtype ?? x.dtype!, device: variance.device)
      let denomVec = variance.adding(epsTensor).sqrt()
      let denom = denomVec.reshaped(broadcastShape)
      let centered = x.subtracting(mean.reshaped(broadcastShape))
      let normalized = centered.dividing(denom)
      let weightB = weight.reshaped(broadcastShape)

      return (value, { upstream in
        let dBeta = reducedSum(upstream)
        let dGamma = reducedSum(upstream.multiplying(normalized))
        let dx = upstream.multiplying(weightB).dividing(denom)
        var tangent = TangentVector.zero
        tangent.weight = dGamma
        tangent.bias = dBeta
        return (tangent, dx)
      })
    }

    var mean = x
    for dim in reduceOrder { mean = mean.mean(dim: dim) }
    let centered = x.subtracting(mean.reshaped(broadcastShape))
    var variance = centered.multiplying(centered)
    for dim in reduceOrder { variance = variance.mean(dim: dim) }

    let value = _batchNorm2D(
      x,
      mean: mean,
      variance: variance,
      weight: weight,
      bias: bias,
      eps: epsilon,
      format: dataFormat)

    let epsTensor = Tensor(epsilon, dtype: variance.dtype ?? x.dtype!, device: variance.device)
    let varPlusEps = variance.adding(epsTensor)
    let denomVec = varPlusEps.sqrt()
    let invStdVec = varPlusEps.pow(-0.5)
    let invStdCubeVec = varPlusEps.pow(-1.5)
    let denom = denomVec.reshaped(broadcastShape)
    let invStd = invStdVec.reshaped(broadcastShape)
    let normalized = centered.dividing(denom)
    let weightB = weight.reshaped(broadcastShape)

    return (value, { upstream in
      let dBeta = reducedSum(upstream)
      let dGamma = reducedSum(upstream.multiplying(normalized))
      let dxHat = upstream.multiplying(weightB)
      let dVar = reducedSum(dxHat.multiplying(centered)).multiplying(-0.5).multiplying(invStdCubeVec)
      let dMean = reducedSum(dxHat.multiplying(invStd).negated())
        .adding(dVar.multiplying(reducedSum(centered)).multiplying(-2.0 / count))
      let dVarB = dVar.reshaped(broadcastShape)
      let dMeanB = dMean.reshaped(broadcastShape)
      let dx = dxHat.multiplying(invStd)
        .adding(centered.multiplying(dVarB).multiplying(2.0 / count))
        .adding(dMeanB.dividing(count))

      var tangent = TangentVector.zero
      tangent.weight = dGamma
      tangent.bias = dBeta
      return (tangent, dx)
    })
  }
}
