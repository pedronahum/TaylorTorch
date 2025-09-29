import Foundation
import _Differentiation

public protocol Optimizer {
  associatedtype Model: ParameterIterableModel
  mutating func update(_ model: inout Model, along direction: Model.TangentVector)
}

// MARK: - SGD

public struct SGD<Model: ParameterIterableModel>: Optimizer {
  public var learningRate: Double
  public var momentum: Double
  public var nesterov: Bool
  public var weightDecayL2: Double?  // classic L2 (coupled)
  public var clipGlobalNorm: Double?  // optional global-norm clipping

  // State (same structure as Model.TangentVector)
  internal var velocity: Model.TangentVector?

  public init(
    for model: __owned Model,
    learningRate: Double,
    momentum: Double = 0.0,
    nesterov: Bool = false,
    weightDecayL2: Double? = nil,
    clipGlobalNorm: Double? = nil
  ) {
    self.learningRate = learningRate
    self.momentum = momentum
    self.nesterov = nesterov
    self.weightDecayL2 = weightDecayL2
    self.clipGlobalNorm = clipGlobalNorm
    // State is lazily shaped on first use from model/grad shapes.
    self.velocity = nil
  }

  public mutating func update(_ model: inout Model, along rawGrad: Model.TangentVector) {
    var g = rawGrad

    // Optional: global-norm clipping (uses your reductions & sqrt). :contentReference[oaicite:10]{index=10}
    if let clip = clipGlobalNorm {
      let norm = Model.globalNorm(of: g)  // norm already on the right device/dtype
      let tiny = Tensor(1e-12).to(dtype: norm.dtype!).to(device: norm.device)
      let clipT = Tensor(clip).to(dtype: norm.dtype!).to(device: norm.device)
      let factor = clipT.dividing(norm.maximum(tiny))  // clip / max(norm, tiny)
      let capped = Tensor(1.0).to(dtype: factor.dtype!).to(device: factor.device).minimum(factor)
      g = Model.map(g) { $0.multiplying(capped) }
    }

    // Optional: classic L2 (coupled) weight decay: g += wd * w
    if let wd = weightDecayL2 {
      let w = model.asTangentVector()
      g = Model.add(g, Model.scale(w, by: wd))
    }

    // Momentum / Nesterov
    if momentum > 0 {
      if velocity == nil { velocity = model.zerosLikeParameters() }  // state with right shapes
      var v = velocity!
      v = Model.add(Model.scale(v, by: momentum), g)  // v = μ v + g
      velocity = v
      let step = nesterov ? Model.add(Model.scale(v, by: momentum), g) : v
      let delta = Model.scale(step, by: -learningRate)
      model.move(by: delta)  // standard AD move
    } else {
      let delta = Model.scale(g, by: -learningRate)
      model.move(by: delta)
    }
  }
}

// MARK: - AdamW (decoupled weight decay)

public struct AdamW<Model: ParameterIterableModel>: Optimizer {
  public var learningRate: Double
  public var beta1: Double
  public var beta2: Double
  public var epsilon: Double
  public var weightDecay: Double  // decoupled (AdamW)
  public var clipGlobalNorm: Double?

  internal var step: Int = 0
  internal var m: Model.TangentVector
  internal var v: Model.TangentVector

  public init(
    for model: __owned Model,
    learningRate: Double = 1e-3,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8,
    weightDecay: Double = 0.0,
    clipGlobalNorm: Double? = nil
  ) {
    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.weightDecay = weightDecay
    self.clipGlobalNorm = clipGlobalNorm

    // Initialize state with correct shapes/dtypes/devices.
    let zeros = model.zerosLikeParameters()
    self.m = zeros
    self.v = zeros
  }

  public mutating func update(_ model: inout Model, along rawGrad: Model.TangentVector) {
    var g = rawGrad

    // Clip (optional)
    if let clip = clipGlobalNorm {
      let norm = Model.globalNorm(of: g)
      let tiny = Tensor(1e-12).to(dtype: norm.dtype!).to(device: norm.device)
      let clipT = Tensor(clip).to(dtype: norm.dtype!).to(device: norm.device)
      let factor = clipT.dividing(norm.maximum(tiny))
      let capped = Tensor(1.0).to(dtype: factor.dtype!).to(device: factor.device).minimum(factor)
      g = Model.map(g) { $0.multiplying(capped) }
    }

    // Bias correction is already using Foundation.pow (good).
    // The rest of your AdamW is solid.

    // Moments
    step &+= 1
    let one = 1.0
    m = Model.add(Model.scale(m, by: beta1), Model.scale(g, by: (one - beta1)))
    let g2 = Model.hadamard(g, g)
    v = Model.add(Model.scale(v, by: beta2), Model.scale(g2, by: (one - beta2)))

    // Bias correction
    let b1c = one - Foundation.pow(beta1, Double(step))
    let b2c = one - Foundation.pow(beta2, Double(step))
    let mHat = Model.scale(m, by: 1.0 / b1c)
    let vHat = Model.scale(v, by: 1.0 / b2c)

    // step = mHat / (sqrt(vHat) + eps)
    let denom = Model.addEpsilon(Model.sqrt(vHat), epsilon)
    var stepTV = Model.ewiseDiv(mHat, denom)

    // Decoupled weight decay: step += wd * w
    if weightDecay != 0 {
      let w = model.asTangentVector()
      stepTV = Model.add(stepTV, Model.scale(w, by: weightDecay))
    }

    // Apply update
    let delta = Model.scale(stepTV, by: -learningRate)
    model.move(by: delta)
  }
}
