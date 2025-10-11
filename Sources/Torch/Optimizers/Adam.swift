import _Differentiation
import Darwin

/// Adam optimizer (with optional AdamW-style decoupled weight decay).
///
/// Works with any Differentiable `Model` whose `TangentVector` conforms to:
/// - `VectorProtocol` (for `scaled(by:)`)
/// - `KeyPathIterable` (to traverse tensor leaves)
/// - `PointwiseMultiplicative` (not strictly required by this implementation, but
///   common across your layers and useful if you extend it later).
public final class Adam<Model: Differentiable & EuclideanDifferentiable>: Optimizer
where
  Model.TangentVector: VectorProtocol & KeyPathIterable & PointwiseMultiplicative,
  Model.TangentVector.VectorSpaceScalar == Float
{
  // Satisfy Optimizer associated types.
  public typealias Scalar = Float
  public typealias Model = Model

  // MARK: - Hyperparameters

  /// Base learning rate (α).
  public var learningRate: Float
  /// First moment decay (β₁).
  public var beta1: Float
  /// Second moment decay (β₂).
  public var beta2: Float
  /// Numerical stability constant (ε).
  public var epsilon: Float
  /// Optional learning‑rate decay (SGD‑style: α_t = α / (1 + decay * t)).
  public var decay: Float
  /// Weight decay coefficient. If `adamW` is `true`, uses decoupled weight decay (AdamW).
  public var weightDecay: Float
  /// If `true`, applies decoupled weight decay (AdamW); if `false`, uses no decay.
  public var adamW: Bool

  // MARK: - State

  /// First moment (per-parameter).
  public var m: Model.TangentVector
  /// Second raw moment (per-parameter).
  public var v: Model.TangentVector
  /// Time step (t ≥ 0).
  public var step: Int = 0

  // MARK: - Init

  /// Creates Adam with sensible defaults.
  public init(
    for _: __shared Model,
    learningRate: Float = 1e-3,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-8,
    decay: Float = 0,
    weightDecay: Float = 0,
    adamW: Bool = true
  ) {
    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.decay = decay
    self.weightDecay = weightDecay
    self.adamW = adamW

    // Start moments at zero (same structure as model tangents).
    self.m = .zero
    self.v = .zero
  }

  // MARK: - Device copy (matches your SGD pattern)

  /// Create a copy of this optimizer moving its state (m, v) to `device`.
  public required init(copying other: Adam, to device: Device) {
    self.learningRate = other.learningRate
    self.beta1 = other.beta1
    self.beta2 = other.beta2
    self.epsilon = other.epsilon
    self.decay = other.decay
    self.weightDecay = other.weightDecay
    self.adamW = other.adamW
    self.step = other.step

    // Move first/second moments to the requested device by visiting all Tensor leaves.
    var mCopy = other.m
    for kp in mCopy.recursivelyAllWritableKeyPaths(to: Tensor.self) {
      mCopy[keyPath: kp] = mCopy[keyPath: kp].to(device: device)
    }
    for kp in mCopy.recursivelyAllWritableKeyPaths(to: Tensor?.self) {
      if let t = mCopy[keyPath: kp] { mCopy[keyPath: kp] = t.to(device: device) }
    }
    for kp in mCopy.recursivelyAllWritableKeyPaths(to: [Tensor].self) {
      mCopy[keyPath: kp] = mCopy[keyPath: kp].map { $0.to(device: device) }
    }

    var vCopy = other.v
    for kp in vCopy.recursivelyAllWritableKeyPaths(to: Tensor.self) {
      vCopy[keyPath: kp] = vCopy[keyPath: kp].to(device: device)
    }
    for kp in vCopy.recursivelyAllWritableKeyPaths(to: Tensor?.self) {
      if let t = vCopy[keyPath: kp] { vCopy[keyPath: kp] = t.to(device: device) }
    }
    for kp in vCopy.recursivelyAllWritableKeyPaths(to: [Tensor].self) {
      vCopy[keyPath: kp] = vCopy[keyPath: kp].map { $0.to(device: device) }
    }

    self.m = mCopy
    self.v = vCopy
  }

  // MARK: - Step

  /// Applies one Adam update along the provided gradient direction.
  ///
  /// - Parameters:
  ///   - model: model parameters to update in place
  ///   - direction: gradient w.r.t. the model (∂L/∂θ)
  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    step &+= 1

    // Optional LR decay (same functional form you used in SGD).
    let lr = learningRate * (1.0 / (1.0 + decay * Float(step)))
    let b1 = beta1
    let b2 = beta2
    let eps = epsilon

    // Update first/second moments per Tensor leaf:
    var newM = m
    var newV = v

    for kp in newM.recursivelyAllWritableKeyPaths(to: Tensor.self) {
      let g = direction[keyPath: kp]
      // m_t = β1 * m_{t-1} + (1-β1) * g_t
      newM[keyPath: kp] = newM[keyPath: kp].multiplying(b1)
        .adding(g.multiplying(1 - b1))
    }
    for kp in newV.recursivelyAllWritableKeyPaths(to: Tensor.self) {
      let g = direction[keyPath: kp]
      // v_t = β2 * v_{t-1} + (1-β2) * (g_t ⊙ g_t)
      let gg = g.multiplying(g)
      newV[keyPath: kp] = newV[keyPath: kp].multiplying(b2)
        .adding(gg.multiplying(1 - b2))
    }

    // Bias correction factors:
    let bc1 = 1 - powf(b1, Float(step))
    let bc2 = 1 - powf(b2, Float(step))

    // Compute the preconditioned step:  m̂ / (sqrt(v̂) + ε)
    var precond = Model.TangentVector.zero
    for kp in precond.recursivelyAllWritableKeyPaths(to: Tensor.self) {
      let mHat = newM[keyPath: kp].dividing(bc1)
      let vHat = newV[keyPath: kp].dividing(bc2)
      let denom = vHat.sqrt().adding(
        Tensor(eps, dtype: vHat.dtype ?? .float32, device: vHat.device))
      precond[keyPath: kp] = mHat.dividing(denom)
    }

    // Optional decoupled weight decay (AdamW): add λ * θ to the step before scaling by -lr.
    if adamW, weightDecay != 0 {
      var wd = model.differentiableVectorView  // same structure as parameters
      for kp in wd.recursivelyAllWritableKeyPaths(to: Tensor.self) {
        wd[keyPath: kp] = wd[keyPath: kp].multiplying(weightDecay)
      }
      precond = precond + wd
    }

    // Apply update: θ ← θ - lr * precond
    model.move(by: precond.scaled(by: -lr))

    // Commit moments.
    m = newM
    v = newV
  }
}
