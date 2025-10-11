import Foundation
import _Differentiation

/// Batch Normalization over channel dimension (NCHW...).
///
/// Shapes:
///   - Rank-2 (MLP): x [N, C]
///   - Rank-4 (Conv): x [N, C, H, W]
///
/// Training:
///   y = (x - μ_B) / sqrt(σ_B^2 + ε) * γ + β
/// Inference:
///   y = (x - runningMean) / sqrt(runningVar + ε) * γ + β
public struct BatchNorm: Layer {
  // Trainable affine parameters
  public var gamma: Tensor  // [C]
  public var beta: Tensor  // [C]

  // Running statistics (updated in training; read in inference)
  @noDerivative public var runningMean: Parameter  // [C]
  @noDerivative public var runningVariance: Parameter  // [C]

  // Hyper-parameters
  @noDerivative public var momentum: Float
  @noDerivative public var epsilon: Float
  @noDerivative public var affine: Bool

  public typealias Input = Tensor
  public typealias Output = Tensor

  // MARK: - Manual TangentVector (avoid synthesis)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var gamma: Tensor
    public var beta: Tensor

    public init(gamma: Tensor = Tensor(0), beta: Tensor = Tensor(0)) {
      self.gamma = gamma
      self.beta = beta
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        gamma: Self.binaryOp(lhs.gamma, rhs.gamma, +, label: "gamma"),
        beta: Self.binaryOp(lhs.beta, rhs.beta, +, label: "beta"))
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        gamma: Self.binaryOp(lhs.gamma, rhs.gamma, -, label: "gamma"),
        beta: Self.binaryOp(lhs.beta, rhs.beta, -, label: "beta"))
    }

    @inline(__always)
    private static func binaryOp(
      _ lhs: Tensor, _ rhs: Tensor,
      _ op: (Tensor, Tensor) -> Tensor,
      label: StaticString
    ) -> Tensor {
      var left = lhs
      var right = rhs

      let reference = right.count >= left.count ? right : left
      let targetDevice = reference.device
      let targetDType = reference.dtype ?? left.dtype ?? right.dtype

      if left.device != targetDevice { left = left.to(device: targetDevice) }
      if right.device != targetDevice { right = right.to(device: targetDevice) }

      if let dtype = targetDType {
        if left.dtype != dtype { left = left.to(dtype: dtype) }
        if right.dtype != dtype { right = right.to(dtype: dtype) }
      }

      if left.shape != right.shape {
        if left.count == right.count, left.count != 0 {
          left = left.reshaped(right.shape)
        } else if left.count == 1, right.count > 1 {
          let zeros = Tensor.zeros(
            shape: right.shape,
            dtype: right.dtype ?? targetDType ?? .float32,
            device: right.device)
          left = zeros.adding(left)
        } else if right.count == 1, left.count > 1 {
          let zeros = Tensor.zeros(
            shape: left.shape,
            dtype: left.dtype ?? targetDType ?? .float32,
            device: left.device)
          right = zeros.adding(right)
        } else if left.count == right.count, left.count == 0 {
          // Nothing to do — both are empty tensors but shapes differ.
        } else {
          preconditionFailure(
            "BatchNorm.TangentVector mismatch for \(label): lhs \(left.shape) vs rhs \(right.shape)")
        }
      }

      return op(left, right)
    }
  }

  public mutating func move(by d: TangentVector) {
    gamma += Self.alignTangentComponent(d.gamma, to: gamma, label: "gamma")
    beta += Self.alignTangentComponent(d.beta, to: beta, label: "beta")
  }

  // MARK: - Init

  /// Create BatchNorm for `channels` with defaults matching PyTorch-style BN.
  public init(
    channels: Int,
    momentum: Float = 0.1,
    epsilon: Float = 1e-5,
    affine: Bool = true,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.gamma = Tensor.ones(shape: [channels], dtype: dtype, device: device)
    self.beta = Tensor.zeros(shape: [channels], dtype: dtype, device: device)
    self.runningMean = Parameter(Tensor.zeros(shape: [channels], dtype: dtype, device: device))
    self.runningVariance = Parameter(Tensor.ones(shape: [channels], dtype: dtype, device: device))
    self.momentum = momentum
    self.epsilon = epsilon
    self.affine = affine
  }

  /// Backward-compat alias (drop-in).
  public init(
    featureCount: Int,
    momentum: Float = 0.1,
    epsilon: Float = 1e-5,
    affine: Bool = true,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.init(
      channels: featureCount, momentum: momentum, epsilon: epsilon,
      affine: affine, dtype: dtype, device: device)
  }

  // MARK: - Helpers (shapes & reductions)

  @inlinable
  @inline(__always)
  internal func paramExpandShape(for x: Tensor) -> [Int] {
    // [1, C, 1, 1, ...] to match NCHW...
    let c = x._dimSize(1)  // internal helper, end-exclusive semantics. :contentReference[oaicite:8]{index=8}
    precondition(
      c == gamma.shape[0],
      "BatchNorm: channel mismatch: x has C=\(c), gamma has \(gamma.shape[0])")
    var shape = [Int](repeating: 1, count: x.rank)
    shape[1] = c
    return shape
  }

  @inlinable
  @inline(__always)
  internal func reduceAxes(for x: Tensor) -> [Int] {
    // All dims except channel (1)
    precondition(x.rank >= 2, "BatchNorm requires rank >= 2 (got \(x.rank)).")
    var axes = [Int]()
    axes.reserveCapacity(x.rank - 1)
    for d in 0..<x.rank where d != 1 { axes.append(d) }
    return axes
  }

  @inlinable
  internal func expandedParams(for x: Tensor) -> (Tensor, Tensor) {
    let gView = broadcastParam(gamma, for: x)
    let bView = broadcastParam(beta, for: x)
    return (gView, bView)
  }

  @inlinable
  internal func expandedRunning(for x: Tensor) -> (Tensor, Tensor) {
    let meanValue = withoutDerivative(at: runningMean.value)
    let varValue = withoutDerivative(at: runningVariance.value)
    return (broadcastParam(meanValue, for: x), broadcastParam(varValue, for: x))
  }

  @inlinable
  internal func computeBatchStats(_ x: Tensor) -> (meanKeep: Tensor, varKeep: Tensor) {
    // Reduce with keepdim to keep broadcasting simple; sequential reductions are safe. :contentReference[oaicite:10]{index=10}
    var mean = x
    var var_ = x
    let axes = withoutDerivative(at: reduceAxes(for: x))
    for ax in axes {
      mean = mean.mean(dim: ax, keepdim: true)
    }
    let centered = x - mean
    var_ = centered.multiplying(centered)
    for ax in axes {
      var_ = var_.mean(dim: ax, keepdim: true)
    }
    return (meanKeep: mean, varKeep: var_)
  }

  @inlinable
  internal func normalize(_ x: Tensor, meanKeep: Tensor, varKeep: Tensor) -> Tensor {
    // 1 / sqrt(var + eps)
    let dtype = withoutDerivative(at: x.dtype ?? .float32)
    let device = withoutDerivative(at: x.device)
    let eps = Tensor(self.epsilon, dtype: dtype, device: device)
    let invStd =
      (Tensor.ones(shape: [], dtype: dtype, device: device)
        .dividing((varKeep + eps).sqrt()))
    // (x - mean) * invStd
    return (x - meanKeep).multiplying(invStd)  // shapes are aligned due to keepdim
  }

  @inlinable
  @inline(__always)
  internal func broadcastParam(_ param: Tensor, for x: Tensor) -> Tensor {
    let inputRank = withoutDerivative(at: x.rank)
    precondition(inputRank >= 2, "BatchNorm requires rank >= 2 input (got \(inputRank)).")
    let paramRank = withoutDerivative(at: param.rank)
    precondition(paramRank == 1, "BatchNorm parameters must be rank-1 (got rank \(paramRank)).")
    var view = param
    // Insert batch axis at the front.
    view = view.unsqueezed(dim: 0)
    // Append singleton axes for spatial dimensions beyond channel.
    if inputRank > 2 {
      for axis in 2..<inputRank {
        view = view.unsqueezed(dim: axis)
      }
    }
    return view
  }

  // MARK: - Forward

  @inlinable
  @inline(__always)
  internal func forwardTraining(_ x: Tensor) -> (output: Tensor, meanVec: Tensor, varVec: Tensor) {
    let (meanKeep, varKeep) = computeBatchStats(x)
    var y = normalize(x, meanKeep: meanKeep, varKeep: varKeep)
    if affine {
      let (g, b) = expandedParams(for: x)
      y = y.multiplying(g).adding(b)
    }
    return (y, meanKeep.squeezed(), varKeep.squeezed())
  }

  @inlinable
  @inline(__always)
  internal func forwardInference(_ x: Tensor) -> Tensor {
    let (rm, rv) = expandedRunning(for: x)
    var y = normalize(x, meanKeep: rm, varKeep: rv)
    if affine {
      let (g, b) = expandedParams(for: x)
      y = y.multiplying(g).adding(b)
    }
    return y
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Decide phase based on thread-local context. :contentReference[oaicite:11]{index=11}
    switch Context.local.learningPhase {
    case .training:
      let (y, meanVec, varVec) = forwardTraining(x)

      let m = Tensor(self.momentum, dtype: meanVec.dtype ?? .float32, device: meanVec.device)
      let oneMinusM = Tensor(
        1 - self.momentum, dtype: meanVec.dtype ?? .float32, device: meanVec.device)

      // running := (1 - m) * running + m * batch
      runningMean.value = runningMean.value.multiplying(oneMinusM).adding(meanVec.multiplying(m))
      runningVariance.value = runningVariance.value.multiplying(oneMinusM).adding(
        varVec.multiplying(m))

      return y

    case .inference:
      return forwardInference(x)
    }
  }
}

// MARK: - Derivatives (avoid curried-self solver path)
extension BatchNorm {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector))
  {
    func primal(_ gamma: Tensor, _ beta: Tensor, _ input: Tensor) -> Tensor {
      var layer = self
      layer.gamma = gamma
      layer.beta = beta
      switch Context.local.learningPhase {
      case .training:
        return layer.forwardTraining(input).output
      case .inference:
        return layer.forwardInference(input)
      }
    }
    let (y, pb) = valueWithPullback(at: self.gamma, self.beta, x, of: primal)
    return (
      y,
      { v in
        let (dGamma, dBeta, dInput) = pb(v)
        let refGamma = withoutDerivative(at: self.gamma)
        let refBeta = withoutDerivative(at: self.beta)
        let g = BatchNorm.alignTangentComponent(dGamma, to: refGamma, label: "gamma")
        let b = BatchNorm.alignTangentComponent(dBeta, to: refBeta, label: "beta")
        return (TangentVector(gamma: g, beta: b), dInput)
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> TangentVector)
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

extension BatchNorm {
  @inline(__always)
  static func alignTangentComponent(
    _ delta: Tensor,
    to parameter: Tensor,
    label: StaticString
  ) -> Tensor {
    var adjusted = delta
    let targetParam = withoutDerivative(at: parameter)
    let targetShape = targetParam.shape
    let targetRank = targetParam.rank
    let targetCount = targetParam.count
    let targetDevice = targetParam.device
    let targetDType = targetParam.dtype

    if let dtype = targetDType, adjusted.dtype != dtype {
      adjusted = adjusted.to(dtype: dtype)
    }
    if adjusted.device != targetDevice {
      adjusted = adjusted.to(device: targetDevice)
    }

    if adjusted.shape != targetShape {
      if targetRank == 1, adjusted.rank > 1, !targetShape.isEmpty {
        let target = targetShape[0]
        if var axisToKeep = adjusted.shape.firstIndex(of: target) {
          var reduced = adjusted
          for axis in stride(from: reduced.rank - 1, through: 0, by: -1) {
            if axis == axisToKeep { continue }
            reduced = reduced.sum(dim: axis)
            if axis < axisToKeep { axisToKeep -= 1 }
          }
          adjusted = reduced
        }
      }

      let squeezed = adjusted.squeezed()
      if squeezed.shape == targetShape {
        adjusted = squeezed
      } else if squeezed.count == targetCount {
        adjusted = squeezed.reshaped(targetShape)
      } else if adjusted.count == targetCount {
        adjusted = adjusted.reshaped(targetShape)
      } else {
        preconditionFailure(
          "BatchNorm parameter \(label) has shape \(targetShape) (count \(targetCount)) "
            + "but tangent provides shape \(adjusted.shape) (count \(adjusted.count)).")
      }
    }

    return adjusted
  }
}
