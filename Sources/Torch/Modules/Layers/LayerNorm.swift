import Foundation
import _Differentiation

/// Layer Normalization along an arbitrary feature axis (default: last / -1).
///
/// Shapes:
///   - x:     [N, D1, D2, ..., C]   // feature axis is `axis`
///   - gamma: [C]
///   - beta:  [C]
///
/// Unlike BatchNorm, LayerNorm is context-agnostic (no running stats).
public struct LayerNorm: Layer {
  // Trainable parameters
  public var gamma: Tensor  // [C]
  public var beta: Tensor  // [C]

  // Hyper-parameters
  @noDerivative public var epsilon: Float
  @noDerivative public var axis: Int
  @noDerivative public var affine: Bool

  public typealias Input = Tensor
  public typealias Output = Tensor

  // MARK: - Manual TangentVector (avoid synthesis pitfalls)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // explicit witnesses
    KeyPathIterable,  // required by Module
    VectorProtocol,  // required by Module
    PointwiseMultiplicative  // required by Module
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
      .init(gamma: lhs.gamma + rhs.gamma, beta: lhs.beta + rhs.beta)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(gamma: lhs.gamma - rhs.gamma, beta: lhs.beta - rhs.beta)
    }
  }

  public mutating func move(by d: TangentVector) {
    gamma += d.gamma
    beta += d.beta
  }

  // MARK: - Init

  /// - Parameters:
  ///   - featureCount: length `C` of the feature axis.
  ///   - axis: feature axis (negative allowed, default `-1`).
  ///   - epsilon: numerical stability constant.
  ///   - affine: if `false`, skip scale/shift.
  public init(
    featureCount: Int,
    axis: Int = -1,
    epsilon: Float = 1e-5,
    affine: Bool = true,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.gamma = Tensor.ones(shape: [featureCount], dtype: dtype, device: device)
    self.beta = Tensor.zeros(shape: [featureCount], dtype: dtype, device: device)
    self.axis = axis
    self.epsilon = epsilon
    self.affine = affine
  }

  // MARK: - Helpers

  /// Resolve possibly-negative `axis` for a given rank (mirrors your pattern elsewhere).
  @inlinable
  func _normAxis(forRank rank: Int) -> Int {
    _normalizeDimension(axis, rank: rank)  // negative axes allowed
  }

  /// Reshape a rank-1 parameter `[C]` to a broadcastable view `[1, ..., C, ..., 1]`.
  @inlinable
  internal func _paramView(_ p: Tensor, like x: Tensor, atAxis a: Int) -> Tensor {
    precondition(p.rank == 1, "LayerNorm params must be rank‑1 (length = featureCount)")
    let rank = withoutDerivative(at: x.rank)
    let featureCount = withoutDerivative(at: x.shape[a])
    precondition(
      p.shape == [featureCount],
      "Param length \(p.shape[0]) must equal feature count \(featureCount) on axis \(a)")
    var target = [Int](repeating: 1, count: rank)
    target[a] = featureCount
    // Broadcasting gives the right forward semantics and its pullback reduces back to [C].
    return p.broadcasted(to: target)
  }

  // MARK: - Forward (general rank)
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    precondition(x.rank >= 1, "LayerNorm expects rank ≥ 1")

    // Resolve axis and validate parameter length
    let a = withoutDerivative(at: _normAxis(forRank: x.rank))
    precondition(
      gamma.shape[0] == x.shape[a],
      "LayerNorm: gamma length (\(gamma.shape[0])) must equal feature size (\(x.shape[a]))")
    precondition(
      beta.shape[0] == x.shape[a],
      "LayerNorm: beta length (\(beta.shape[0])) must equal feature size (\(x.shape[a]))")

    // Per-sample mean/var over the feature axis (keep reduced dim for broadcasting).
    let mean = x.mean(dim: a, keepdim: true)  // [N, …, 1]
    let centered = x - mean  // [N, …, C]
    let var_ = centered.multiplying(centered).mean(dim: a, keepdim: true)  // [N, …, 1]

    // invStd = 1 / sqrt(var + eps) using your dtype/device-correct scalar division pattern.
    let std = (var_.adding(Tensor(epsilon))).sqrt()
    let stdDType = withoutDerivative(at: std.dtype ?? (x.dtype ?? .float32))
    let stdDevice = withoutDerivative(at: std.device)
    let scalarOne = Tensor.ones(shape: [], dtype: stdDType, device: stdDevice)
    let invStd = scalarOne.dividing(std)  // [N, …, 1]
    let yNorm = centered.multiplying(invStd)  // [N, …, C]

    if !affine { return yNorm }

    // Broadcast gamma/beta to the full shape by reshaping to [1, …, C, …, 1].
    let g = _paramView(gamma, like: x, atAxis: a)
    let b = _paramView(beta, like: x, atAxis: a)
    return yNorm.multiplying(g).adding(b)
  }
}

// MARK: - Manual derivatives (avoid curried-member code path)
extension LayerNorm {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: LayerNorm, _ i: Tensor) -> Tensor {
      precondition(i.rank >= 1, "LayerNorm expects rank ≥ 1")
      let a = withoutDerivative(at: _normalizeDimension(s.axis, rank: i.rank))

      let mean = i.mean(dim: a, keepdim: true)
      let centered = i - mean
      let var_ = centered.multiplying(centered).mean(dim: a, keepdim: true)

      let std = (var_.adding(Tensor(s.epsilon))).sqrt()
      let stdDType = withoutDerivative(at: std.dtype ?? (i.dtype ?? .float32))
      let stdDevice = withoutDerivative(at: std.device)
      let scalarOne = Tensor.ones(shape: [], dtype: stdDType, device: stdDevice)
      let invStd = scalarOne.dividing(std)
      let yNorm = centered.multiplying(invStd)

      if !s.affine { return yNorm }
      let g = s._paramView(s.gamma, like: i, atAxis: a)
      let b = s._paramView(s.beta, like: i, atAxis: a)
      return yNorm.multiplying(g).adding(b)
    }

    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (y, pb)
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
