import Foundation
import _Differentiation

/// Group Normalization over an arbitrary feature axis (default: last / -1).
///
/// Shapes:
///   - x:     [N, D1, D2, ..., C]   // feature axis is `axis`
///   - gamma: [C]
///   - beta:  [C]
///
/// For each sample `n` and group `g`, statistics are computed across the group's
/// channels **and all sample/spatial axes** (i.e., all axes except batch and the
/// feature axis). Affine parameters are per‑channel.
public struct GroupNorm: Layer {
  // Trainable parameters
  public var gamma: Tensor  // [C]
  public var beta: Tensor  // [C]

  // Hyper-parameters / config (not differentiable)
  @noDerivative public let groups: Int
  @noDerivative public var axis: Int
  @noDerivative public var epsilon: Float
  @noDerivative public var affine: Bool

  public typealias Input = Tensor
  public typealias Output = Tensor

  // MARK: - Manual TangentVector (avoid synthesis pitfalls)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // explicit zero/+/- witnesses
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

  // Required when we define a manual TangentVector.
  public mutating func move(by d: TangentVector) {
    gamma += d.gamma
    beta += d.beta
  }

  // MARK: - Init

  /// - Parameters:
  ///   - featureCount: length `C` along the feature axis.
  ///   - groups: number of groups (must divide `C`).
  ///   - axis: feature axis (default last / `-1`; negative allowed).
  ///   - epsilon: numerical stability constant.
  ///   - affine: if `false`, skip scale/shift.
  public init(
    featureCount: Int,
    groups: Int,
    axis: Int = -1,
    epsilon: Float = 1e-5,
    affine: Bool = true,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    precondition(groups > 0, "GroupNorm: groups must be > 0")
    precondition(featureCount % groups == 0, "GroupNorm: C must be divisible by groups")
    self.gamma = Tensor.ones(shape: [featureCount], dtype: dtype, device: device)
    self.beta = Tensor.zeros(shape: [featureCount], dtype: dtype, device: device)
    self.groups = groups
    self.axis = axis
    self.epsilon = epsilon
    self.affine = affine
  }

  // MARK: - Helpers

  /// Resolve possibly-negative `axis` for a given rank.
  @inlinable
  func _normAxis(forRank rank: Int) -> Int {
    _normalizeDimension(axis, rank: rank)  // matches your existing pattern. :contentReference[oaicite:4]{index=4}
  }

  /// Broadcast a rank-1 param `[C]` to `[1, …, C, …, 1]` at the chosen axis.
  @inlinable
  func _paramView(_ p: Tensor, like x: Tensor, atAxis a: Int) -> Tensor {
    precondition(p.rank == 1, "GroupNorm: parameter must be rank-1 [C]")
    let rank = withoutDerivative(at: x.rank)
    let featureCount = withoutDerivative(at: p.shape[0])
    var shape = [Int](repeating: 1, count: rank)
    shape[a] = featureCount
    return p.reshaped(shape)  // reshaped view, then broadcast via arithmetic. :contentReference[oaicite:5]{index=5}
  }

  /// Product of a slice of dimensions (returns at least 1).
  @inlinable
  func _product(_ dims: ArraySlice<Int>) -> Int {
    var out = 1
    for d in dims { out &*= max(d, 1) }
    return out
  }

  // MARK: - Forward (general rank)
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    precondition(x.rank >= 2, "GroupNorm expects rank ≥ 2 (at minimum [N, C])")

    // Resolve axis and validate parameter length / groups
    let a = withoutDerivative(at: _normAxis(forRank: x.rank))
    let C = withoutDerivative(at: x.shape[a])
    precondition(
      C == gamma.shape[0] && C == beta.shape[0],
      "GroupNorm: gamma/beta length must equal feature size C (\(C))")
    let G = withoutDerivative(at: groups)
    precondition(C % G == 0, "GroupNorm: C (\(C)) must be divisible by groups (\(G))")
    let groupSize = withoutDerivative(at: C / G)

    // 1) Move channels to the last axis: [d0, ..., d_{a-1}, C, d_{a+1}, ...] → [N, S, C]
    var order = Array(0..<x.rank)
    order.remove(at: a)
    order.append(a)
    let xPerm = x.permuted(order)  // :contentReference[oaicite:6]{index=6}
    let pShape = withoutDerivative(at: xPerm.shape)
    let N = withoutDerivative(at: pShape[0])
    let S = withoutDerivative(at: _product(pShape.dropFirst().dropLast()))
    let xFlat = xPerm.reshaped([N, S, C])  // :contentReference[oaicite:7]{index=7}

    // 2) Split channels into groups: [N, S, C] → [N, S, G, groupSize]
    let xGrp = xFlat.reshaped([N, S, G, groupSize])

    // 3) Per-sample, per-group mean/var across (spatial S) and (groupSize)
    // keepdim: true so dimension indices remain stable for sequential reductions. :contentReference[oaicite:8]{index=8}
    let meanS = xGrp.mean(dim: 1, keepdim: true).mean(dim: 3, keepdim: true)
    let centered = xGrp - meanS
    let varS = centered.multiplying(centered)
      .mean(dim: 1, keepdim: true).mean(dim: 3, keepdim: true)  // :contentReference[oaicite:9]{index=9}

    // 4) invStd = 1 / sqrt(var + eps) — use dtype/device-safe scalar division (LayerNorm pattern). :contentReference[oaicite:10]{index=10}
    let std = (varS.adding(Tensor(epsilon))).sqrt()  // :contentReference[oaicite:11]{index=11}
    let stdDType = withoutDerivative(at: std.dtype ?? (x.dtype ?? .float32))
    let stdDevice = withoutDerivative(at: std.device)
    let scalarOne = Tensor.ones(shape: [], dtype: stdDType, device: stdDevice)
    let invStd = scalarOne.dividing(std)
    let yGrp = centered.multiplying(invStd)

    // 5) Restore original shape: [N, S, G, groupSize] → [N, S, C] → permute-back → [original]
    let yFlat = yGrp.reshaped([N, S, C])
    let yPerm = yFlat.reshaped(pShape)
    // inverse permutation
    var inv = Array(repeating: 0, count: order.count)
    for (i, j) in order.enumerated() { inv[j] = i }
    var y = yPerm.permuted(inv)  // :contentReference[oaicite:12]{index=12}

    // 6) Affine transform per channel.
    if affine {
      let g = _paramView(gamma, like: y, atAxis: a)
      let b = _paramView(beta, like: y, atAxis: a)
      y = y.multiplying(g).adding(b)
    }
    return y
  }
}

// MARK: - Manual derivatives (bypass “curried self” code path)
extension GroupNorm {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: GroupNorm, _ i: Tensor) -> Tensor {
      precondition(i.rank >= 2, "GroupNorm expects rank ≥ 2")
      let a = withoutDerivative(at: _normalizeDimension(s.axis, rank: i.rank))

      let C = withoutDerivative(at: i.shape[a])
      precondition(
        C == s.gamma.shape[0] && C == s.beta.shape[0],
        "GroupNorm: gamma/beta length must equal C")
      let G = withoutDerivative(at: s.groups)
      precondition(C % G == 0, "GroupNorm: C must be divisible by groups")
      let groupSize = withoutDerivative(at: C / G)

      let rank = withoutDerivative(at: i.rank)
      var order = Array(0..<rank)
      order.remove(at: a)
      order.append(a)
      let xPerm = i.permuted(order)  // :contentReference[oaicite:13]{index=13}
      let pShape = withoutDerivative(at: xPerm.shape)
      let N = withoutDerivative(at: pShape[0])
      let S = withoutDerivative(
        at: (pShape.count > 2) ? pShape[1..<(pShape.count - 1)].reduce(1, *) : 1)
      let xFlat = xPerm.reshaped([N, S, C])  // :contentReference[oaicite:14]{index=14}
      let xGrp = xFlat.reshaped([N, S, G, groupSize])

      let meanS = xGrp.mean(dim: 1, keepdim: true).mean(dim: 3, keepdim: true)
      let centered = xGrp - meanS
      let varS = centered.multiplying(centered)
        .mean(dim: 1, keepdim: true).mean(dim: 3, keepdim: true)

      let std = (varS.adding(Tensor(withoutDerivative(at: s.epsilon)))).sqrt()
      let stdDType = withoutDerivative(at: std.dtype ?? (i.dtype ?? .float32))
      let stdDevice = withoutDerivative(at: std.device)
      let scalarOne = Tensor.ones(shape: [], dtype: stdDType, device: stdDevice)
      let invStd = scalarOne.dividing(std)
      let yGrp = centered.multiplying(invStd)

      let yFlat = yGrp.reshaped([N, S, C])
      let yPerm = yFlat.reshaped(pShape)
      var inv = Array(repeating: 0, count: order.count)
      for (idx, j) in order.enumerated() { inv[j] = idx }
      var y = yPerm.permuted(inv)  // :contentReference[oaicite:15]{index=15}

      if s.affine {
        let g = s._paramView(s.gamma, like: y, atAxis: a)
        let b = s._paramView(s.beta, like: y, atAxis: a)
        y = y.multiplying(g).adding(b)
      }
      return y
    }

    // Free-function VJP to avoid “curried self” member reference path.
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
