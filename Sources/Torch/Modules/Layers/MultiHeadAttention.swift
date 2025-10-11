import Foundation
import _Differentiation

/// Multi-Head Attention (scaled dot-product, features last).
///
/// Shapes:
///   - query: [N, Lq, C], key: [N, Lk, C], value: [N, Lk, C], with C % numHeads == 0
///   - mask (optional, @noDerivative): broadcastable to [N, H, Lq, Lk] where `true` masks positions.
///
/// Weights are stored as [in, out] and applied as `x.matmul(W) + b`.
public struct MultiHeadAttention: Layer {
  // Hyper-parameters (not differentiable)
  @noDerivative public let embedDim: Int
  @noDerivative public let numHeads: Int
  @inlinable @noDerivative var headDim: Int { embedDim / numHeads }

  // Parameters
  public var wq: Tensor  // [C, C]
  public var wk: Tensor  // [C, C]
  public var wv: Tensor  // [C, C]
  public var wo: Tensor  // [C, C]
  public var bq: Tensor  // [C]
  public var bk: Tensor  // [C]
  public var bv: Tensor  // [C]
  public var bo: Tensor  // [C]

  // Inside `public struct MultiHeadAttention: Layer { ... }`

  public struct Input: Differentiable {
    public var query: Tensor
    public var key: Tensor
    public var value: Tensor
    @noDerivative public var mask: Tensor?  // broadcastable to [N, H, Lq, Lk]

    @differentiable(reverse)
    public init(query: Tensor, key: Tensor, value: Tensor, mask: Tensor? = nil) {
      self.query = query
      self.key = key
      self.value = value
      self.mask = mask
    }

    /// Convenience for self-attention.
    public static func selfAttention(_ x: Tensor, mask: Tensor? = nil) -> Self {
      .init(query: x, key: x, value: x, mask: mask)
    }

    // ---- Manual TangentVector (explicit witnesses; no synthesis) ----
    public struct TangentVector:
      Differentiable,
      AdditiveArithmetic,  // explicit zero / + / -
      KeyPathIterable,  // reflection-friendly
      VectorProtocol,  // scalar ops come from reflection defaults
      PointwiseMultiplicative  // elementwise ops via reflection defaults
    {
      public typealias VectorSpaceScalar = Float

      public var query: Tensor
      public var key: Tensor
      public var value: Tensor

      // Use scalar zeros to be broadcast-safe when shapes differ in AD plumbing.
      public init(
        query: Tensor = Tensor(0),
        key: Tensor = Tensor(0),
        value: Tensor = Tensor(0)
      ) {
        self.query = query
        self.key = key
        self.value = value
      }

      // AdditiveArithmetic — spelled out to avoid solver hitting synthesized witnesses.
      public static var zero: Self { .init() }
      public static func + (lhs: Self, rhs: Self) -> Self {
        .init(
          query: lhs.query + rhs.query,
          key: lhs.key + rhs.key,
          value: lhs.value + rhs.value)
      }
      public static func - (lhs: Self, rhs: Self) -> Self {
        .init(
          query: lhs.query - rhs.query,
          key: lhs.key - rhs.key,
          value: lhs.value - rhs.value)
      }

      // Tiny manual VJPs to bypass the “curried self” path entirely.
      @derivative(of: +)
      public static func _vjpAdd(_ lhs: Self, _ rhs: Self)
        -> (value: Self, pullback: (Self) -> (Self, Self))
      {
        let y = lhs + rhs
        return (y, { v in (v, v) })
      }

      @derivative(of: -)
      public static func _vjpSub(_ lhs: Self, _ rhs: Self)
        -> (value: Self, pullback: (Self) -> (Self, Self))
      {
        let y = lhs - rhs
        return (y, { v in (v, .zero - v) })
      }
    }

    // Required because we defined a manual TangentVector.
    public mutating func move(by d: TangentVector) {
      query += d.query
      key += d.key
      value += d.value
    }
  }

  public typealias Output = Tensor

  // MARK: - Manual TangentVector (avoid synthesis pitfalls)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor
    public var bq: Tensor, bk: Tensor, bv: Tensor, bo: Tensor

    public init(
      wq: Tensor = Tensor(0), wk: Tensor = Tensor(0), wv: Tensor = Tensor(0),
      wo: Tensor = Tensor(0),
      bq: Tensor = Tensor(0), bk: Tensor = Tensor(0), bv: Tensor = Tensor(0), bo: Tensor = Tensor(0)
    ) {
      self.wq = wq
      self.wk = wk
      self.wv = wv
      self.wo = wo
      self.bq = bq
      self.bk = bk
      self.bv = bv
      self.bo = bo
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        wq: lhs.wq + rhs.wq, wk: lhs.wk + rhs.wk, wv: lhs.wv + rhs.wv, wo: lhs.wo + rhs.wo,
        bq: lhs.bq + rhs.bq, bk: lhs.bk + rhs.bk, bv: lhs.bv + rhs.bv, bo: lhs.bo + rhs.bo)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        wq: lhs.wq - rhs.wq, wk: lhs.wk - rhs.wk, wv: lhs.wv - rhs.wv, wo: lhs.wo - rhs.wo,
        bq: lhs.bq - rhs.bq, bk: lhs.bk - rhs.bk, bv: lhs.bv - rhs.bv, bo: lhs.bo - rhs.bo)
    }
  }

  public mutating func move(by d: TangentVector) {
    wq += d.wq
    wk += d.wk
    wv += d.wv
    wo += d.wo
    bq += d.bq
    bk += d.bk
    bv += d.bv
    bo += d.bo
  }

  // MARK: - Init
  public init(
    embedDim: Int, numHeads: Int,
    dtype: DType = .float32, device: Device = .cpu
  ) {
    precondition(numHeads > 0 && embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
    self.embedDim = embedDim
    self.numHeads = numHeads

    let a = Foundation.sqrt(6.0 / Double(embedDim + embedDim))  // Glorot-style span
    self.wq = Tensor.uniform(
      low: -a, high: a, shape: [embedDim, embedDim], dtype: dtype, device: device)
    self.wk = Tensor.uniform(
      low: -a, high: a, shape: [embedDim, embedDim], dtype: dtype, device: device)
    self.wv = Tensor.uniform(
      low: -a, high: a, shape: [embedDim, embedDim], dtype: dtype, device: device)
    self.wo = Tensor.uniform(
      low: -a, high: a, shape: [embedDim, embedDim], dtype: dtype, device: device)
    self.bq = Tensor.zeros(shape: [embedDim], dtype: dtype, device: device)
    self.bk = Tensor.zeros(shape: [embedDim], dtype: dtype, device: device)
    self.bv = Tensor.zeros(shape: [embedDim], dtype: dtype, device: device)
    self.bo = Tensor.zeros(shape: [embedDim], dtype: dtype, device: device)
  }

  // MARK: - Helpers (shape-safe head split/combine)
  @inlinable func splitHeads(_ x: Tensor) -> Tensor {
    // [N, L, C] -> [N, H, L, Dh]
    let shape = withoutDerivative(at: x.shape)
    let N = shape[0]
    let L = shape[1]
    let C = shape[2]
    precondition(C == embedDim && C % numHeads == 0, "MHA: channel size must equal embedDim")
    let Dh = C / numHeads
    return x.reshaped([N, L, numHeads, Dh]).permuted([0, 2, 1, 3])  // permute: put H ahead of L
  }
  @inlinable func combineHeads(_ x: Tensor) -> Tensor {
    // [N, H, L, Dh] -> [N, L, C]
    let shape = withoutDerivative(at: x.shape)
    let N = shape[0]
    let H = shape[1]
    let L = shape[2]
    let Dh = shape[3]
    return x.permuted([0, 2, 1, 3]).reshaped([N, L, H * Dh])
  }

  @inlinable func softmaxLastDim(_ x: Tensor) -> Tensor {
    // numerically stable softmax over the last dimension
    let maxVals = x.max(dim: -1, keepdim: true).values  // [*, 1]
    let shifted = x - maxVals
    let expX = shifted.exp()
    let denom = expX.sum(dim: -1, keepdim: true)
    return expX.dividing(denom)
  }

  // MARK: - Forward
  @differentiable(reverse)
  public func callAsFunction(_ input: Input) -> Tensor {
    // Projections
    let q = input.query.matmul(wq).adding(bq)  // [N, Lq, C]
    let k = input.key.matmul(wk).adding(bk)  // [N, Lk, C]
    let v = input.value.matmul(wv).adding(bv)  // [N, Lk, C]

    // Heads
    let qh = splitHeads(q)  // [N, H, Lq, Dh]
    let kh = splitHeads(k)  // [N, H, Lk, Dh]
    let vh = splitHeads(v)  // [N, H, Lk, Dh]

    // Scaled dot-product attention
    let scale = 1.0 / Foundation.sqrt(Float(headDim))
    var scores = qh.matmul(kh.transposed(-2, -1)).multiplying(scale)  // [N, H, Lq, Lk]
    if let mask = input.mask {
      // Mask is broadcastable to [N, H, Lq, Lk]; true = masked
      scores = scores.maskedFill(where: mask, with: -Float.greatestFiniteMagnitude)
    }
    let attn = softmaxLastDim(scores)  // [N, H, Lq, Lk]
    let ctx = attn.matmul(vh)  // [N, H, Lq, Dh]

    // Merge heads + output projection
    let z = combineHeads(ctx)  // [N, Lq, C]
    return z.matmul(wo).adding(bo)  // [N, Lq, C]
  }
}

// MARK: - Manual derivatives (avoid “curried self” path)
extension MultiHeadAttention {
  @derivative(of: callAsFunction, wrt: (self, input))
  public func _vjpCallAsFunction(_ input: Input)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Input.TangentVector)
    )
  {
    func primal(_ s: MultiHeadAttention, _ i: Input) -> Tensor {
      let q = i.query.matmul(s.wq).adding(s.bq)
      let k = i.key.matmul(s.wk).adding(s.bk)
      let v = i.value.matmul(s.wv).adding(s.bv)

      let qh = s.splitHeads(q)
      let kh = s.splitHeads(k)
      let vh = s.splitHeads(v)

      let scale = 1.0 / Foundation.sqrt(Float(s.headDim))
      var scores = qh.matmul(kh.transposed(-2, -1)).multiplying(scale)
      if let mask = i.mask {
        scores = scores.maskedFill(where: mask, with: -Float.greatestFiniteMagnitude)
      }
      let attn = s.softmaxLastDim(scores)
      let ctx = attn.matmul(vh)
      let z = s.combineHeads(ctx)
      return z.matmul(s.wo).adding(s.bo)
    }
    let (y, pb) = valueWithPullback(at: self, input, of: primal)
    return (y, pb)
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ input: Input)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(input)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}
