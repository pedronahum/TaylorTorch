import Foundation
import _Differentiation

/// Multi-head self-attention mechanism for transformers.
///
/// `MultiHeadAttention` is the core building block of transformer architectures, enabling the model
/// to attend to different parts of the input sequence in parallel. It's essential for BERT, GPT,
/// Vision Transformers, and all modern attention-based models.
///
/// ## Overview
///
/// Multi-head attention extends the standard attention mechanism by running multiple attention
/// operations in parallel ("heads"), each learning to focus on different aspects of the input.
/// The outputs are then concatenated and linearly transformed.
///
/// ### Operation
///
/// For each attention head independently:
/// ```
/// Q, K, V = query*Wq, key*Wk, value*Wv  // Project inputs
/// scores = (Q * K^T) / sqrt(d_head)      // Compute attention scores
/// attention = softmax(scores)             // Normalize scores
/// output = attention * V                  // Weighted sum of values
/// ```
///
/// Then concatenate all heads and project:
/// ```
/// concatenated = concat(head_1, head_2, ..., head_H)
/// output = concatenated * Wo + bo
/// ```
///
/// ## Creating MultiHeadAttention
///
/// ```swift
/// // Standard transformer attention (BERT/GPT style)
/// let mha = MultiHeadAttention(embedDim: 512, numHeads: 8)
///
/// // BERT-Base configuration
/// let bertAttn = MultiHeadAttention(embedDim: 768, numHeads: 12)
///
/// // GPT-2 configuration
/// let gptAttn = MultiHeadAttention(embedDim: 768, numHeads: 12)
///
/// // Vision Transformer
/// let vitAttn = MultiHeadAttention(embedDim: 768, numHeads: 12)
/// ```
///
/// ## Usage Examples
///
/// ### Self-Attention (most common)
///
/// ```swift
/// let mha = MultiHeadAttention(embedDim: 512, numHeads: 8)
/// let tokens = Tensor.randn([32, 100, 512])  // [batch, seqLen, embedDim]
///
/// // Self-attention: query, key, value all come from same input
/// let input = MultiHeadAttention.Input.selfAttention(tokens)
/// let output = mha(input)  // [32, 100, 512]
/// ```
///
/// ### Cross-Attention (encoder-decoder)
///
/// ```swift
/// let mha = MultiHeadAttention(embedDim: 512, numHeads: 8)
///
/// let decoderTokens = Tensor.randn([32, 50, 512])   // Query from decoder
/// let encoderOutput = Tensor.randn([32, 100, 512])  // Key/Value from encoder
///
/// // Cross-attention: query from decoder, key/value from encoder
/// let input = MultiHeadAttention.Input(
///     query: decoderTokens,
///     key: encoderOutput,
///     value: encoderOutput
/// )
/// let output = mha(input)  // [32, 50, 512]
/// ```
///
/// ### With Attention Mask (causal/padding)
///
/// ```swift
/// let mha = MultiHeadAttention(embedDim: 512, numHeads: 8)
/// let tokens = Tensor.randn([32, 100, 512])
///
/// // Causal mask for autoregressive models (GPT-style)
/// let causalMask = createCausalMask(sequenceLength: 100)  // Upper triangular
///
/// let input = MultiHeadAttention.Input.selfAttention(tokens, mask: causalMask)
/// let output = mha(input)  // Each position can only attend to past
/// ```
///
/// ## Shape Specifications
///
/// ### Inputs
/// - **Query**: `[batch, queryLength, embedDim]`
/// - **Key**: `[batch, keyLength, embedDim]`
/// - **Value**: `[batch, keyLength, embedDim]`
/// - **Mask** (optional): `[batch, numHeads, queryLength, keyLength]` or broadcastable
///
/// ### Output
/// - `[batch, queryLength, embedDim]`
///
/// ### Internal Shapes
/// After splitting into heads:
/// - Q, K, V: `[batch, numHeads, seqLength, headDim]` where `headDim = embedDim / numHeads`
/// - Attention scores: `[batch, numHeads, queryLength, keyLength]`
///
/// ```swift
/// let mha = MultiHeadAttention(embedDim: 512, numHeads: 8)  // headDim = 64
///
/// let query = Tensor.randn([32, 50, 512])   // [batch, queryLen, embedDim]
/// let key = Tensor.randn([32, 100, 512])    // [batch, keyLen, embedDim]
/// let value = Tensor.randn([32, 100, 512])  // [batch, keyLen, embedDim]
///
/// let input = MultiHeadAttention.Input(query: query, key: key, value: value)
/// let output = mha(input)  // [32, 50, 512] - query length preserved
/// ```
///
/// ## Attention Mechanisms
///
/// ### Self-Attention
/// All three inputs (Q, K, V) come from the same sequence:
///
/// ```swift
/// // BERT-style bidirectional self-attention
/// let input = MultiHeadAttention.Input.selfAttention(embeddings)
/// let output = attention(input)
/// ```
///
/// Each token can attend to all other tokens in the sequence.
///
/// ### Masked Self-Attention (Causal)
/// For autoregressive models (GPT), prevent attending to future positions:
///
/// ```swift
/// // Create causal mask (upper triangular = masked)
/// func createCausalMask(sequenceLength: Int) -> Tensor {
///     // Returns shape [1, 1, seqLen, seqLen] with upper triangle = true
///     // ... implementation ...
/// }
///
/// let mask = createCausalMask(sequenceLength: seqLen)
/// let input = MultiHeadAttention.Input.selfAttention(tokens, mask: mask)
/// let output = mha(input)  // Each position only sees past
/// ```
///
/// ### Cross-Attention
/// Query from one sequence, Key/Value from another (encoder-decoder):
///
/// ```swift
/// // Decoder attending to encoder
/// let input = MultiHeadAttention.Input(
///     query: decoderState,    // What we're computing
///     key: encoderOutput,      // What we're attending to
///     value: encoderOutput     // What we're retrieving
/// )
/// let output = mha(input)
/// ```
///
/// ## In Transformer Architectures
///
/// ### Transformer Encoder Layer
///
/// ```swift
/// struct TransformerEncoderLayer: Layer {
///     var selfAttention: MultiHeadAttention
///     var norm1: LayerNorm
///     var feedForward: Sequential<...>
///     var norm2: LayerNorm
///
///     init(modelDim: Int, numHeads: Int) {
///         selfAttention = MultiHeadAttention(embedDim: modelDim, numHeads: numHeads)
///         norm1 = LayerNorm(featureCount: modelDim)
///         feedForward = Sequential {
///             Linear(inputSize: modelDim, outputSize: modelDim * 4)
///             GELU()
///             Linear(inputSize: modelDim * 4, outputSize: modelDim)
///         }
///         norm2 = LayerNorm(featureCount: modelDim)
///     }
///
///     @differentiable
///     func callAsFunction(_ x: Tensor, mask: Tensor? = nil) -> Tensor {
///         // Pre-norm style (modern)
///         var h = x
///         let attnInput = MultiHeadAttention.Input.selfAttention(norm1(h), mask: mask)
///         h = h + selfAttention(attnInput)
///         h = h + feedForward(norm2(h))
///         return h
///     }
/// }
/// ```
///
/// ### Transformer Decoder Layer
///
/// ```swift
/// struct TransformerDecoderLayer: Layer {
///     var selfAttention: MultiHeadAttention
///     var crossAttention: MultiHeadAttention
///     var feedForward: Sequential<...>
///     var norm1: LayerNorm
///     var norm2: LayerNorm
///     var norm3: LayerNorm
///
///     @differentiable
///     func callAsFunction(
///         _ x: Tensor,
///         encoderOutput: Tensor,
///         srcMask: Tensor?,
///         tgtMask: Tensor?
///     ) -> Tensor {
///         var h = x
///
///         // Masked self-attention
///         let selfAttnInput = MultiHeadAttention.Input.selfAttention(norm1(h), mask: tgtMask)
///         h = h + selfAttention(selfAttnInput)
///
///         // Cross-attention to encoder
///         let crossAttnInput = MultiHeadAttention.Input(
///             query: norm2(h),
///             key: encoderOutput,
///             value: encoderOutput,
///             mask: srcMask
///         )
///         h = h + crossAttention(crossAttnInput)
///
///         // Feed-forward
///         h = h + feedForward(norm3(h))
///         return h
///     }
/// }
/// ```
///
/// ## Attention Masks
///
/// Masks control which positions can attend to which. `true` values are masked (prevented from attending).
///
/// ### Padding Mask
/// Ignore padding tokens:
///
/// ```swift
/// // Shape: [batch, 1, 1, seqLen]
/// // true where tokens are padding
/// ```
///
/// ### Causal Mask
/// Prevent attending to future positions (GPT-style):
///
/// ```swift
/// // Shape: [1, 1, seqLen, seqLen]
/// // Upper triangle is true (future positions masked)
/// ```
///
/// ### Combined Masks
/// Combine padding and causal masks with logical OR.
///
/// ## Multi-Head Benefits
///
/// Multiple heads allow the model to:
/// 1. **Attend to different positions**: Each head can focus on different parts of the sequence
/// 2. **Learn different relationships**: Syntactic, semantic, positional patterns
/// 3. **Increase capacity**: More parameters without increasing sequence length
/// 4. **Stabilize training**: Redundancy helps with gradient flow
///
/// ## Parameters
///
/// For `embedDim=512, numHeads=8`:
/// - Query projection: 512 × 512 + 512 = 262,656
/// - Key projection: 512 × 512 + 512 = 262,656
/// - Value projection: 512 × 512 + 512 = 262,656
/// - Output projection: 512 × 512 + 512 = 262,656
/// - **Total**: ~1M parameters
///
/// ## Common Configurations
///
/// | Model | embedDim | numHeads | headDim | Layers |
/// |-------|----------|----------|---------|--------|
/// | BERT-Base | 768 | 12 | 64 | 12 |
/// | BERT-Large | 1024 | 16 | 64 | 24 |
/// | GPT-2 Small | 768 | 12 | 64 | 12 |
/// | GPT-2 Medium | 1024 | 16 | 64 | 24 |
/// | GPT-2 Large | 1280 | 20 | 64 | 36 |
/// | ViT-Base | 768 | 12 | 64 | 12 |
///
/// ## Automatic Differentiation
///
/// MultiHeadAttention is fully differentiable:
///
/// ```swift
/// let mha = MultiHeadAttention(embedDim: 512, numHeads: 8)
/// let tokens = Tensor.randn([32, 100, 512])
/// let input = MultiHeadAttention.Input.selfAttention(tokens)
///
/// let (output, pullback) = valueWithPullback(at: mha, input) { layer, inp in
///     layer(inp)
/// }
///
/// let gradOutput = Tensor.ones([32, 100, 512])
/// let (mhaGrad, inputGrad) = pullback(gradOutput)
/// // mhaGrad contains gradients for all projection matrices
/// // inputGrad contains gradients for query, key, value
/// ```
///
/// ## Performance Considerations
///
/// - **Complexity**: O(L² × D) where L is sequence length, D is embedding dimension
/// - **Memory**: Attention scores require O(L²) memory per head
/// - **Optimization**: Use efficient attention implementations for long sequences
/// - **Batching**: Larger batches improve GPU utilization
/// - **Sequence Length**: Quadratic scaling makes very long sequences expensive
///
/// ## Topics
///
/// ### Creating MultiHeadAttention
///
/// - ``init(embedDim:numHeads:dtype:device:)``
///
/// ### Input Structure
///
/// - ``Input``
/// - ``Input/init(query:key:value:mask:)``
/// - ``Input/selfAttention(_:mask:)``
///
/// ### Properties
///
/// - ``embedDim``
/// - ``numHeads``
/// - ``headDim``
/// - ``wq`` - Query projection weight
/// - ``wk`` - Key projection weight
/// - ``wv`` - Value projection weight
/// - ``wo`` - Output projection weight
/// - ``bq`` - Query projection bias
/// - ``bk`` - Key projection bias
/// - ``bv`` - Value projection bias
/// - ``bo`` - Output projection bias
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``LayerNorm`` - Essential for transformer architectures
/// - ``Linear`` - Feed-forward layers in transformers
/// - ``GELU`` - Common activation in transformers
/// - ``Sequential`` - Compose transformer components
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
