import Foundation
import _Differentiation

// MARK: - Feed-Forward block (Linear → ReLU → Dropout → Linear)
public struct FeedForward: Layer {
  public var fc1: Linear
  public var fc2: Linear
  public var dropout: Dropout

  public typealias Input = Tensor
  public typealias Output = Tensor

  public init(dModel: Int, hidden: Int, dropout: Float = 0.1) {
    self.fc1 = Linear(inputSize: dModel, outputSize: hidden)
    self.fc2 = Linear(inputSize: hidden, outputSize: dModel)
    self.dropout = Dropout(probability: dropout)
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let a = ReLU()(fc1(x))
    return fc2(dropout(a))
  }
}

/// A standard Transformer encoder block (pre/post-norm; residuals inside).
///
/// Input/Output shape: [N, L, C] where C == `embedDim`.
public struct TransformerEncoderLayer: Layer {
  // MARK: Hyperparameters (non-differentiable)
  @noDerivative public let embedDim: Int
  @noDerivative public let ffHiddenDim: Int
  @noDerivative public let numHeads: Int

  // MARK: Sub-layers / parameters
  public var selfAttn: MultiHeadAttention
  public var norm1: LayerNorm
  public var norm2: LayerNorm
  public var ff1: Linear
  public var ff2: Linear
  @noDerivative public var activation: GELU  // parameterless activation layer

  // MARK: I/O
  public struct Input: Differentiable {
    /// Token embeddings: [N, L, C]
    public var x: Tensor
    /// Optional attention mask, broadcastable to [N, H, L, L]. (true = masked)
    @noDerivative public var attnMask: Tensor?

    @differentiable(reverse)
    public init(x: Tensor, attnMask: Tensor? = nil) {
      self.x = x
      self.attnMask = attnMask
    }

    // ---- Manual TangentVector: explicit zero/+/- to avoid synthesis pitfalls.
    public struct TangentVector:
      Differentiable,
      AdditiveArithmetic,
      KeyPathIterable,
      VectorProtocol,
      PointwiseMultiplicative
    {
      public typealias VectorSpaceScalar = Float

      public var x: Tensor

      public init(x: Tensor = Tensor(0)) { self.x = x }

      public static var zero: Self { .init() }
      public static func + (lhs: Self, rhs: Self) -> Self { .init(x: lhs.x + rhs.x) }
      public static func - (lhs: Self, rhs: Self) -> Self { .init(x: lhs.x - rhs.x) }

      // Optional: explicit VJPs for +/-, mirroring what worked well for MHA.Input.
    }

    public mutating func move(by direction: TangentVector) {
      x += direction.x
    }
  }

  public typealias Output = Tensor  // [N, L, C]

  // MARK: Layer TangentVector (explicit, like your other layers)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var selfAttn: MultiHeadAttention.TangentVector
    public var norm1: LayerNorm.TangentVector
    public var norm2: LayerNorm.TangentVector
    public var ff1: Linear.TangentVector
    public var ff2: Linear.TangentVector

    public init(
      selfAttn: MultiHeadAttention.TangentVector = .zero,
      norm1: LayerNorm.TangentVector = .zero,
      norm2: LayerNorm.TangentVector = .zero,
      ff1: Linear.TangentVector = .zero,
      ff2: Linear.TangentVector = .zero
    ) {
      self.selfAttn = selfAttn
      self.norm1 = norm1
      self.norm2 = norm2
      self.ff1 = ff1
      self.ff2 = ff2
    }

    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        selfAttn: lhs.selfAttn + rhs.selfAttn,
        norm1: lhs.norm1 + rhs.norm1,
        norm2: lhs.norm2 + rhs.norm2,
        ff1: lhs.ff1 + rhs.ff1,
        ff2: lhs.ff2 + rhs.ff2
      )
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        selfAttn: lhs.selfAttn - rhs.selfAttn,
        norm1: lhs.norm1 - rhs.norm1,
        norm2: lhs.norm2 - rhs.norm2,
        ff1: lhs.ff1 - rhs.ff1,
        ff2: lhs.ff2 - rhs.ff2
      )
    }
  }

  public mutating func move(by d: TangentVector) {
    selfAttn.move(by: d.selfAttn)
    norm1.move(by: d.norm1)
    norm2.move(by: d.norm2)
    ff1.move(by: d.ff1)
    ff2.move(by: d.ff2)
  }

  // MARK: Init
  public init(
    embedDim: Int,
    numHeads: Int,
    ffHiddenDim: Int,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.embedDim = embedDim
    self.ffHiddenDim = ffHiddenDim
    self.numHeads = numHeads

    self.selfAttn = MultiHeadAttention(
      embedDim: embedDim, numHeads: numHeads, dtype: dtype, device: device)
    self.norm1 = LayerNorm(
      featureCount: embedDim, epsilon: 1e-5, affine: true, dtype: dtype, device: device)
    self.norm2 = LayerNorm(
      featureCount: embedDim, epsilon: 1e-5, affine: true, dtype: dtype, device: device)
    self.ff1 = Linear(inputSize: embedDim, outputSize: ffHiddenDim, dtype: dtype, device: device)
    self.ff2 = Linear(inputSize: ffHiddenDim, outputSize: embedDim, dtype: dtype, device: device)
    self.activation = GELU(approximate: true)  // parameterless activation wrapper
  }

  // MARK: Forward (post-norm: Add -> Norm)
  @differentiable(reverse)
  public func callAsFunction(_ input: Input) -> Tensor {
    let x = input.x  // [N, L, C]
    let attnOut = selfAttn(.init(query: x, key: x, value: x, mask: input.attnMask))  // [N, L, C]
    let y1 = norm1(x + attnOut)  // residual 1 + norm
    let ff = ff2(activation(ff1(y1)))  // MLP
    let out = norm2(y1 + ff)  // residual 2 + norm
    return out  // [N, L, C]
  }
}

// MARK: - Manual derivatives (avoid “curried self” path)
extension TransformerEncoderLayer {
  @derivative(of: callAsFunction, wrt: (self, input))
  public func _vjpCallAsFunction(_ input: Input)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> (TangentVector, Input.TangentVector))
  {
    func primal(_ s: TransformerEncoderLayer, _ i: Input) -> Tensor {
      let x = i.x
      let attnOut = s.selfAttn(.init(query: x, key: x, value: x, mask: i.attnMask))
      let y1 = s.norm1(x + attnOut)
      let ff = s.ff2(s.activation(s.ff1(y1)))
      let out = s.norm2(y1 + ff)
      return out
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

// MARK: - Decoder block (masked self-attn, cross-attn, FFN; post-norm)
public struct TransformerDecoderLayer: Layer {
  // Hyperparams
  @noDerivative public let embedDim: Int
  @noDerivative public let ffHiddenDim: Int
  @noDerivative public let numHeads: Int

  // Sub-layers
  public var selfAttn: MultiHeadAttention
  public var crossAttn: MultiHeadAttention
  public var norm1: LayerNorm
  public var norm2: LayerNorm
  public var norm3: LayerNorm
  public var ff1: Linear
  public var ff2: Linear
  @noDerivative public var activation: GELU

  // I/O
  public struct Input: Differentiable {
    /// Decoder stream [N, Lt, C]
    public var x: Tensor
    /// Encoder memory [N, Ls, C]
    public var memory: Tensor
    /// Optional masks (broadcastable to [N, H, Lt, Lt] and [N, H, Lt, Ls]). true = masked.
    @noDerivative public var selfMask: Tensor?
    @noDerivative public var crossMask: Tensor?

    @differentiable(reverse)
    public init(x: Tensor, memory: Tensor, selfMask: Tensor? = nil, crossMask: Tensor? = nil) {
      self.x = x
      self.memory = memory
      self.selfMask = selfMask
      self.crossMask = crossMask
    }

    // Manual tangent vector to avoid synthesis corner cases.
    public struct TangentVector:
      Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
    {
      public typealias VectorSpaceScalar = Float
      public var x: Tensor
      public var memory: Tensor
      public init(x: Tensor = Tensor(0), memory: Tensor = Tensor(0)) {
        self.x = x
        self.memory = memory
      }
      public static var zero: Self { .init() }
      public static func + (lhs: Self, rhs: Self) -> Self {
        .init(x: lhs.x + rhs.x, memory: lhs.memory + rhs.memory)
      }
      public static func - (lhs: Self, rhs: Self) -> Self {
        .init(x: lhs.x - rhs.x, memory: lhs.memory - rhs.memory)
      }

    }

    public mutating func move(by d: TangentVector) {
      x += d.x
      memory += d.memory
    }
  }

  public typealias Output = Tensor

  // Layer tangent
  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var selfAttn: MultiHeadAttention.TangentVector
    public var crossAttn: MultiHeadAttention.TangentVector
    public var norm1: LayerNorm.TangentVector
    public var norm2: LayerNorm.TangentVector
    public var norm3: LayerNorm.TangentVector
    public var ff1: Linear.TangentVector
    public var ff2: Linear.TangentVector

    public init(
      selfAttn: MultiHeadAttention.TangentVector = .zero,
      crossAttn: MultiHeadAttention.TangentVector = .zero,
      norm1: LayerNorm.TangentVector = .zero,
      norm2: LayerNorm.TangentVector = .zero,
      norm3: LayerNorm.TangentVector = .zero,
      ff1: Linear.TangentVector = .zero,
      ff2: Linear.TangentVector = .zero
    ) {
      self.selfAttn = selfAttn
      self.crossAttn = crossAttn
      self.norm1 = norm1
      self.norm2 = norm2
      self.norm3 = norm3
      self.ff1 = ff1
      self.ff2 = ff2
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        selfAttn: lhs.selfAttn + rhs.selfAttn,
        crossAttn: lhs.crossAttn + rhs.crossAttn,
        norm1: lhs.norm1 + rhs.norm1, norm2: lhs.norm2 + rhs.norm2, norm3: lhs.norm3 + rhs.norm3,
        ff1: lhs.ff1 + rhs.ff1, ff2: lhs.ff2 + rhs.ff2
      )
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        selfAttn: lhs.selfAttn - rhs.selfAttn,
        crossAttn: lhs.crossAttn - rhs.crossAttn,
        norm1: lhs.norm1 - rhs.norm1, norm2: lhs.norm2 - rhs.norm2, norm3: lhs.norm3 - rhs.norm3,
        ff1: lhs.ff1 - rhs.ff1, ff2: lhs.ff2 - rhs.ff2
      )
    }
  }

  public mutating func move(by d: TangentVector) {
    selfAttn.move(by: d.selfAttn)
    crossAttn.move(by: d.crossAttn)
    norm1.move(by: d.norm1)
    norm2.move(by: d.norm2)
    norm3.move(by: d.norm3)
    ff1.move(by: d.ff1)
    ff2.move(by: d.ff2)
  }

  public init(
    embedDim: Int, numHeads: Int, ffHiddenDim: Int,
    dtype: DType = .float32, device: Device = .cpu
  ) {
    self.embedDim = embedDim
    self.ffHiddenDim = ffHiddenDim
    self.numHeads = numHeads
    self.selfAttn = MultiHeadAttention(
      embedDim: embedDim, numHeads: numHeads, dtype: dtype, device: device)
    self.crossAttn = MultiHeadAttention(
      embedDim: embedDim, numHeads: numHeads, dtype: dtype, device: device)
    self.norm1 = LayerNorm(
      featureCount: embedDim, epsilon: 1e-5, affine: true, dtype: dtype, device: device)
    self.norm2 = LayerNorm(
      featureCount: embedDim, epsilon: 1e-5, affine: true, dtype: dtype, device: device)
    self.norm3 = LayerNorm(
      featureCount: embedDim, epsilon: 1e-5, affine: true, dtype: dtype, device: device)
    self.ff1 = Linear(inputSize: embedDim, outputSize: ffHiddenDim, dtype: dtype, device: device)
    self.ff2 = Linear(inputSize: ffHiddenDim, outputSize: embedDim, dtype: dtype, device: device)
    self.activation = GELU(approximate: true)
  }

  @differentiable(reverse)
  public func callAsFunction(_ input: Input) -> Tensor {
    let x = input.x
    // 1) masked decoder self-attention
    let a1 = selfAttn(.init(query: x, key: x, value: x, mask: input.selfMask))
    let y1 = norm1(x + a1)
    // 2) encoder–decoder cross attention
    let a2 = crossAttn(
      .init(query: y1, key: input.memory, value: input.memory, mask: input.crossMask))
    let y2 = norm2(y1 + a2)
    // 3) FFN
    let f = ff2(activation(ff1(y2)))
    return norm3(y2 + f)
  }
}

// Manual derivatives to avoid the “curried self” path
extension TransformerDecoderLayer {
  @derivative(of: callAsFunction, wrt: (self, input))
  public func _vjpCallAsFunction(_ input: Input)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> (TangentVector, Input.TangentVector))
  {
    func primal(_ s: TransformerDecoderLayer, _ i: Input) -> Tensor {
      let a1 = s.selfAttn(.init(query: i.x, key: i.x, value: i.x, mask: i.selfMask))
      let y1 = s.norm1(i.x + a1)
      let a2 = s.crossAttn(.init(query: y1, key: i.memory, value: i.memory, mask: i.crossMask))
      let y2 = s.norm2(y1 + a2)
      let f = s.ff2(s.activation(s.ff1(y2)))
      return s.norm3(y2 + f)
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

// MARK: - Tiny encoder–decoder wrapper
public struct TinyTransformer: Layer {
  // Embeddings + positional enc
  public var srcEmb: Embedding
  public var tgtEmb: Embedding
  @noDerivative public var posEnc: PositionalEncoding

  // Two-layer encoder, one-layer decoder (tiny!)
  public var enc1: TransformerEncoderLayer
  public var enc2: TransformerEncoderLayer
  public var dec1: TransformerDecoderLayer

  // Final projection to vocab
  public var outProj: Linear

  // I/O: src indices + teacher-forced target inputs
  public struct Input: Differentiable {
    public var src: Tensor  // [N, Ls] int64
    public var tgtIn: Tensor  // [N, Lt] int64 (BOS + tokens)
    @noDerivative public var srcMask: Tensor?  // [N,1,1,Ls], true=masked
    @noDerivative public var tgtMask: Tensor?  // [N,1,Lt,Lt], true=masked (pad OR future)

    @differentiable(reverse)
    public init(src: Tensor, tgtIn: Tensor, srcMask: Tensor? = nil, tgtMask: Tensor? = nil) {
      self.src = src
      self.tgtIn = tgtIn
      self.srcMask = srcMask
      self.tgtMask = tgtMask
    }

    public struct TangentVector:
      Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
    {
      public typealias VectorSpaceScalar = Float
      public var src: Tensor
      public var tgtIn: Tensor
      public init(src: Tensor = Tensor(0), tgtIn: Tensor = Tensor(0)) {
        self.src = src
        self.tgtIn = tgtIn
      }
      public static var zero: Self { .init() }
      public static func + (lhs: Self, rhs: Self) -> Self {
        .init(src: lhs.src + rhs.src, tgtIn: lhs.tgtIn + rhs.tgtIn)
      }
      public static func - (lhs: Self, rhs: Self) -> Self {
        .init(src: lhs.src - rhs.src, tgtIn: lhs.tgtIn - rhs.tgtIn)
      }
    }

    public mutating func move(by d: TangentVector) {
      src += d.src
      tgtIn += d.tgtIn
    }
  }

  public typealias Output = Tensor  // logits [N, Lt, V]

  // Layer tangent
  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var srcEmb: Embedding.TangentVector
    public var tgtEmb: Embedding.TangentVector
    public var enc1: TransformerEncoderLayer.TangentVector
    public var enc2: TransformerEncoderLayer.TangentVector
    public var dec1: TransformerDecoderLayer.TangentVector
    public var outProj: Linear.TangentVector

    public init(
      srcEmb: Embedding.TangentVector = .zero,
      tgtEmb: Embedding.TangentVector = .zero,
      enc1: TransformerEncoderLayer.TangentVector = .zero,
      enc2: TransformerEncoderLayer.TangentVector = .zero,
      dec1: TransformerDecoderLayer.TangentVector = .zero,
      outProj: Linear.TangentVector = .zero
    ) {
      self.srcEmb = srcEmb
      self.tgtEmb = tgtEmb
      self.enc1 = enc1
      self.enc2 = enc2
      self.dec1 = dec1
      self.outProj = outProj
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        srcEmb: lhs.srcEmb + rhs.srcEmb,
        tgtEmb: lhs.tgtEmb + rhs.tgtEmb,
        enc1: lhs.enc1 + rhs.enc1, enc2: lhs.enc2 + rhs.enc2,
        dec1: lhs.dec1 + rhs.dec1,
        outProj: lhs.outProj + rhs.outProj
      )
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        srcEmb: lhs.srcEmb - rhs.srcEmb,
        tgtEmb: lhs.tgtEmb - rhs.tgtEmb,
        enc1: lhs.enc1 - rhs.enc1, enc2: lhs.enc2 - rhs.enc2,
        dec1: lhs.dec1 - rhs.dec1,
        outProj: lhs.outProj - rhs.outProj
      )
    }
  }

  public mutating func move(by d: TangentVector) {
    srcEmb.move(by: d.srcEmb)
    tgtEmb.move(by: d.tgtEmb)
    enc1.move(by: d.enc1)
    enc2.move(by: d.enc2)
    dec1.move(by: d.dec1)
    outProj.move(by: d.outProj)
  }

  public init(
    srcVocab: Int, tgtVocab: Int,
    dModel: Int, heads: Int, ff: Int,
    maxLength: Int,
    dtype: DType = .float32, device: Device = .cpu
  ) {
    self.srcEmb = Embedding(vocabSize: srcVocab, embedSize: dModel, dtype: dtype, device: device)
    self.tgtEmb = Embedding(vocabSize: tgtVocab, embedSize: dModel, dtype: dtype, device: device)
    self.posEnc = PositionalEncoding(
      maxLength: maxLength, embedSize: dModel, dtype: dtype, device: device)

    self.enc1 = TransformerEncoderLayer(
      embedDim: dModel, numHeads: heads, ffHiddenDim: ff, dtype: dtype, device: device)
    self.enc2 = TransformerEncoderLayer(
      embedDim: dModel, numHeads: heads, ffHiddenDim: ff, dtype: dtype, device: device)
    self.dec1 = TransformerDecoderLayer(
      embedDim: dModel, numHeads: heads, ffHiddenDim: ff, dtype: dtype, device: device)

    self.outProj = Linear(inputSize: dModel, outputSize: tgtVocab, dtype: dtype, device: device)
  }

  @differentiable(reverse)
  public func callAsFunction(_ input: Input) -> Tensor {
    // Embedding + positional enc
    let srcE = posEnc(srcEmb(input.src))  // [N, Ls, C]
    let tgtE = posEnc(tgtEmb(input.tgtIn))  // [N, Lt, C]

    // Encoder (2 layers)
    let e1 = enc1(.init(x: srcE, attnMask: input.srcMask))
    let mem = enc2(.init(x: e1, attnMask: input.srcMask))  // [N, Ls, C]

    // Cross mask can reuse srcPad form [N,1,1,Ls] — broadcast to [N,H,Lt,Ls].
    let crossMask = input.srcMask

    // Decoder (1 layer)
    let y = dec1(.init(x: tgtE, memory: mem, selfMask: input.tgtMask, crossMask: crossMask))  // [N, Lt, C]

    // Project to vocab
    return outProj(y)  // [N, Lt, V]
  }
}
