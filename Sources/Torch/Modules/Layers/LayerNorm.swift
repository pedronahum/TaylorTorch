import Foundation
import _Differentiation

/// Layer normalization for stabilizing and accelerating neural network training.
///
/// `LayerNorm` normalizes activations across the feature dimension for each sample independently.
/// It's the standard normalization technique for transformers and recurrent networks, providing
/// stable training without dependence on batch statistics.
///
/// ## Overview
///
/// Layer normalization computes statistics (mean and variance) across features for each sample,
/// then normalizes and applies a learned affine transformation. Unlike batch normalization, it:
/// - Works identically during training and inference (no running statistics)
/// - Is independent of batch size (works with batch size = 1)
/// - Is ideal for sequence models and transformers
/// - Normalizes across features, not across the batch
///
/// ## Operation
///
/// For each sample independently:
/// ```
/// mean = E[x] over features
/// variance = E[(x - mean)²] over features
/// normalized = (x - mean) / sqrt(variance + epsilon)
/// output = gamma * normalized + beta
/// ```
///
/// Where `gamma` (scale) and `beta` (shift) are learned parameters.
///
/// ## Creating LayerNorm Layers
///
/// ```swift
/// // Standard layer norm for transformers (last dimension)
/// let norm = LayerNorm(featureCount: 512)
///
/// // Without learnable affine parameters
/// let norm = LayerNorm(featureCount: 256, affine: false)
///
/// // Custom epsilon for numerical stability
/// let norm = LayerNorm(featureCount: 768, epsilon: 1e-6)
/// ```
///
/// ## Usage in Transformers
///
/// ```swift
/// // Transformer layer with pre-norm
/// struct TransformerBlock: Layer {
///     var norm1: LayerNorm
///     var attention: MultiHeadAttention
///     var norm2: LayerNorm
///     var ffn: Sequential<...>
///
///     init(modelDim: Int) {
///         norm1 = LayerNorm(featureCount: modelDim)
///         attention = MultiHeadAttention(modelDim: modelDim, numHeads: 8)
///         norm2 = LayerNorm(featureCount: modelDim)
///         ffn = Sequential {
///             Linear(inputSize: modelDim, outputSize: modelDim * 4)
///             ReLU()
///             Linear(inputSize: modelDim * 4, outputSize: modelDim)
///         }
///     }
///
///     @differentiable
///     func callAsFunction(_ x: Tensor) -> Tensor {
///         // Pre-norm residual connections
///         var h = x + attention(norm1(x))
///         h = h + ffn(norm2(h))
///         return h
///     }
/// }
/// ```
///
/// ## Shape Specifications
///
/// By default, LayerNorm normalizes over the last dimension:
///
/// - **Input**: `[batch, ..., features]` (any rank ≥ 1)
/// - **Parameters**: `gamma`, `beta` with shape `[features]`
/// - **Output**: Same shape as input `[batch, ..., features]`
///
/// ### Common Shapes
///
/// ```swift
/// let norm = LayerNorm(featureCount: 512)
///
/// // Transformer sequence
/// let seq = Tensor.randn([32, 128, 512])  // [batch, seqLen, modelDim]
/// let normed = norm(seq)  // [32, 128, 512]
///
/// // Single vector
/// let vec = Tensor.randn([512])
/// let normedVec = norm(vec)  // [512]
///
/// // Image features
/// let features = Tensor.randn([16, 256, 14, 14])  // Normalize over channels
/// let normChannels = LayerNorm(featureCount: 256, axis: 1)
/// let normed = normChannels(features)  // [16, 256, 14, 14]
/// ```
///
/// ## LayerNorm vs BatchNorm
///
/// **Use LayerNorm when:**
/// - Building transformers or attention-based models (standard choice)
/// - Working with RNNs or sequence models
/// - Batch size varies or is very small
/// - Need identical behavior during training and inference
/// - Online learning or single-sample inference
///
/// **Use BatchNorm when:**
/// - Building CNNs for computer vision
/// - Large, consistent batch sizes available
/// - Following established CNN architectures (ResNet, VGG)
/// - Spatial regularization is beneficial
///
/// ### Key Differences
///
/// | Aspect | LayerNorm | BatchNorm |
/// |--------|-----------|-----------|
/// | Normalizes across | Features (per sample) | Batch (per feature) |
/// | Running statistics | No | Yes |
/// | Training vs Inference | Identical | Different |
/// | Batch size dependency | Independent | Dependent |
/// | Common use | Transformers, RNNs | CNNs |
///
/// ## Usage Patterns
///
/// ### Pre-Norm (Modern Transformers)
///
/// ```swift
/// // Normalize before sub-layer (GPT-2, GPT-3 style)
/// var x = input
/// x = x + attention(norm1(x))
/// x = x + ffn(norm2(x))
/// ```
///
/// ### Post-Norm (Original Transformer)
///
/// ```swift
/// // Normalize after sub-layer (original "Attention is All You Need")
/// var x = input
/// x = norm1(x + attention(x))
/// x = norm2(x + ffn(x))
/// ```
///
/// ### RNN / LSTM
///
/// ```swift
/// // Stabilize recurrent connections
/// struct LayerNormLSTM: Layer {
///     var lstm: LSTMCell
///     var norm: LayerNorm
///
///     @differentiable
///     func callAsFunction(_ input: Tensor, state: (Tensor, Tensor)) -> (Tensor, (Tensor, Tensor)) {
///         let (h, c) = lstm(input, state: state)
///         let hNorm = norm(h)  // Normalize hidden state
///         return (hNorm, (hNorm, c))
///     }
/// }
/// ```
///
/// ## Affine Parameters
///
/// When `affine = true` (default), LayerNorm learns scale (`gamma`) and shift (`beta`):
///
/// ```swift
/// // With learnable affine (default)
/// let norm = LayerNorm(featureCount: 512, affine: true)
/// // Learns gamma and beta
///
/// // Without affine (just normalization)
/// let norm = LayerNorm(featureCount: 512, affine: false)
/// // Fixed gamma=1, beta=0
/// ```
///
/// Setting `affine = false` can be useful when you want pure normalization without
/// additional parameters, or when the affine transformation is handled elsewhere.
///
/// ## Numerical Stability
///
/// The `epsilon` parameter prevents division by zero:
///
/// ```swift
/// // Standard epsilon (default)
/// let norm = LayerNorm(featureCount: 512, epsilon: 1e-5)
///
/// // Tighter epsilon for float32 (if needed)
/// let norm = LayerNorm(featureCount: 512, epsilon: 1e-6)
/// ```
///
/// ## Automatic Differentiation
///
/// LayerNorm is fully differentiable with efficient gradient computation:
///
/// ```swift
/// let norm = LayerNorm(featureCount: 512)
/// let input = Tensor.randn([32, 128, 512])
///
/// let (output, pullback) = valueWithPullback(at: norm, input) { n, x in
///     n(x)
/// }
///
/// let gradOutput = Tensor.ones([32, 128, 512])
/// let (normGrad, inputGrad) = pullback(gradOutput)
/// // normGrad.gamma: gradients for scale parameters
/// // normGrad.beta: gradients for shift parameters
/// // inputGrad: gradients w.r.t. input
/// ```
///
/// ## Performance Considerations
///
/// - **No Batch Dependency**: Scales well with any batch size
/// - **No Running Stats**: No extra memory or state to manage
/// - **GPU Efficient**: Well-optimized on modern hardware
/// - **Inference**: Identical to training (no mode switching needed)
///
/// ## Topics
///
/// ### Creating LayerNorm Layers
///
/// - ``init(featureCount:axis:epsilon:affine:dtype:device:)``
///
/// ### Properties
///
/// - ``gamma``
/// - ``beta``
/// - ``epsilon``
/// - ``axis``
/// - ``affine``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``BatchNorm`` - Batch normalization for CNNs
/// - ``GroupNorm`` - Group normalization alternative
/// - ``MultiHeadAttention`` - Often used with LayerNorm
/// - ``Sequential`` - Compose layers including LayerNorm
public struct LayerNorm: Layer {
  /// The learnable scale (gain) parameter.
  ///
  /// Shape: `[featureCount]`
  ///
  /// Multiplies the normalized values. Initialized to all ones. Only used when `affine = true`.
  public var gamma: Tensor

  /// The learnable shift (bias) parameter.
  ///
  /// Shape: `[featureCount]`
  ///
  /// Added to the scaled normalized values. Initialized to all zeros. Only used when `affine = true`.
  public var beta: Tensor

  /// Small constant added to variance for numerical stability.
  ///
  /// Prevents division by zero when variance is very small. Default is `1e-5`.
  @noDerivative public var epsilon: Float

  /// The dimension to normalize over.
  ///
  /// Default is `-1` (last dimension). Negative indices count from the end.
  /// For transformer sequences `[batch, seqLen, features]`, use `-1` to normalize over features.
  @noDerivative public var axis: Int

  /// Whether to apply learnable affine transformation (scale and shift).
  ///
  /// When `true` (default), learns `gamma` and `beta` parameters.
  /// When `false`, only normalizes without learnable parameters.
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

  /// Creates a layer normalization layer.
  ///
  /// Initializes `gamma` to ones and `beta` to zeros, following standard practice.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features to normalize over (size of the normalization dimension).
  ///                   For transformers with model dimension 512, use `featureCount: 512`.
  ///   - axis: The dimension to normalize over. Defaults to `-1` (last dimension).
  ///           Negative indices count from the end. For shape `[batch, seq, features]`,
  ///           use `-1` to normalize over features.
  ///   - epsilon: Small constant for numerical stability. Defaults to `1e-5`.
  ///              Added to variance before taking square root to prevent division by zero.
  ///   - affine: Whether to learn scale (`gamma`) and shift (`beta`) parameters.
  ///             Defaults to `true`. Set to `false` for pure normalization without learnable parameters.
  ///   - dtype: Data type for parameters. Defaults to `.float32`.
  ///   - device: Device where parameters will be allocated. Defaults to `.cpu`.
  ///
  /// ```swift
  /// // Standard transformer layer norm
  /// let norm = LayerNorm(featureCount: 512)
  ///
  /// // BERT-style (featureCount = 768)
  /// let bertNorm = LayerNorm(featureCount: 768, epsilon: 1e-12)
  ///
  /// // Without learnable parameters
  /// let fixedNorm = LayerNorm(featureCount: 256, affine: false)
  ///
  /// // Normalize over channel dimension (axis=1 for [batch, channels, ...])
  /// let channelNorm = LayerNorm(featureCount: 256, axis: 1)
  /// ```
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

  /// Applies layer normalization to the input.
  ///
  /// Normalizes the input across the feature dimension for each sample independently,
  /// then applies the learned affine transformation (if `affine = true`).
  ///
  /// - Parameter x: Input tensor with rank ≥ 1. The dimension at `axis` must match `featureCount`.
  ///
  /// - Returns: Normalized tensor with the same shape as input.
  ///
  /// ```swift
  /// let norm = LayerNorm(featureCount: 512)
  ///
  /// // Transformer sequence
  /// let tokens = Tensor.randn([32, 128, 512])  // [batch, seqLen, modelDim]
  /// let normed = norm(tokens)  // [32, 128, 512]
  ///
  /// // Single sample
  /// let sample = Tensor.randn([512])
  /// let normedSample = norm(sample)  // [512]
  ///
  /// // In a residual connection
  /// let residual = input
  /// var x = norm(input)
  /// x = attention(x)
  /// x = x + residual  // Add & Norm pattern
  /// ```
  ///
  /// - Note: This method is marked `@differentiable`, enabling gradient computation
  ///         through the normalization and affine transformation.
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
