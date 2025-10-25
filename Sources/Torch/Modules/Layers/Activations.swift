import Foundation
import _Differentiation

// MARK: - ReLU

/// Rectified Linear Unit activation function: `f(x) = max(0, x)`.
///
/// ReLU is the most commonly used activation function in deep learning due to its simplicity
/// and effectiveness. It introduces non-linearity while being computationally efficient.
///
/// ## Mathematical Definition
///
/// ```
/// ReLU(x) = max(0, x) = {  x  if x > 0
///                       {  0  otherwise
/// ```
///
/// ## Usage
///
/// ```swift
/// let model = Sequential {
///     Linear(inputSize: 784, outputSize: 512)
///     ReLU()  // Most common activation
///     Linear(inputSize: 512, outputSize: 10)
/// }
/// ```
///
/// ## Characteristics
///
/// - **Advantages**: Fast computation, helps with vanishing gradient problem, sparsity
/// - **Disadvantages**: "Dying ReLU" problem (neurons can permanently die)
/// - **Gradient**: 1 for x > 0, 0 for x ≤ 0
/// - **Range**: [0, ∞)
///
/// ## See Also
///
/// - ``LeakyReLU`` - Variant that allows small negative values
/// - ``GELU`` - Smoother alternative used in transformers
/// - ``SiLU`` - Self-gated activation function
public struct ReLU: Layer {
  public typealias Input = Tensor
  public typealias Output = Tensor

  /// Creates a ReLU activation layer.
  public init() {}

  /// Applies ReLU activation element-wise: `max(0, x)`.
  ///
  /// - Parameter x: Input tensor of any shape.
  /// - Returns: Output tensor of the same shape with ReLU applied element-wise.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.relu() }
}

extension ReLU {
  // Manual VJP via free closure to avoid the "curried self" path.
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (ReLU.TangentVector, Tensor))
  {
    func primal(_ s: ReLU, _ i: Tensor) -> Tensor { i.relu() }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (ReLU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> ReLU.TangentVector)
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

// MARK: - LeakyReLU (parameterless; slope is a hyper-parameter)
public struct LeakyReLU: Layer {
  @noDerivative public let negativeSlope: Float
  public init(negativeSlope: Float = 0.01) { self.negativeSlope = negativeSlope }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // leaky_relu(x) = relu(x) - a * relu(-x)  (no non-diff max/min needed). :contentReference[oaicite:7]{index=7}
    x.relu().subtracting(x.negated().relu().multiplying(negativeSlope))
  }
}

extension LeakyReLU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (LeakyReLU.TangentVector, Tensor))
  {
    func primal(_ s: LeakyReLU, _ i: Tensor) -> Tensor {
      i.relu().subtracting(i.negated().relu().multiplying(s.negativeSlope))
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (LeakyReLU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> LeakyReLU.TangentVector)
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

// MARK: - SiLU / Swish: x * sigmoid(x)

/// Sigmoid Linear Unit (SiLU), also known as Swish: `f(x) = x * σ(x)`.
///
/// SiLU is a smooth, non-monotonic activation function that has shown strong performance
/// in deep networks. It's a self-gated activation where the input is multiplied by its sigmoid.
///
/// ## Mathematical Definition
///
/// ```
/// SiLU(x) = x * σ(x) = x / (1 + e^(-x))
/// ```
///
/// Where σ(x) is the sigmoid function.
///
/// ## Usage
///
/// ```swift
/// let model = Sequential {
///     Linear(inputSize: 512, outputSize: 512)
///     SiLU()  // Smooth alternative to ReLU
///     Linear(inputSize: 512, outputSize: 256)
/// }
/// ```
///
/// ## Characteristics
///
/// - **Advantages**: Smooth, non-monotonic, self-gating, better than ReLU in some cases
/// - **Disadvantages**: More computationally expensive than ReLU
/// - **Gradient**: Smooth and bounded
/// - **Range**: (-∞, ∞) but mostly in [0, ∞)
/// - **Also Known As**: Swish activation
///
/// ## When to Use
///
/// - Deep residual networks
/// - As a smooth alternative to ReLU
/// - When training very deep networks
/// - Mobile and efficient networks (MobileNetV3)
///
/// ## See Also
///
/// - ``ReLU`` - Simpler, faster activation
/// - ``GELU`` - Similar smooth activation used in transformers
/// - ``Sigmoid`` - Component of SiLU
public struct SiLU: Layer {
  public typealias Input = Tensor
  public typealias Output = Tensor

  /// Creates a SiLU activation layer.
  public init() {}

  /// Applies SiLU activation element-wise: `x * σ(x)`.
  ///
  /// - Parameter x: Input tensor of any shape.
  /// - Returns: Output tensor of the same shape with SiLU applied element-wise.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.multiplying(x.sigmoid()) }
}

extension SiLU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (SiLU.TangentVector, Tensor))
  {
    func primal(_ s: SiLU, _ i: Tensor) -> Tensor { i.multiplying(i.sigmoid()) }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (SiLU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> SiLU.TangentVector)
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

// MARK: - GELU (exact via erf, or fast tanh-approx)

/// Gaussian Error Linear Unit (GELU): A smooth approximation to ReLU.
///
/// GELU is the activation function used in state-of-the-art language models like BERT and GPT.
/// It provides a smooth, non-monotonic activation that weights inputs by their magnitude rather
/// than using a hard cutoff like ReLU.
///
/// ## Mathematical Definition
///
/// The exact form using the error function:
/// ```
/// GELU(x) = x * Φ(x) = x * (1/2)[1 + erf(x/√2)]
/// ```
///
/// Fast approximation using tanh:
/// ```
/// GELU(x) ≈ 0.5 * x * [1 + tanh(√(2/π) * (x + 0.044715 * x³))]
/// ```
///
/// Where Φ(x) is the cumulative distribution function of the standard normal distribution.
///
/// ## Usage
///
/// ```swift
/// // Standard usage (fast approximation)
/// let transformer = Sequential {
///     Linear(inputSize: 768, outputSize: 3072)
///     GELU()  // Used in BERT, GPT, etc.
///     Linear(inputSize: 3072, outputSize: 768)
/// }
///
/// // Exact version (slightly slower)
/// let exact = GELU(approximate: false)
/// ```
///
/// ## Characteristics
///
/// - **Advantages**: Smooth, differentiable everywhere, better than ReLU for transformers
/// - **Disadvantages**: More computationally expensive than ReLU
/// - **Gradient**: Smooth and continuous
/// - **Range**: (-∞, ∞) but mostly in [0, ∞)
/// - **Used In**: BERT, GPT-2, GPT-3, Vision Transformers
///
/// ## Approximate vs Exact
///
/// - **Approximate** (default): Fast tanh-based approximation, ~98% accurate
/// - **Exact**: Uses error function, slightly slower but mathematically exact
///
/// ```swift
/// // Fast for production (default)
/// let fast = GELU(approximate: true)
///
/// // Exact for research/verification
/// let exact = GELU(approximate: false)
/// ```
///
/// ## When to Use
///
/// - Transformer architectures (BERT, GPT)
/// - Vision transformers (ViT)
/// - When you need smooth activation with good gradient properties
/// - State-of-the-art NLP models
///
/// ## See Also
///
/// - ``ReLU`` - Simpler, faster activation
/// - ``SiLU`` - Another smooth activation
/// - ``Tanh`` - Used in GELU approximation
public struct GELU: Layer {
  /// Whether to use the fast tanh approximation (default: true).
  ///
  /// - `true`: Fast approximation using tanh, ~98% accurate
  /// - `false`: Exact computation using error function
  @noDerivative public let approximate: Bool

  /// Creates a GELU activation layer.
  ///
  /// - Parameter approximate: If `true`, uses fast tanh approximation. If `false`, uses exact
  ///                         error function computation. Defaults to `true`.
  ///
  /// ```swift
  /// // Fast approximation (recommended)
  /// let gelu = GELU()  // or GELU(approximate: true)
  ///
  /// // Exact computation
  /// let exactGelu = GELU(approximate: false)
  /// ```
  public init(approximate: Bool = true) { self.approximate = approximate }

  public typealias Input = Tensor
  public typealias Output = Tensor

  /// Applies GELU activation element-wise.
  ///
  /// - Parameter x: Input tensor of any shape.
  /// - Returns: Output tensor of the same shape with GELU applied element-wise.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    if approximate {
      // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
      let k: Float = 0.7978845608028654  // √(2/π)
      let c: Float = 0.044715
      let inner = x.adding(x.pow(3).multiplying(c)).multiplying(k)  // pow has VJP. :contentReference[oaicite:9]{index=9}
      return x.multiplying(0.5).multiplying(inner.tanh().adding(1))  // tanh VJP exists. :contentReference[oaicite:10]{index=10}
    } else {
      // 0.5 * x * (1 + erf(x / √2))
      let invRt2: Float = 0.7071067811865476
      return x.multiplying(0.5).multiplying(x.multiplying(invRt2).erf().adding(1))  // erf VJP exists. :contentReference[oaicite:11]{index=11}
    }
  }
}

extension GELU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (GELU.TangentVector, Tensor))
  {
    let y = callAsFunction(x)

    let derivative: Tensor
    if approximate {
      let k = Tensor(0.7978845608028654, dtype: x.dtype ?? .float32, device: x.device)  // √(2/π)
      let c = Tensor(0.044715, dtype: x.dtype ?? .float32, device: x.device)
      let half = Tensor(0.5, dtype: x.dtype ?? .float32, device: x.device)

      let xSquared = x.multiplying(x)
      let inner = k.multiplying(x.adding(xSquared.multiplying(x).multiplying(c)))
      let tanhInner = inner.tanh()
      let sech2 = Tensor(1, dtype: x.dtype ?? .float32, device: x.device).subtracting(tanhInner.multiplying(tanhInner))
      let fPrime = k.multiplying(Tensor(1, dtype: x.dtype ?? .float32, device: x.device).adding(xSquared.multiplying(c).multiplying(3)))
      derivative = half.multiplying(tanhInner.adding(1)).adding(half.multiplying(x).multiplying(sech2).multiplying(fPrime))
    } else {
      let dtype = x.dtype ?? .float32
      let device = x.device
      let half = Tensor(0.5, dtype: dtype, device: device)
      let invRt2 = Tensor(0.7071067811865476, dtype: dtype, device: device)
      let sqrt2OverPi = Tensor(0.7978845608028654, dtype: dtype, device: device)
      let base = x.multiplying(invRt2).erf().adding(1)
      let expTerm = x.multiplying(x).dividing(-2).exp()
      derivative = half.multiplying(base).adding(x.multiplying(expTerm).multiplying(sqrt2OverPi))
    }

    return (
      y,
      { v in
        let dx = v.multiplying(derivative)
        return (GELU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> GELU.TangentVector)
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

// MARK: - Tanh / Sigmoid (thin wrappers)

/// Hyperbolic tangent activation function: `f(x) = tanh(x)`.
///
/// Tanh is a classic activation function that squashes input values to the range (-1, 1).
/// It's a scaled and shifted version of the sigmoid function, centered at zero.
///
/// ## Mathematical Definition
///
/// ```
/// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = (e^(2x) - 1) / (e^(2x) + 1)
/// ```
///
/// ## Usage
///
/// ```swift
/// let model = Sequential {
///     Linear(inputSize: 100, outputSize: 50)
///     Tanh()  // Outputs in range (-1, 1)
///     Linear(inputSize: 50, outputSize: 10)
/// }
/// ```
///
/// ## Characteristics
///
/// - **Advantages**: Zero-centered (unlike sigmoid), smooth, bounded output
/// - **Disadvantages**: Vanishing gradient problem for large |x|, slower than ReLU
/// - **Gradient**: Maximum at x=0 (gradient=1), approaches 0 for large |x|
/// - **Range**: (-1, 1)
/// - **Historically Used In**: RNNs, LSTMs (for gating)
///
/// ## When to Use
///
/// - RNN/LSTM gates (standard component)
/// - When you need outputs in (-1, 1) range
/// - Older architectures (less common in modern CNNs)
/// - As a normalizing function
///
/// ## Tanh vs Sigmoid
///
/// Tanh is generally preferred over sigmoid for hidden layers because:
/// - Zero-centered output helps with optimization
/// - Stronger gradients in the middle range
/// - Related by: tanh(x) = 2*sigmoid(2x) - 1
///
/// ## See Also
///
/// - ``Sigmoid`` - Related function with range (0, 1)
/// - ``ReLU`` - Modern alternative for hidden layers
/// - ``GELU`` - Smooth modern activation
public struct Tanh: Layer {
  public typealias Input = Tensor
  public typealias Output = Tensor

  /// Creates a Tanh activation layer.
  public init() {}

  /// Applies tanh activation element-wise, squashing to (-1, 1).
  ///
  /// - Parameter x: Input tensor of any shape.
  /// - Returns: Output tensor of the same shape with tanh applied element-wise.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.tanh() }
}
extension Tanh {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Tanh.TangentVector, Tensor))
  {
    func primal(_ s: Tanh, _ i: Tensor) -> Tensor { i.tanh() }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (Tanh.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Tanh.TangentVector)
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

/// Sigmoid activation function: `f(x) = 1 / (1 + e^(-x))`.
///
/// Sigmoid squashes input values to the range (0, 1), making it useful for binary classification
/// and probability outputs. It's also used as a gating mechanism in RNNs and LSTMs.
///
/// ## Mathematical Definition
///
/// ```
/// σ(x) = 1 / (1 + e^(-x))
/// ```
///
/// ## Usage
///
/// ```swift
/// // Binary classification output
/// let binaryClassifier = Sequential {
///     Linear(inputSize: 256, outputSize: 128)
///     ReLU()
///     Linear(inputSize: 128, outputSize: 1)
///     Sigmoid()  // Probability output in (0, 1)
/// }
///
/// // Multi-label classification (independent probabilities)
/// let multiLabel = Sequential {
///     Linear(inputSize: 512, outputSize: 10)
///     Sigmoid()  // Each output is independent probability
/// }
/// ```
///
/// ## Characteristics
///
/// - **Advantages**: Smooth, bounded output, interpretable as probability
/// - **Disadvantages**: Vanishing gradient problem, not zero-centered, slower than ReLU
/// - **Gradient**: Maximum at x=0 (gradient=0.25), approaches 0 for large |x|
/// - **Range**: (0, 1)
/// - **Used In**: Binary classification, LSTM gates, attention weights
///
/// ## When to Use
///
/// - **Binary classification**: Final layer for binary outputs
/// - **Multi-label classification**: When classes are independent
/// - **Gating mechanisms**: In LSTMs and GRUs
/// - **Attention weights**: To produce probability distributions
/// - **NOT for hidden layers**: Use ReLU, GELU, or other modern activations instead
///
/// ## Sigmoid vs Softmax
///
/// - **Sigmoid**: For independent binary decisions (multi-label)
/// - **Softmax**: For mutually exclusive classes (multi-class)
///
/// ```swift
/// // Multi-label (can be multiple positives)
/// let labels = Sigmoid()(logits)  // Each in (0, 1), independent
///
/// // Multi-class (exactly one class)
/// let probs = Softmax()(logits)   // Sum to 1, mutually exclusive
/// ```
///
/// ## See Also
///
/// - ``Tanh`` - Related function with range (-1, 1)
/// - ``Softmax`` - For multi-class classification
/// - ``SiLU`` - Uses sigmoid as gating mechanism
public struct Sigmoid: Layer {
  public typealias Input = Tensor
  public typealias Output = Tensor

  /// Creates a Sigmoid activation layer.
  public init() {}

  /// Applies sigmoid activation element-wise, squashing to (0, 1).
  ///
  /// - Parameter x: Input tensor of any shape.
  /// - Returns: Output tensor of the same shape with sigmoid applied element-wise.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.sigmoid() }
}
extension Sigmoid {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Sigmoid.TangentVector, Tensor))
  {
    func primal(_ s: Sigmoid, _ i: Tensor) -> Tensor { i.sigmoid() }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (Sigmoid.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Sigmoid.TangentVector)
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

// MARK: - ELU (alpha is hyper-parameter)
public struct ELU: Layer {
  @noDerivative public let alpha: Float
  public init(alpha: Float = 1.0) { self.alpha = alpha }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // where x>0 ? x : alpha*(exp(x)-1) — route grad via TorchWhere.select. :contentReference[oaicite:14]{index=14}
    let cond = withoutDerivative(at: x.gt(0))  // boolean mask
    let pos = x
    let neg = x.exp().subtracting(1).multiplying(alpha)
    return TorchWhere.select(condition: cond, pos, neg)  // differentiable select. :contentReference[oaicite:15]{index=15}
  }
}
extension ELU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (ELU.TangentVector, Tensor))
  {
    func primal(_ s: ELU, _ i: Tensor) -> Tensor {
      let cond = withoutDerivative(at: i.gt(0))
      let pos = i
      let neg = i.exp().subtracting(1).multiplying(s.alpha)
      return TorchWhere.select(condition: cond, pos, neg)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (ELU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> ELU.TangentVector)
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

// MARK: - Softplus: y = (1/β) * log(1 + exp(βx)) with stable branching
public struct Softplus: Layer {
  @noDerivative public let beta: Float
  public init(beta: Float = 1.0) { self.beta = beta }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Stable form:
    // z = βx
    // if z > 0: z + log(1 + exp(-z)) else: log(1 + exp(z))
    let z = x.multiplying(beta)
    let one = Tensor(1)
    let pos = z.adding((z.negated()).exp().adding(one).log())
    let neg = z.exp().adding(one).log()
    let cond = withoutDerivative(at: z.gt(0))
    return TorchWhere.select(condition: cond, pos, neg).dividing(beta)  // select has VJP. :contentReference[oaicite:16]{index=16}
  }
}
extension Softplus {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Softplus.TangentVector, Tensor))
  {
    func primal(_ s: Softplus, _ i: Tensor) -> Tensor {
      let z = i.multiplying(s.beta)
      let one = Tensor(1)
      let pos = z.adding((z.negated()).exp().adding(one).log())
      let neg = z.exp().adding(one).log()
      let cond = withoutDerivative(at: z.gt(0))
      return TorchWhere.select(condition: cond, pos, neg).dividing(s.beta)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (Softplus.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Softplus.TangentVector)
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

// MARK: - Softmax (axis-aware; stable with detached shift)
public struct Softmax: Layer {
  @noDerivative public let axis: Int
  public init(axis: Int = -1) { self.axis = axis }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let a = withoutDerivative(at: _normalizeDimension(axis, rank: x.rank))  // supports negatives. :contentReference[oaicite:17]{index=17}
    // Detach the shift (max) so we do not need a derivative for max. :contentReference[oaicite:18]{index=18}
    let shift = withoutDerivative(at: x.max(dim: a, keepdim: true).values)
    let expX = (x - shift).exp()  // exp VJP. :contentReference[oaicite:19]{index=19}
    let denom = expX.sum(dim: a, keepdim: true)  // sum VJP. :contentReference[oaicite:20]{index=20}
    return expX.dividing(denom)
  }
}
extension Softmax {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Softmax.TangentVector, Tensor))
  {
    func primal(_ s: Softmax, _ i: Tensor) -> Tensor {
      let a = withoutDerivative(at: _normalizeDimension(s.axis, rank: i.rank))
      let shift = withoutDerivative(at: i.max(dim: a, keepdim: true).values)
      let expX = (i - shift).exp()
      let denom = expX.sum(dim: a, keepdim: true)
      return expX.dividing(denom)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Softmax.TangentVector)
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

// MARK: - LogSoftmax (axis-aware; stable with detached shift)
public struct LogSoftmax: Layer {
  @noDerivative public let axis: Int
  public init(axis: Int = -1) { self.axis = axis }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let a = _normalizeDimension(axis, rank: x.rank)  // :contentReference[oaicite:21]{index=21}
    let shift = withoutDerivative(at: x.max(dim: a, keepdim: true).values)  // :contentReference[oaicite:22]{index=22}
    let z = x - shift
    let lse = z.exp().sum(dim: a, keepdim: true).log()  // log VJP exists. :contentReference[oaicite:23]{index=23}
    return z - lse
  }
}
extension LogSoftmax {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (LogSoftmax.TangentVector, Tensor))
  {
    let axisNorm = withoutDerivative(at: _normalizeDimension(axis, rank: x.rank))
    let y = callAsFunction(x)
    let soft = y.exp()
    return (
      y,
      { v in
        let sumV = v.sum(dim: axisNorm, keepdim: true)
        let dx = v - soft.multiplying(sumV)
        return (LogSoftmax.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> LogSoftmax.TangentVector)
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
