import Foundation
import _Differentiation

/// Batch normalization for stabilizing and accelerating CNN training.
///
/// `BatchNorm` normalizes activations across the batch dimension for each channel independently.
/// It's the standard normalization technique for Convolutional Neural Networks, providing stable
/// training and acting as a regularizer.
///
/// ## Overview
///
/// Batch normalization computes statistics (mean and variance) across the batch for each channel,
/// then normalizes and applies a learned affine transformation. Key characteristics:
/// - Normalizes across the batch dimension (N) for each channel (C)
/// - Maintains running statistics for inference
/// - Different behavior during training vs inference
/// - Acts as a regularizer (reduces need for dropout)
/// - Allows higher learning rates
///
/// ## Operation
///
/// ### Training
/// For each channel, compute batch statistics then normalize:
/// ```
/// μ_batch = E[x] over batch
/// σ²_batch = Var[x] over batch
/// normalized = (x - μ_batch) / sqrt(σ²_batch + ε)
/// output = γ * normalized + β
///
/// // Update running statistics
/// running_mean = (1 - momentum) * running_mean + momentum * μ_batch
/// running_var = (1 - momentum) * running_var + momentum * σ²_batch
/// ```
///
/// ### Inference
/// Use accumulated running statistics:
/// ```
/// normalized = (x - running_mean) / sqrt(running_var + ε)
/// output = γ * normalized + β
/// ```
///
/// Where `γ` (scale) and `β` (shift) are learned parameters per channel.
///
/// ## Creating BatchNorm Layers
///
/// ```swift
/// // For CNNs (after Conv2D)
/// let bn = BatchNorm(channels: 64)
///
/// // For MLPs (after Linear/Dense)
/// let bn = BatchNorm(channels: 256)
///
/// // Custom momentum (default 0.1)
/// let bn = BatchNorm(channels: 128, momentum: 0.01)
///
/// // Without learnable affine parameters
/// let bn = BatchNorm(channels: 64, affine: false)
/// ```
///
/// ## Usage in CNNs
///
/// ```swift
/// // Standard ResNet-style block
/// let cnn = Sequential {
///     Conv2D(inChannels: 64, outChannels: 128, kernelSize: (3, 3), padding: (1, 1))
///     BatchNorm(channels: 128)
///     ReLU()
///     Conv2D(inChannels: 128, outChannels: 128, kernelSize: (3, 3), padding: (1, 1))
///     BatchNorm(channels: 128)
/// }
///
/// // VGG-style block
/// let vggBlock = Sequential {
///     Conv2D(inChannels: 128, outChannels: 256, kernelSize: (3, 3), padding: (1, 1))
///     BatchNorm(channels: 256)
///     ReLU()
///     Conv2D(inChannels: 256, outChannels: 256, kernelSize: (3, 3), padding: (1, 1))
///     BatchNorm(channels: 256)
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))
/// }
/// ```
///
/// ## Shape Specifications
///
/// BatchNorm works with tensors of rank ≥ 2, normalizing across all dimensions except channel (dim 1):
///
/// ### For MLPs (Rank-2)
/// - **Input**: `[batch, channels]`
/// - **Normalizes over**: batch dimension
/// - **Parameters**: `[channels]` for both γ and β
///
/// ### For CNNs (Rank-4)
/// - **Input**: `[batch, channels, height, width]`
/// - **Normalizes over**: batch, height, width (keeps channels separate)
/// - **Parameters**: `[channels]` for both γ and β
/// - **Output**: Same shape as input
///
/// ```swift
/// let bn = BatchNorm(channels: 64)
///
/// // MLP usage
/// let mlp = Tensor.randn([32, 64])  // [batch, features]
/// let normed = bn(mlp)  // [32, 64]
///
/// // CNN usage
/// let features = Tensor.randn([32, 64, 28, 28])  // [batch, channels, h, w]
/// let normedCNN = bn(features)  // [32, 64, 28, 28]
/// ```
///
/// ## Training vs Inference
///
/// BatchNorm behaves differently depending on the learning phase:
///
/// ```swift
/// let bn = BatchNorm(channels: 64)
/// let input = Tensor.randn([32, 64, 28, 28])
///
/// // Training mode - uses batch statistics
/// Context.local.learningPhase = .training
/// let trainOutput = bn(input)
/// // Computes mean/var from current batch
/// // Updates running statistics
///
/// // Inference mode - uses running statistics
/// Context.local.learningPhase = .inference
/// let testOutput = bn(input)
/// // Uses accumulated running_mean and running_var
/// // More stable, batch-size independent
/// ```
///
/// ## BatchNorm vs LayerNorm
///
/// **Use BatchNorm when:**
/// - Building CNNs for computer vision
/// - Training with consistent, large batch sizes (≥16)
/// - Want spatial regularization effects
/// - Following established CNN architectures (ResNet, VGG, EfficientNet)
///
/// **Use LayerNorm when:**
/// - Building transformers or attention-based models
/// - Working with RNNs or sequence models
/// - Batch size is small or varies
/// - Need identical training and inference behavior
///
/// ### Key Differences
///
/// | Aspect | BatchNorm | LayerNorm |
/// |--------|-----------|-----------|
/// | Normalizes across | Batch (per channel) | Features (per sample) |
/// | Running statistics | Yes | No |
/// | Training vs Inference | Different | Identical |
/// | Batch size dependency | High | None |
/// | Common use | CNNs | Transformers, RNNs |
/// | Spatial regularization | Yes | No |
///
/// ## Placement in Networks
///
/// ### Modern Practice (Post-Activation)
///
/// ```swift
/// Conv2D → BatchNorm → ReLU
/// ```
///
/// ### Original ResNet (Pre-Activation)
///
/// ```swift
/// BatchNorm → ReLU → Conv2D
/// ```
///
/// Both work, but post-activation (Conv → BN → ReLU) is more common in modern architectures.
///
/// ## Momentum Parameter
///
/// Controls the exponential moving average of running statistics:
///
/// ```swift
/// // Fast adaptation (higher momentum)
/// let bn = BatchNorm(channels: 64, momentum: 0.1)  // Default, PyTorch-style
///
/// // Slower adaptation (more stable)
/// let bn = BatchNorm(channels: 64, momentum: 0.01)
///
/// // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
/// ```
///
/// Higher momentum = faster adaptation to new statistics.
///
/// ## Benefits of BatchNorm
///
/// 1. **Faster Training**: Allows higher learning rates
/// 2. **Better Convergence**: More stable gradient flow
/// 3. **Regularization**: Acts as regularizer, reduces overfitting
/// 4. **Less Sensitive to Initialization**: More forgiving of poor weight init
/// 5. **Reduces Internal Covariate Shift**: Stabilizes distribution of activations
///
/// ## Common Patterns
///
/// ### ResNet Bottleneck Block
///
/// ```swift
/// struct BottleneckBlock: Layer {
///     var conv1: Conv2D
///     var bn1: BatchNorm
///     var conv2: Conv2D
///     var bn2: BatchNorm
///     var conv3: Conv2D
///     var bn3: BatchNorm
///
///     init(channels: Int) {
///         conv1 = Conv2D(
///             kaimingUniformInChannels: channels,
///             outChannels: channels / 4,
///             kernelSize: (1, 1)
///         )
///         bn1 = BatchNorm(channels: channels / 4)
///
///         conv2 = Conv2D(
///             kaimingUniformInChannels: channels / 4,
///             outChannels: channels / 4,
///             kernelSize: (3, 3),
///             padding: (1, 1)
///         )
///         bn2 = BatchNorm(channels: channels / 4)
///
///         conv3 = Conv2D(
///             kaimingUniformInChannels: channels / 4,
///             outChannels: channels,
///             kernelSize: (1, 1)
///         )
///         bn3 = BatchNorm(channels: channels)
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         let residual = input
///         var x = conv1(input)
///         x = bn1(x).relu()
///         x = conv2(x)
///         x = bn2(x).relu()
///         x = conv3(x)
///         x = bn3(x)
///         return (x + residual).relu()
///     }
/// }
/// ```
///
/// ## Considerations
///
/// ### Batch Size
/// - **Small batches (<16)**: Statistics are noisy, consider LayerNorm or GroupNorm
/// - **Large batches (≥32)**: BatchNorm works best
///
/// ### Inference
/// - Ensure you've trained long enough for running statistics to stabilize
/// - Running statistics are updated during training only
///
/// ### Distributed Training
/// - Consider synchronized BatchNorm across devices for consistent statistics
///
/// ## Automatic Differentiation
///
/// BatchNorm is fully differentiable with efficient gradient computation:
///
/// ```swift
/// let bn = BatchNorm(channels: 64)
/// let input = Tensor.randn([32, 64, 28, 28])
///
/// Context.local.learningPhase = .training
/// let (output, pullback) = valueWithPullback(at: bn, input) { layer, x in
///     layer(x)
/// }
///
/// let gradOutput = Tensor.ones([32, 64, 28, 28])
/// let (bnGrad, inputGrad) = pullback(gradOutput)
/// // bnGrad.gamma: gradients for scale parameters
/// // bnGrad.beta: gradients for shift parameters
/// // inputGrad: gradients w.r.t. input
/// ```
///
/// ## Topics
///
/// ### Creating BatchNorm Layers
///
/// - ``init(channels:momentum:epsilon:affine:dtype:device:)``
/// - ``init(featureCount:momentum:epsilon:affine:dtype:device:)``
///
/// ### Properties
///
/// - ``gamma``
/// - ``beta``
/// - ``runningMean``
/// - ``runningVariance``
/// - ``momentum``
/// - ``epsilon``
/// - ``affine``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``LayerNorm`` - Layer normalization for transformers
/// - ``GroupNorm`` - Group normalization (batch-independent)
/// - ``Conv2D`` - Convolutional layers (commonly used with BatchNorm)
/// - ``Sequential`` - Compose layers including BatchNorm
public struct BatchNorm: Layer {
  /// The learnable scale (gain) parameter per channel.
  ///
  /// Shape: `[channels]`. Initialized to ones. Multiplies normalized values.
  public var gamma: Tensor

  /// The learnable shift (bias) parameter per channel.
  ///
  /// Shape: `[channels]`. Initialized to zeros. Added after scaling.
  public var beta: Tensor

  /// Running mean statistics accumulated during training, used during inference.
  ///
  /// Shape: `[channels]`. Updated via exponential moving average with `momentum`.
  @noDerivative public var runningMean: Parameter

  /// Running variance statistics accumulated during training, used during inference.
  ///
  /// Shape: `[channels]`. Updated via exponential moving average with `momentum`.
  @noDerivative public var runningVariance: Parameter

  /// Momentum for updating running statistics.
  ///
  /// Default: `0.1` (PyTorch-style). Formula: `running = (1 - momentum) * running + momentum * batch`.
  @noDerivative public var momentum: Float

  /// Small constant for numerical stability.
  ///
  /// Default: `1e-5`. Added to variance before taking square root.
  @noDerivative public var epsilon: Float

  /// Whether to apply learnable affine transformation.
  ///
  /// When `true` (default), learns `gamma` and `beta`. When `false`, only normalizes.
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

  /// Creates a batch normalization layer.
  ///
  /// Initializes `gamma` to ones, `beta` to zeros, and running statistics appropriately.
  ///
  /// - Parameters:
  ///   - channels: Number of channels (features) to normalize. For CNNs, this is the number
  ///               of output channels from the previous Conv2D layer.
  ///   - momentum: Momentum for running statistics updates. Default `0.1` (PyTorch-style).
  ///               Higher values adapt faster to new statistics.
  ///   - epsilon: Small constant for numerical stability. Default `1e-5`.
  ///   - affine: Whether to learn scale and shift parameters. Default `true`.
  ///   - dtype: Data type for parameters. Default `.float32`.
  ///   - device: Device for parameters. Default `.cpu`.
  ///
  /// ```swift
  /// // After Conv2D with 64 output channels
  /// let bn = BatchNorm(channels: 64)
  ///
  /// // With custom momentum
  /// let bn = BatchNorm(channels: 128, momentum: 0.01)
  ///
  /// // Without learnable affine
  /// let bn = BatchNorm(channels: 256, affine: false)
  /// ```
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

  /// Creates a batch normalization layer (backward compatibility alias).
  ///
  /// This initializer provides backward compatibility. Prefer using ``init(channels:momentum:epsilon:affine:dtype:device:)``.
  ///
  /// - Parameters:
  ///   - featureCount: Number of features/channels (alias for `channels`).
  ///   - momentum: Momentum for running statistics. Default `0.1`.
  ///   - epsilon: Numerical stability constant. Default `1e-5`.
  ///   - affine: Whether to learn affine parameters. Default `true`.
  ///   - dtype: Data type. Default `.float32`.
  ///   - device: Device. Default `.cpu`.
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

  /// Applies batch normalization to the input.
  ///
  /// Behavior depends on the current learning phase:
  /// - **Training**: Computes batch statistics, normalizes, updates running statistics
  /// - **Inference**: Uses accumulated running statistics for normalization
  ///
  /// - Parameter x: Input tensor with rank ≥ 2. Channel dimension must be at index 1.
  ///                Shape: `[batch, channels, ...]` (e.g., `[N, C]` or `[N, C, H, W]`)
  ///
  /// - Returns: Normalized tensor with same shape as input.
  ///
  /// ```swift
  /// let bn = BatchNorm(channels: 64)
  ///
  /// // Training mode
  /// Context.local.learningPhase = .training
  /// let features = Tensor.randn([32, 64, 28, 28])
  /// let normed = bn(features)  // Uses batch statistics, updates running stats
  ///
  /// // Inference mode
  /// Context.local.learningPhase = .inference
  /// let testFeatures = Tensor.randn([1, 64, 28, 28])
  /// let testNormed = bn(testFeatures)  // Uses running statistics
  ///
  /// // In a CNN
  /// var x = Tensor.randn([16, 64, 32, 32])
  /// x = conv(x)
  /// x = bn(x)
  /// x = x.relu()
  /// ```
  ///
  /// - Note: The layer automatically switches behavior based on `Context.local.learningPhase`.
  ///         Gradients are computed appropriately for training mode.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Decide phase based on thread-local context.
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
