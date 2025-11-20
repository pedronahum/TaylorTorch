import Foundation
import _Differentiation

// MARK: - Conv2D

/// A 2D convolutional layer for processing spatial data like images.
///
/// `Conv2D` applies learnable convolutional filters over 2D input tensors, making it the
/// fundamental building block for Convolutional Neural Networks (CNNs). It's designed for
/// computer vision tasks where spatial relationships and local patterns matter.
///
/// ## Overview
///
/// Convolutional layers work by sliding small learnable filters (kernels) across the input,
/// computing dot products at each position. This creates feature maps that detect patterns
/// like edges, textures, and more complex features in deeper layers.
///
/// ### Operation
///
/// For input `x` with shape `[batch, inputChannels, height, width]`:
///
/// ```
/// output[b, c_out, h', w'] = Σ(weight[c_out, c_in, kh, kw] * input[b, c_in, h+kh, w+kw]) + bias[c_out]
/// ```
///
/// Where the sum is over input channels and kernel spatial dimensions.
///
/// ## Creating Conv2D Layers
///
/// ```swift
/// // Basic 3x3 convolution
/// let conv = Conv2D(
///     inChannels: 3,      // RGB input
///     outChannels: 64,     // 64 feature maps
///     kernelSize: (3, 3)
/// )
///
/// // With stride and padding (for size preservation)
/// let conv = Conv2D(
///     inChannels: 64,
///     outChannels: 128,
///     kernelSize: (3, 3),
///     stride: (1, 1),
///     padding: (1, 1)  // Same padding
/// )
///
/// // Kaiming initialization (better for ReLU)
/// let conv = Conv2D(
///     kaimingUniformInChannels: 128,
///     outChannels: 256,
///     kernelSize: (3, 3),
///     padding: (1, 1),
///     nonlinearity: .relu
/// )
/// ```
///
/// ## Usage in CNNs
///
/// ```swift
/// // Simple CNN block
/// let cnn = Sequential {
///     Conv2D(inChannels: 3, outChannels: 32, kernelSize: (3, 3), padding: (1, 1))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))
///     Conv2D(inChannels: 32, outChannels: 64, kernelSize: (3, 3), padding: (1, 1))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))
///     Flatten()
///     Linear(inputSize: 64 * 8 * 8, outputSize: 10)
/// }
///
/// let images = Tensor.randn([32, 3, 32, 32])  // CIFAR-10 batch
/// let logits = cnn(images)  // [32, 10]
/// ```
///
/// ## Shape Specifications
///
/// - **Input**: `[batch, inputChannels, height, width]`
/// - **Weight**: `[outputChannels, inputChannels/groups, kernelHeight, kernelWidth]`
/// - **Bias**: `[outputChannels]`
/// - **Output**: `[batch, outputChannels, outputHeight, outputWidth]`
///
/// Output spatial dimensions are calculated as:
///
/// ```
/// outputHeight = floor((height + 2*padding[0] - dilation[0]*(kernelSize[0]-1) - 1) / stride[0] + 1)
/// outputWidth  = floor((width + 2*padding[1] - dilation[1]*(kernelSize[1]-1) - 1) / stride[1] + 1)
/// ```
///
/// ### Common Configurations
///
/// ```swift
/// // Same padding (preserves spatial dimensions when stride=1)
/// let conv = Conv2D(
///     inChannels: 64,
///     outChannels: 64,
///     kernelSize: (3, 3),
///     stride: (1, 1),
///     padding: (1, 1)  // (kernelSize - 1) / 2
/// )
/// // Input [N, 64, 32, 32] → Output [N, 64, 32, 32]
///
/// // Downsampling (reduces spatial dimensions)
/// let downsample = Conv2D(
///     inChannels: 64,
///     outChannels: 128,
///     kernelSize: (3, 3),
///     stride: (2, 2),
///     padding: (1, 1)
/// )
/// // Input [N, 64, 32, 32] → Output [N, 128, 16, 16]
/// ```
///
/// ## Parameters
///
/// ### Stride
///
/// Controls how much the filter moves at each step. Larger strides reduce output size:
///
/// ```swift
/// // stride=(1,1): dense scanning (default)
/// // stride=(2,2): skip every other position (2x downsampling)
/// ```
///
/// ### Padding
///
/// Adds zeros around the input border to control output size:
///
/// ```swift
/// // padding=(0,0): valid convolution (reduces size)
/// // padding=(1,1): for 3x3 kernels, preserves size when stride=1
/// ```
///
/// ### Dilation
///
/// Introduces gaps in the kernel for larger receptive fields:
///
/// ```swift
/// // dilation=(1,1): standard convolution
/// // dilation=(2,2): atrous/dilated convolution (wider receptive field)
/// ```
///
/// ### Groups
///
/// Splits convolution into independent groups:
///
/// ```swift
/// // groups=1: standard convolution (default)
/// // groups=inputChannels: depthwise convolution (MobileNets)
/// ```
///
/// ## Weight Initialization
///
/// Two initialization strategies are available:
///
/// ### 1. Glorot (Xavier) Initialization (Default)
///
/// Good for sigmoid/tanh activations:
///
/// ```swift
/// let conv = Conv2D(inChannels: 64, outChannels: 128, kernelSize: (3, 3))
/// // Weights ~ U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
/// ```
///
/// ### 2. Kaiming (He) Initialization
///
/// Better for ReLU activations (recommended for modern CNNs):
///
/// ```swift
/// let conv = Conv2D(
///     kaimingUniformInChannels: 64,
///     outChannels: 128,
///     kernelSize: (3, 3),
///     nonlinearity: .relu  // or .leakyReLU(negativeSlope: 0.01)
/// )
/// ```
///
/// ## Common CNN Architectures
///
/// ### VGG-style Block
///
/// ```swift
/// let vggBlock = Sequential {
///     Conv2D(inChannels: 64, outChannels: 128, kernelSize: (3, 3), padding: (1, 1))
///     ReLU()
///     Conv2D(inChannels: 128, outChannels: 128, kernelSize: (3, 3), padding: (1, 1))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))
/// }
/// ```
///
/// ### ResNet-style Block (needs custom struct for skip connection)
///
/// ```swift
/// struct ResNetBlock: Layer {
///     var conv1: Conv2D
///     var conv2: Conv2D
///
///     init(channels: Int) {
///         conv1 = Conv2D(
///             kaimingUniformInChannels: channels,
///             outChannels: channels,
///             kernelSize: (3, 3),
///             padding: (1, 1)
///         )
///         conv2 = Conv2D(
///             kaimingUniformInChannels: channels,
///             outChannels: channels,
///             kernelSize: (3, 3),
///             padding: (1, 1)
///         )
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         let residual = input
///         var x = conv1(input).relu()
///         x = conv2(x)
///         return (x + residual).relu()
///     }
/// }
/// ```
///
/// ## Automatic Differentiation
///
/// Conv2D is fully differentiable with gradients computed for both weights and inputs:
///
/// ```swift
/// let conv = Conv2D(inChannels: 3, outChannels: 64, kernelSize: (3, 3))
/// let input = Tensor.randn([1, 3, 32, 32])
///
/// let (output, pullback) = valueWithPullback(at: conv, input) { c, x in
///     c(x)
/// }
///
/// let gradOutput = Tensor.ones([1, 64, 30, 30])
/// let (convGrad, inputGrad) = pullback(gradOutput)
/// // convGrad.weight: gradients for filter weights
/// // convGrad.bias: gradients for biases
/// // inputGrad: gradients w.r.t. input
/// ```
///
/// ## Performance Considerations
///
/// - **GPU Acceleration**: Convolutions are highly optimized on GPUs
/// - **Batch Size**: Larger batches improve GPU utilization
/// - **Channel Count**: Powers of 2 (32, 64, 128, 256) often perform better
/// - **Kernel Size**: 3x3 is most common, 1x1 for channel mixing
/// - **Memory**: Larger kernels and more channels increase memory usage
///
/// ## Topics
///
/// ### Creating Conv2D Layers
///
/// - ``init(inChannels:outChannels:kernelSize:stride:padding:dilation:groups:dtype:device:)``
/// - ``init(kaimingUniformInChannels:outChannels:kernelSize:stride:padding:dilation:groups:mode:nonlinearity:dtype:device:)``
///
/// ### Properties
///
/// - ``weight``
/// - ``bias``
/// - ``stride``
/// - ``padding``
/// - ``dilation``
/// - ``groups``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ### Initialization Options
///
/// - ``FanMode``
/// - ``KaimingNonlinearity``
///
/// ## See Also
///
/// - ``MaxPool2D`` - Max pooling for downsampling
/// - ``AvgPool2D`` - Average pooling
/// - ``BatchNorm`` - Batch normalization for conv layers
/// - ``Sequential`` - Compose layers into CNNs
/// - ``Flatten`` - Convert conv output to vector for dense layers
public struct Conv2D: Layer {
  /// The learnable convolutional filter weights.
  ///
  /// Shape: `[outputChannels, inputChannels/groups, kernelHeight, kernelWidth]`
  ///
  /// Each output channel has its own set of filters, one per input channel group.
  /// These weights are learned during training to detect specific patterns in the input.
  public var weight: Tensor

  /// The learnable bias terms, one per output channel.
  ///
  /// Shape: `[outputChannels]`
  ///
  /// Added to each output channel after the convolution operation.
  public var bias: Tensor

  /// The stride of the convolution: `(strideHeight, strideWidth)`.
  ///
  /// Controls how many pixels the filter moves at each step. Default is `(1, 1)`.
  /// Use `(2, 2)` for 2x spatial downsampling.
  @noDerivative public let stride: (Int, Int)

  /// The padding applied to the input: `(paddingHeight, paddingWidth)`.
  ///
  /// Zeros are added symmetrically around the input borders. Use `(1, 1)` with 3x3 kernels
  /// and stride `(1, 1)` to preserve spatial dimensions (same padding).
  @noDerivative public let padding: (Int, Int)

  /// The dilation (spacing) between kernel elements: `(dilationHeight, dilationWidth)`.
  ///
  /// Default is `(1, 1)` for standard convolutions. Use `(2, 2)` or higher for dilated
  /// (atrous) convolutions that increase the receptive field without adding parameters.
  @noDerivative public let dilation: (Int, Int)

  /// The number of blocked connections from input to output channels.
  ///
  /// - `groups = 1`: Standard convolution (default)
  /// - `groups = inputChannels`: Depthwise convolution (each input channel is convolved separately)
  /// - Other values: Grouped convolutions for efficiency
  ///
  /// `inputChannels` must be divisible by `groups`.
  @noDerivative public let groups: Int

  public typealias Input = Tensor
  public typealias Output = Tensor

  // Manual TangentVector to avoid nested synthesis issues.
  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var weight: Tensor
    public var bias: Tensor

    public init(weight: Tensor = Tensor(0), bias: Tensor = Tensor(0)) {
      self.weight = weight
      self.bias = bias
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight + rhs.weight, bias: lhs.bias + rhs.bias)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight - rhs.weight, bias: lhs.bias - rhs.bias)
    }
  }

  public mutating func move(by d: TangentVector) {
    weight += d.weight
    bias += d.bias
  }

  /// Creates a 2D convolutional layer with Glorot (Xavier) uniform initialization.
  ///
  /// Initializes weights using Glorot uniform initialization:
  /// `weight ~ U(-a, a)` where `a = sqrt(6 / (fan_in + fan_out))`
  ///
  /// This initialization works well with sigmoid/tanh activations. For ReLU activations,
  /// consider using ``init(kaimingUniformInChannels:outChannels:kernelSize:stride:padding:dilation:groups:mode:nonlinearity:dtype:device:)``
  /// instead.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels. Must be divisible by `groups`.
  ///   - outChannels: Number of output channels (number of filters).
  ///   - kernelSize: Size of the convolutional kernel as `(height, width)`.
  ///                 Common values: `(3, 3)`, `(5, 5)`, `(1, 1)`.
  ///   - stride: Stride of the convolution as `(strideHeight, strideWidth)`.
  ///             Defaults to `(1, 1)`. Use `(2, 2)` for downsampling.
  ///   - padding: Zero-padding added to both sides of the input as `(padHeight, padWidth)`.
  ///              Defaults to `(0, 0)` (valid convolution). Use `(1, 1)` for same padding
  ///              with 3x3 kernels.
  ///   - dilation: Spacing between kernel elements as `(dilationHeight, dilationWidth)`.
  ///               Defaults to `(1, 1)`. Higher values create dilated convolutions.
  ///   - groups: Number of blocked connections. Defaults to `1` (standard convolution).
  ///             Set to `inChannels` for depthwise convolution.
  ///   - dtype: Data type for weights and bias. Defaults to `.float32`.
  ///   - device: Device where tensors will be allocated. Defaults to `.cpu`.
  ///
  /// ```swift
  /// // Standard 3x3 convolution
  /// let conv = Conv2D(
  ///     inChannels: 3,
  ///     outChannels: 64,
  ///     kernelSize: (3, 3),
  ///     padding: (1, 1)
  /// )
  ///
  /// // Downsampling convolution
  /// let downsample = Conv2D(
  ///     inChannels: 64,
  ///     outChannels: 128,
  ///     kernelSize: (3, 3),
  ///     stride: (2, 2),
  ///     padding: (1, 1)
  /// )
  ///
  /// // 1x1 convolution for channel mixing
  /// let pointwise = Conv2D(
  ///     inChannels: 256,
  ///     outChannels: 128,
  ///     kernelSize: (1, 1)
  /// )
  /// ```
  ///
  /// - Precondition: `inChannels` must be divisible by `groups`.
  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: (Int, Int),
    stride: (Int, Int) = (1, 1),
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    groups: Int = 1,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    precondition(inChannels % groups == 0, "inChannels must be divisible by groups")
    let (kH, kW) = kernelSize
    let fanIn = inChannels / groups * kH * kW
    let fanOut = outChannels * kH * kW
    let a = Float((6.0 / Double(fanIn + fanOut)).squareRoot())

    self.weight = Tensor.uniform(
      low: -Double(a), high: Double(a),
      shape: [outChannels, inChannels / groups, kH, kW],
      dtype: dtype, device: device
    )
    self.bias = Tensor.zeros(shape: [outChannels], dtype: dtype, device: device)

    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
  }

  /// Applies the 2D convolution to the input tensor.
  ///
  /// Performs the convolution operation: slides the learned filters over the input spatial
  /// dimensions, computing weighted sums at each position and adding the bias.
  ///
  /// - Parameter x: Input tensor with shape `[batch, inputChannels, height, width]`.
  ///                The number of input channels must match the layer's `inChannels`.
  ///
  /// - Returns: Output tensor with shape `[batch, outputChannels, outputHeight, outputWidth]`.
  ///
  /// The output spatial dimensions are computed as:
  /// ```
  /// outputHeight = floor((height + 2*padding.0 - dilation.0*(kernelSize.0-1) - 1) / stride.0 + 1)
  /// outputWidth  = floor((width + 2*padding.1 - dilation.1*(kernelSize.1-1) - 1) / stride.1 + 1)
  /// ```
  ///
  /// ```swift
  /// let conv = Conv2D(
  ///     inChannels: 3,
  ///     outChannels: 64,
  ///     kernelSize: (3, 3),
  ///     stride: (1, 1),
  ///     padding: (1, 1)
  /// )
  ///
  /// // Single image
  /// let image = Tensor.randn([1, 3, 224, 224])
  /// let features = conv(image)  // Shape: [1, 64, 224, 224]
  ///
  /// // Batch of images
  /// let batch = Tensor.randn([32, 3, 224, 224])
  /// let batchFeatures = conv(batch)  // Shape: [32, 64, 224, 224]
  ///
  /// // In a CNN pipeline
  /// var x = Tensor.randn([16, 3, 32, 32])
  /// x = conv(x).relu()  // [16, 64, 32, 32] with ReLU activation
  /// ```
  ///
  /// - Note: This method is marked `@differentiable`, enabling automatic gradient computation
  ///         for both the layer's parameters and the input during backpropagation.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let s = withoutDerivative(at: [Int64(stride.0), Int64(stride.1)])
    let p = withoutDerivative(at: [Int64(padding.0), Int64(padding.1)])
    let d = withoutDerivative(at: [Int64(dilation.0), Int64(dilation.1)])
    let g = withoutDerivative(at: Int64(groups))
    return x.conv2d(weight: weight, bias: bias, stride: s, padding: p, dilation: d, groups: g)
  }
}

// Avoid “curried self” path using a free closure in the VJP.
extension Conv2D {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector))
  {
    func primal(_ s: Conv2D, _ i: Tensor) -> Tensor {
      let str = withoutDerivative(at: [Int64(s.stride.0), Int64(s.stride.1)])
      let pad = withoutDerivative(at: [Int64(s.padding.0), Int64(s.padding.1)])
      let dil = withoutDerivative(at: [Int64(s.dilation.0), Int64(s.dilation.1)])
      let grp = withoutDerivative(at: Int64(s.groups))
      return i.conv2d(
        weight: s.weight, bias: s.bias, stride: str, padding: pad, dilation: dil, groups: grp)
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

// MARK: - MaxPool2D

/// A 2D max pooling layer for downsampling spatial dimensions.
///
/// `MaxPool2D` reduces the spatial dimensions of feature maps by taking the maximum value
/// within each pooling window. It's commonly used in CNNs to reduce computational cost,
/// provide translation invariance, and increase the receptive field.
///
/// ## Overview
///
/// Max pooling slides a window over the input and outputs the maximum value from each window.
/// This operation:
/// - Reduces spatial dimensions (downsampling)
/// - Preserves the most prominent features
/// - Has no learnable parameters
/// - Provides mild translation invariance
///
/// ## Operation
///
/// For each position and each channel independently:
/// ```
/// output[b, c, h', w'] = max(input[b, c, h*stride+kh, w*stride+kw])
///                        for kh in [0, kernelHeight), kw in [0, kernelWidth)
/// ```
///
/// ## Creating MaxPool2D Layers
///
/// ```swift
/// // Standard 2x2 pooling (halves spatial dimensions)
/// let pool = MaxPool2D(kernelSize: (2, 2))
///
/// // 2x2 pooling with explicit stride
/// let pool = MaxPool2D(
///     kernelSize: (2, 2),
///     stride: (2, 2)  // Same as kernelSize by default
/// )
///
/// // 3x3 pooling with stride 2 (overlapping windows)
/// let pool = MaxPool2D(
///     kernelSize: (3, 3),
///     stride: (2, 2)
/// )
/// ```
///
/// ## Usage in CNNs
///
/// ```swift
/// // Classic CNN architecture
/// let cnn = Sequential {
///     Conv2D(inChannels: 3, outChannels: 64, kernelSize: (3, 3), padding: (1, 1))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))  // 224×224 → 112×112
///
///     Conv2D(inChannels: 64, outChannels: 128, kernelSize: (3, 3), padding: (1, 1))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))  // 112×112 → 56×56
///
///     Flatten()
///     Linear(inputSize: 128 * 56 * 56, outputSize: 1000)
/// }
/// ```
///
/// ## Shape Specifications
///
/// - **Input**: `[batch, channels, height, width]`
/// - **Output**: `[batch, channels, outputHeight, outputWidth]`
///
/// Number of channels remains unchanged. Output spatial dimensions:
/// ```
/// outputHeight = floor((height + 2*padding[0] - dilation[0]*(kernelSize[0]-1) - 1) / stride[0] + 1)
/// outputWidth  = floor((width + 2*padding[1] - dilation[1]*(kernelSize[1]-1) - 1) / stride[1] + 1)
/// ```
///
/// Or if `ceilMode = true`:
/// ```
/// outputHeight = ceil((height + 2*padding[0] - dilation[0]*(kernelSize[0]-1) - 1) / stride[0] + 1)
/// outputWidth  = ceil((width + 2*padding[1] - dilation[1]*(kernelSize[1]-1) - 1) / stride[1] + 1)
/// ```
///
/// ### Common Configurations
///
/// ```swift
/// // Standard 2x2 downsampling (most common)
/// let pool = MaxPool2D(kernelSize: (2, 2))
/// // Input [N, C, 32, 32] → Output [N, C, 16, 16]
///
/// // 3x3 with stride 2 (overlapping, more information preserved)
/// let pool = MaxPool2D(kernelSize: (3, 3), stride: (2, 2))
/// // Input [N, C, 32, 32] → Output [N, C, 15, 15]
/// ```
///
/// ## MaxPool vs AvgPool
///
/// **Use MaxPool when:**
/// - You want to preserve strong activations (sharp features)
/// - Building classification networks
/// - Following standard CNN architectures (AlexNet, VGG, ResNet)
///
/// **Use AvgPool when:**
/// - You want smoother downsampling
/// - Global pooling at the end of networks
/// - Segmentation tasks where spatial smoothness matters
///
/// ## Automatic Differentiation
///
/// MaxPool2D is differentiable. During backpropagation, gradients flow only to the
/// positions that contained the maximum values:
///
/// ```swift
/// let pool = MaxPool2D(kernelSize: (2, 2))
/// let input = Tensor.randn([1, 64, 32, 32])
///
/// let (output, pullback) = valueWithPullback(at: input) { x in
///     pool(x)
/// }
/// // Gradients route back to max positions
/// ```
///
/// ## Performance Considerations
///
/// - **No Parameters**: MaxPool has no learnable weights, making it very fast
/// - **GPU Friendly**: Highly optimized on all hardware
/// - **Memory**: Stores indices of max values for backprop
/// - **Receptive Field**: Increases receptive field of subsequent layers
///
/// ## Topics
///
/// ### Creating MaxPool2D Layers
///
/// - ``init(kernelSize:stride:padding:dilation:ceilMode:)``
///
/// ### Properties
///
/// - ``kernelSize``
/// - ``stride``
/// - ``padding``
/// - ``dilation``
/// - ``ceilMode``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``AvgPool2D`` - Average pooling alternative
/// - ``Conv2D`` - Convolutional layers
/// - ``Sequential`` - Compose layers into CNNs
public struct MaxPool2D: ParameterlessLayer {
  /// The size of the pooling window as `(height, width)`.
  ///
  /// Common values: `(2, 2)` for 2x downsampling, `(3, 3)` for overlapping windows.
  @noDerivative public let kernelSize: (Int, Int)

  /// The stride of the pooling window as `(strideHeight, strideWidth)`.
  ///
  /// Defaults to `kernelSize` if not specified. Use `(2, 2)` for non-overlapping
  /// 2x2 pooling.
  @noDerivative public let stride: (Int, Int)

  /// Zero-padding added to both sides of the input as `(padHeight, padWidth)`.
  ///
  /// Defaults to `(0, 0)`. Padding is rarely used with max pooling.
  @noDerivative public let padding: (Int, Int)

  /// The dilation (spacing) between elements in the pooling window.
  ///
  /// Defaults to `(1, 1)`. Dilation is rarely used with max pooling.
  @noDerivative public let dilation: (Int, Int)

  /// Whether to use `ceil` instead of `floor` when computing output dimensions.
  ///
  /// When `true`, uses ceiling division for output size calculation, potentially
  /// creating a slightly larger output. Defaults to `false`.
  @noDerivative public let ceilMode: Bool

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  /// Creates a 2D max pooling layer.
  ///
  /// - Parameters:
  ///   - kernelSize: The size of the pooling window as `(height, width)`.
  ///                 Common values: `(2, 2)`, `(3, 3)`.
  ///   - stride: The stride of the pooling operation as `(strideHeight, strideWidth)`.
  ///             If `nil` (default), uses the same value as `kernelSize` for non-overlapping pooling.
  ///   - padding: Zero-padding added to both sides as `(padHeight, padWidth)`.
  ///              Defaults to `(0, 0)`. Rarely used with max pooling.
  ///   - dilation: The spacing between pooling window elements as `(dilationHeight, dilationWidth)`.
  ///               Defaults to `(1, 1)`. Rarely used with max pooling.
  ///   - ceilMode: If `true`, uses ceiling division when computing output dimensions.
  ///               Defaults to `false` (floor division).
  ///
  /// ```swift
  /// // Standard 2x2 pooling (most common)
  /// let pool = MaxPool2D(kernelSize: (2, 2))
  ///
  /// // 3x3 with stride 2 (overlapping windows)
  /// let pool = MaxPool2D(
  ///     kernelSize: (3, 3),
  ///     stride: (2, 2)
  /// )
  ///
  /// // With padding (unusual but sometimes useful)
  /// let pool = MaxPool2D(
  ///     kernelSize: (3, 3),
  ///     stride: (2, 2),
  ///     padding: (1, 1)
  /// )
  /// ```
  public init(
    kernelSize: (Int, Int),
    stride: (Int, Int)? = nil,
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    ceilMode: Bool = false
  ) {
    self.kernelSize = kernelSize
    self.stride = stride ?? kernelSize
    self.padding = padding
    self.dilation = dilation
    self.ceilMode = ceilMode
  }

  /// Applies 2D max pooling to the input.
  ///
  /// Takes the maximum value within each pooling window for each channel independently.
  ///
  /// - Parameter x: Input tensor with shape `[batch, channels, height, width]`.
  ///
  /// - Returns: Output tensor with shape `[batch, channels, outputHeight, outputWidth]`.
  ///            The number of channels remains unchanged; only spatial dimensions are reduced.
  ///
  /// ```swift
  /// let pool = MaxPool2D(kernelSize: (2, 2))
  ///
  /// // Single image
  /// let image = Tensor.randn([1, 64, 32, 32])
  /// let pooled = pool(image)  // Shape: [1, 64, 16, 16]
  ///
  /// // Batch of feature maps
  /// let batch = Tensor.randn([32, 128, 56, 56])
  /// let batchPooled = pool(batch)  // Shape: [32, 128, 28, 28]
  ///
  /// // In a CNN
  /// var x = Tensor.randn([16, 64, 64, 64])
  /// x = conv(x).relu()
  /// x = pool(x)  // Downsample to [16, 64, 32, 32]
  /// ```
  ///
  /// - Note: This method is marked `@differentiable`. During backpropagation, gradients
  ///         are routed only to the positions that contained the maximum values.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let ks = withoutDerivative(at: [Int64(kernelSize.0), Int64(kernelSize.1)])
    let st = withoutDerivative(at: [Int64(stride.0), Int64(stride.1)])
    let pd = withoutDerivative(at: [Int64(padding.0), Int64(padding.1)])
    let dl = withoutDerivative(at: [Int64(dilation.0), Int64(dilation.1)])
    let cm = withoutDerivative(at: ceilMode)
    return x.maxPool2d(kernelSize: ks, stride: st, padding: pd, dilation: dl, ceilMode: cm)
  }
}

// MARK: - AvgPool2D

/// A 2D average pooling layer for smoother downsampling of spatial dimensions.
///
/// `AvgPool2D` reduces spatial dimensions by computing the average value within each pooling
/// window. It provides smoother downsampling compared to max pooling and is commonly used for
/// global pooling at the end of CNNs or in tasks requiring spatial smoothness.
///
/// ## Overview
///
/// Average pooling slides a window over the input and outputs the mean value from each window:
/// - Reduces spatial dimensions (downsampling)
/// - Provides smoother features than max pooling
/// - Has no learnable parameters
/// - Useful for global average pooling (GAP)
///
/// ## Operation
///
/// For each position and channel:
/// ```
/// output[b, c, h', w'] = mean(input[b, c, h*stride+kh, w*stride+kw])
///                        for kh in [0, kernelHeight), kw in [0, kernelWidth)
/// ```
///
/// ## Creating AvgPool2D Layers
///
/// ```swift
/// // Standard 2x2 average pooling
/// let pool = AvgPool2D(kernelSize: (2, 2))
///
/// // Global average pooling (common at end of CNNs)
/// let gap = AvgPool2D(kernelSize: (7, 7))  // For 7x7 feature maps
///
/// // Custom stride
/// let pool = AvgPool2D(
///     kernelSize: (3, 3),
///     stride: (2, 2)
/// )
/// ```
///
/// ## Usage Examples
///
/// ```swift
/// // In a classification network (replacing fully connected layers)
/// let cnn = Sequential {
///     Conv2D(inChannels: 512, outChannels: 1024, kernelSize: (3, 3), padding: (1, 1))
///     ReLU()
///     AvgPool2D(kernelSize: (7, 7))  // Global Average Pooling
///     // Output: [batch, 1024, 1, 1]
///     Flatten()
///     Linear(inputSize: 1024, outputSize: 1000)
/// }
///
/// // Smooth downsampling in segmentation
/// var x = Tensor.randn([8, 256, 64, 64])
/// x = AvgPool2D(kernelSize: (2, 2))(x)  // [8, 256, 32, 32]
/// ```
///
/// ## AvgPool vs MaxPool
///
/// **Use AvgPool when:**
/// - Global pooling at the end of networks (GAP)
/// - Segmentation or dense prediction tasks
/// - You want smoother, less sparse features
/// - Reducing parameters in place of fully connected layers
///
/// **Use MaxPool when:**
/// - Classification tasks (more common)
/// - Preserving strong activations is important
/// - Following standard architectures (VGG, ResNet uses max)
///
/// ## Shape Specifications
///
/// - **Input**: `[batch, channels, height, width]`
/// - **Output**: `[batch, channels, outputHeight, outputWidth]`
///
/// Output dimensions calculated the same as ``MaxPool2D``.
///
/// ## Global Average Pooling
///
/// A popular technique to replace fully connected layers:
///
/// ```swift
/// // Traditional approach (many parameters)
/// Flatten()
/// Linear(inputSize: 512 * 7 * 7, outputSize: 1000)  // 25M parameters
///
/// // Modern approach with GAP (no parameters)
/// AvgPool2D(kernelSize: (7, 7))  // [batch, 512, 1, 1]
/// Flatten()
/// Linear(inputSize: 512, outputSize: 1000)  // 512K parameters
/// ```
///
/// ## Topics
///
/// ### Creating AvgPool2D Layers
///
/// - ``init(kernelSize:stride:padding:ceilMode:)``
///
/// ### Properties
///
/// - ``kernelSize``
/// - ``stride``
/// - ``padding``
/// - ``ceilMode``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``MaxPool2D`` - Max pooling alternative
/// - ``Conv2D`` - Convolutional layers
/// - ``Flatten`` - Flatten pooled features
public struct AvgPool2D: ParameterlessLayer {
  /// The size of the pooling window as `(height, width)`.
  @noDerivative public let kernelSize: (Int, Int)

  /// The stride of the pooling window as `(strideHeight, strideWidth)`.
  ///
  /// Defaults to `kernelSize` if not specified.
  @noDerivative public let stride: (Int, Int)

  /// Zero-padding added to both sides of the input as `(padHeight, padWidth)`.
  @noDerivative public let padding: (Int, Int)

  /// Whether to use `ceil` instead of `floor` when computing output dimensions.
  @noDerivative public let ceilMode: Bool

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  /// Creates a 2D average pooling layer.
  ///
  /// - Parameters:
  ///   - kernelSize: The size of the pooling window as `(height, width)`.
  ///   - stride: The stride of the pooling operation. If `nil`, uses `kernelSize`.
  ///   - padding: Zero-padding added to both sides. Defaults to `(0, 0)`.
  ///   - ceilMode: If `true`, uses ceiling division for output dimensions. Defaults to `false`.
  ///
  /// ```swift
  /// // Standard 2x2 average pooling
  /// let pool = AvgPool2D(kernelSize: (2, 2))
  ///
  /// // Global average pooling
  /// let gap = AvgPool2D(kernelSize: (7, 7))
  /// ```
  public init(
    kernelSize: (Int, Int),
    stride: (Int, Int)? = nil,
    padding: (Int, Int) = (0, 0),
    ceilMode: Bool = false
  ) {
    self.kernelSize = kernelSize
    self.stride = stride ?? kernelSize
    self.padding = padding
    self.ceilMode = ceilMode
  }

  /// Applies 2D average pooling to the input.
  ///
  /// Computes the average value within each pooling window for each channel independently.
  ///
  /// - Parameter x: Input tensor with shape `[batch, channels, height, width]`.
  ///
  /// - Returns: Output tensor with shape `[batch, channels, outputHeight, outputWidth]`.
  ///
  /// ```swift
  /// let pool = AvgPool2D(kernelSize: (2, 2))
  /// let input = Tensor.randn([32, 128, 64, 64])
  /// let output = pool(input)  // Shape: [32, 128, 32, 32]
  ///
  /// // Global average pooling
  /// let gap = AvgPool2D(kernelSize: (8, 8))
  /// let features = Tensor.randn([16, 512, 8, 8])
  /// let pooled = gap(features)  // Shape: [16, 512, 1, 1]
  /// ```
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let ks = withoutDerivative(at: [Int64(kernelSize.0), Int64(kernelSize.1)])
    let st = withoutDerivative(at: [Int64(stride.0), Int64(stride.1)])
    let pd = withoutDerivative(at: [Int64(padding.0), Int64(padding.1)])
    let cm = withoutDerivative(at: ceilMode)
    return x.avgPool2d(kernelSize: ks, stride: st, padding: pd, ceilMode: cm)
  }
}

// MARK: - MaxPool2D VJP (non-recursive)
extension MaxPool2D {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (EmptyTangentVector, Tensor))
  {
    // <-- do not call s(i) here
    func primal(_ s: MaxPool2D, _ i: Tensor) -> Tensor {
      let ks = withoutDerivative(at: [Int64(s.kernelSize.0), Int64(s.kernelSize.1)])
      let st = withoutDerivative(at: [Int64(s.stride.0), Int64(s.stride.1)])
      let pd = withoutDerivative(at: [Int64(s.padding.0), Int64(s.padding.1)])
      let dl = withoutDerivative(at: [Int64(s.dilation.0), Int64(s.dilation.1)])
      let cm = withoutDerivative(at: s.ceilMode)
      return i.maxPool2d(kernelSize: ks, stride: st, padding: pd, dilation: dl, ceilMode: cm)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (EmptyTangentVector(), dx)
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> EmptyTangentVector)
  {
    let (y, _) = _vjpCallAsFunction(x)
    return (y, { _ in EmptyTangentVector() })
  }
}

// MARK: - AvgPool2D VJP (non-recursive)
extension AvgPool2D {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (EmptyTangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: AvgPool2D, _ i: Tensor) -> Tensor {
      let ks = withoutDerivative(at: [Int64(s.kernelSize.0), Int64(s.kernelSize.1)])
      let st = withoutDerivative(at: [Int64(s.stride.0), Int64(s.stride.1)])
      let pd = withoutDerivative(at: [Int64(s.padding.0), Int64(s.padding.1)])
      let cm = withoutDerivative(at: s.ceilMode)
      return i.avgPool2d(kernelSize: ks, stride: st, padding: pd, ceilMode: cm)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (EmptyTangentVector(), dx)
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor.TangentVector) -> EmptyTangentVector)
  {
    let (y, _) = _vjpCallAsFunction(x)
    return (y, { _ in EmptyTangentVector() })
  }
}

// MARK: - Flatten

/// Flattens contiguous dimensions of a tensor into a single dimension.
///
/// `Flatten` is a utility layer that reshapes multi-dimensional tensors into vectors or
/// lower-dimensional tensors. It's essential for connecting convolutional layers to fully
/// connected layers in CNNs.
///
/// ## Overview
///
/// Flatten combines multiple dimensions into one while preserving the data order. It's most
/// commonly used to convert spatial feature maps into vectors for classification heads.
///
/// ## Creating Flatten Layers
///
/// ```swift
/// // Standard flatten (preserves batch dimension)
/// let flatten = Flatten()  // Flattens from dim 1 onwards
///
/// // Custom range
/// let flatten = Flatten(startDim: 2, endDim: 4)
///
/// // Flatten everything
/// let flatten = Flatten(startDim: 0, endDim: -1)
/// ```
///
/// ## Usage in CNNs
///
/// ```swift
/// // Classic CNN to fully connected transition
/// let model = Sequential {
///     Conv2D(inChannels: 64, outChannels: 128, kernelSize: (3, 3))
///     ReLU()
///     MaxPool2D(kernelSize: (2, 2))
///     // Output shape: [batch, 128, 14, 14]
///
///     Flatten()  // Flattens to [batch, 128*14*14] = [batch, 25088]
///
///     Linear(inputSize: 25088, outputSize: 512)
///     ReLU()
///     Linear(inputSize: 512, outputSize: 10)
/// }
/// ```
///
/// ## Shape Transformations
///
/// ```swift
/// let flatten = Flatten()
///
/// // CNN features to vector
/// let features = Tensor.randn([32, 128, 7, 7])  // [batch, channels, h, w]
/// let vector = flatten(features)  // [32, 6272] (128*7*7)
///
/// // After global pooling
/// let pooled = Tensor.randn([32, 512, 1, 1])
/// let flat = flatten(pooled)  // [32, 512]
/// ```
///
/// ## Default Behavior
///
/// By default, `Flatten()` starts from dimension 1, preserving the batch dimension:
///
/// - Input: `[batch, d1, d2, ..., dn]`
/// - Output: `[batch, d1 * d2 * ... * dn]`
///
/// This is the most common use case in neural networks.
///
/// ## Custom Ranges
///
/// You can flatten specific dimension ranges:
///
/// ```swift
/// // Flatten spatial dimensions only
/// let flatten = Flatten(startDim: 2, endDim: 3)
/// let input = Tensor.randn([8, 64, 7, 7])
/// let output = flatten(input)  // [8, 64, 49]
///
/// // Flatten all dimensions
/// let flattenAll = Flatten(startDim: 0, endDim: -1)
/// let input = Tensor.randn([2, 3, 4, 5])
/// let output = flattenAll(input)  // [120]
/// ```
///
/// ## Flatten vs Reshape
///
/// - **Flatten**: Convenient layer for sequential models
/// - **Reshape**: More general operation for arbitrary shape changes
///
/// ```swift
/// // Using Flatten in Sequential
/// Sequential {
///     Conv2D(...)
///     Flatten()
///     Linear(...)
/// }
///
/// // Using reshape directly
/// let x = convOutput.reshaped(to: [batchSize, -1])
/// ```
///
/// ## Topics
///
/// ### Creating Flatten Layers
///
/// - ``init(startDim:endDim:)``
///
/// ### Properties
///
/// - ``startDim``
/// - ``endDim``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``Conv2D`` - Convolutional layers producing spatial features
/// - ``MaxPool2D`` - Pooling before flattening
/// - ``Linear`` - Fully connected layers after flattening
/// - ``Sequential`` - Compose layers including Flatten
public struct Flatten: ParameterlessLayer {
  /// The first dimension to flatten (inclusive).
  ///
  /// Default is `1`, preserving the batch dimension.
  @noDerivative public let startDim: Int

  /// The last dimension to flatten (inclusive).
  ///
  /// Default is `-1` (last dimension). Negative indices count from the end.
  @noDerivative public let endDim: Int

  public typealias Input = Tensor
  public typealias Output = Tensor
  public typealias TangentVector = EmptyTangentVector

  /// Creates a flatten layer.
  ///
  /// - Parameters:
  ///   - startDim: The first dimension to flatten (inclusive). Defaults to `1` to preserve batch dimension.
  ///   - endDim: The last dimension to flatten (inclusive). Defaults to `-1` (last dimension).
  ///
  /// ```swift
  /// // Standard flatten (preserves batch)
  /// let flatten = Flatten()
  /// // [batch, c, h, w] → [batch, c*h*w]
  ///
  /// // Flatten spatial dimensions only
  /// let flatten = Flatten(startDim: 2, endDim: 3)
  /// // [batch, channels, h, w] → [batch, channels, h*w]
  ///
  /// // Flatten everything
  /// let flatten = Flatten(startDim: 0, endDim: -1)
  /// // [d1, d2, d3] → [d1*d2*d3]
  /// ```
  public init(startDim: Int = 1, endDim: Int = -1) {
    self.startDim = startDim
    self.endDim = endDim
  }

  /// Applies the flattening operation to the input.
  ///
  /// Combines dimensions from `startDim` to `endDim` (inclusive) into a single dimension.
  ///
  /// - Parameter x: Input tensor to flatten.
  ///
  /// - Returns: Flattened tensor with specified dimensions combined.
  ///
  /// ```swift
  /// let flatten = Flatten()
  ///
  /// // Typical CNN usage
  /// let features = Tensor.randn([32, 128, 7, 7])
  /// let vector = flatten(features)  // [32, 6272]
  ///
  /// // After global pooling
  /// let pooled = Tensor.randn([16, 512, 1, 1])
  /// let flat = flatten(pooled)  // [16, 512]
  /// ```
  ///
  /// - Note: This operation is fully differentiable, preserving gradients through the reshape.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    x.flattened(startDim: startDim, endDim: endDim)
  }
}

extension Conv2D {
  /// Specifies which fan to use for computing variance in Kaiming initialization.
  ///
  /// In Kaiming (He) initialization, the variance of weights depends on either the number
  /// of input connections (fan-in) or output connections (fan-out).
  ///
  /// - `fanIn`: Recommended for most cases, especially when using ReLU activations
  /// - `fanOut`: Alternative mode, less commonly used
  public enum FanMode {
    /// Use fan-in (number of input connections) for variance calculation.
    ///
    /// This is the recommended and default mode for ReLU-based networks.
    case fanIn

    /// Use fan-out (number of output connections) for variance calculation.
    case fanOut
  }

  /// Specifies the activation function type for computing the Kaiming gain factor.
  ///
  /// The gain factor adjusts the initialization variance based on the activation function's
  /// properties to maintain stable gradient flow during training.
  ///
  /// ## Gain Values
  ///
  /// - ReLU: gain = √2 ≈ 1.414
  /// - Leaky ReLU: gain = √(2 / (1 + negative_slope²))
  /// - Linear: gain = 1
  public enum KaimingNonlinearity: Equatable {
    /// ReLU activation function. Gain = √2.
    ///
    /// Use this when your conv layer is followed by ReLU activation (most common).
    case relu

    /// Leaky ReLU activation with specified negative slope.
    ///
    /// Use this when your conv layer is followed by Leaky ReLU. Common negative
    /// slopes are 0.01 or 0.2.
    ///
    /// - Parameter negativeSlope: The slope for negative values (typically 0.01).
    case leakyReLU(negativeSlope: Float)

    /// Linear (identity) activation. Gain = 1.
    ///
    /// Use this when there's no activation or for activations like tanh/sigmoid.
    case linear
  }

  @inlinable
  internal static func _kaimingGain(_ nl: KaimingNonlinearity) -> Float {
    switch nl {
    case .relu:
      return Float(2).squareRoot()  // √2
    case .leakyReLU(let a):
      // √(2 / (1 + a^2))
      let denom = 1.0 + Double(a) * Double(a)
      return Float((2.0 / denom).squareRoot())
    case .linear:
      return 1
    }
  }

  /// Creates a 2D convolutional layer with Kaiming (He) uniform initialization.
  ///
  /// Kaiming initialization is designed specifically for ReLU-based networks and helps prevent
  /// vanishing/exploding gradients by considering the activation function's properties.
  ///
  /// ## Initialization Formula
  ///
  /// Weights are sampled from a uniform distribution:
  /// ```
  /// weight ~ U(-bound, bound)
  /// where bound = √3 * std
  /// and std = gain / √fan
  /// ```
  ///
  /// The gain depends on the activation function:
  /// - ReLU: gain = √2
  /// - Leaky ReLU: gain = √(2 / (1 + negative_slope²))
  /// - Linear: gain = 1
  ///
  /// ## When to Use
  ///
  /// **Use Kaiming initialization when:**
  /// - Using ReLU or Leaky ReLU activations (recommended)
  /// - Building deep CNNs (ResNet, VGG, etc.)
  /// - Training from scratch
  ///
  /// **Use Glorot initialization when:**
  /// - Using sigmoid or tanh activations
  /// - Shallow networks
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels. Must be divisible by `groups`.
  ///   - outChannels: Number of output channels (number of filters).
  ///   - kernelSize: Size of the convolutional kernel as `(height, width)`.
  ///   - stride: Stride of the convolution. Defaults to `(1, 1)`.
  ///   - padding: Zero-padding added to input. Defaults to `(0, 0)`.
  ///   - dilation: Spacing between kernel elements. Defaults to `(1, 1)`.
  ///   - groups: Number of blocked connections. Defaults to `1`.
  ///   - mode: Which fan to use for variance calculation. Defaults to `.fanIn` (recommended).
  ///   - nonlinearity: The activation function following this layer. Defaults to `.relu`.
  ///                   Use `.leakyReLU(negativeSlope: 0.01)` for Leaky ReLU.
  ///   - dtype: Data type for weights and bias. Defaults to `.float32`.
  ///   - device: Device where tensors will be allocated. Defaults to `.cpu`.
  ///
  /// ```swift
  /// // Standard ReLU-based conv (most common)
  /// let conv = Conv2D(
  ///     kaimingUniformInChannels: 3,
  ///     outChannels: 64,
  ///     kernelSize: (3, 3),
  ///     padding: (1, 1),
  ///     nonlinearity: .relu
  /// )
  ///
  /// // For Leaky ReLU activation
  /// let leakyConv = Conv2D(
  ///     kaimingUniformInChannels: 64,
  ///     outChannels: 128,
  ///     kernelSize: (3, 3),
  ///     padding: (1, 1),
  ///     nonlinearity: .leakyReLU(negativeSlope: 0.2)
  /// )
  ///
  /// // ResNet-style initialization
  /// let resnetConv = Conv2D(
  ///     kaimingUniformInChannels: 64,
  ///     outChannels: 64,
  ///     kernelSize: (3, 3),
  ///     padding: (1, 1),
  ///     mode: .fanIn,
  ///     nonlinearity: .relu
  /// )
  /// ```
  ///
  /// - Precondition: `inChannels` must be divisible by `groups`.
  ///
  /// ## See Also
  ///
  /// - ``init(inChannels:outChannels:kernelSize:stride:padding:dilation:groups:dtype:device:)`` - Glorot initialization
  /// - ``FanMode`` - Fan mode selection
  /// - ``KaimingNonlinearity`` - Activation function specification
  public init(
    kaimingUniformInChannels inChannels: Int,
    outChannels: Int,
    kernelSize: (Int, Int),
    stride: (Int, Int) = (1, 1),
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    groups: Int = 1,
    mode: FanMode = .fanIn,
    nonlinearity: KaimingNonlinearity = .relu,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    precondition(inChannels % groups == 0, "inChannels must be divisible by groups")

    let (kH, kW) = kernelSize
    let fanIn = (inChannels / groups) * kH * kW
    let fanOut = outChannels * kH * kW
    let fan = (mode == .fanIn) ? fanIn : fanOut

    // PyTorch-style Kaiming uniform: bound = √3 * std, std = gain / √fan
    let gain = Double(Self._kaimingGain(nonlinearity))
    let std = gain / Double(fan).squareRoot()
    let bound = 3.0.squareRoot() * std

    self.weight = Tensor.uniform(
      low: -bound, high: bound,
      shape: [outChannels, inChannels / groups, kH, kW],
      dtype: dtype, device: device
    )

    // Common practice: bias ~ U[-1/√fan_in, 1/√fan_in]
    let bBound = 1.0 / Double(fanIn).squareRoot()
    self.bias = Tensor.uniform(
      low: -bBound, high: bBound, shape: [outChannels], dtype: dtype, device: device)

    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
  }
}
