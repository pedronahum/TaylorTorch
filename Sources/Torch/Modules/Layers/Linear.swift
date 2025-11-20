// Sources/Torch/Modules/Linear.swift
import Foundation
import _Differentiation

/// A fully connected layer that applies an affine transformation to the input.
///
/// `Linear` performs the operation `y = xW + b`, where `x` is the input, `W` is the weight matrix,
/// and `b` is the bias vector. This is one of the most fundamental building blocks in neural networks,
/// also known as a fully connected layer or dense layer.
///
/// ## Overview
///
/// Linear layers transform input features into output features through a learned linear mapping.
/// They're used throughout deep learning architectures:
/// - As hidden layers in multi-layer perceptrons (MLPs)
/// - As output layers for classification and regression
/// - In attention mechanisms for query/key/value projections
/// - In transformers and language models
///
/// ## Mathematical Operation
///
/// For input tensor `x` of shape `[batch, inputSize]`:
///
/// ```
/// y = xW + b
/// ```
///
/// Where:
/// - `W` is the weight matrix of shape `[inputSize, outputSize]`
/// - `b` is the bias vector of shape `[outputSize]`
/// - `y` is the output of shape `[batch, outputSize]`
///
/// ## Creating a Linear Layer
///
/// ```swift
/// // Simple hidden layer
/// let hidden = Linear(inputSize: 784, outputSize: 256)
///
/// // Output layer for 10-class classification
/// let output = Linear(inputSize: 256, outputSize: 10)
///
/// // Layer on GPU
/// let gpuLayer = Linear(inputSize: 512, outputSize: 512, device: .cuda(0))
/// ```
///
/// ## Using in a Network
///
/// ```swift
/// let model = Sequential {
///     Linear(inputSize: 784, outputSize: 512)
///     ReLU()
///     Dropout(probability: 0.2)
///     Linear(inputSize: 512, outputSize: 256)
///     ReLU()
///     Linear(inputSize: 256, outputSize: 10)
/// }
///
/// let input = Tensor.randn([32, 784])  // Batch of 32 samples
/// let output = model(input)  // Shape: [32, 10]
/// ```
///
/// ## Weight Initialization
///
/// By default, weights are initialized using Glorot (Xavier) uniform initialization,
/// which helps maintain gradient flow in deep networks:
///
/// ```swift
/// // Weights sampled from U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
/// let layer = Linear(inputSize: 100, outputSize: 50)
/// ```
///
/// The bias is initialized to zeros.
///
/// ## Shape Specifications
///
/// - **Input**: `[batch, inputSize]` or `[..., inputSize]` (supports arbitrary leading dimensions)
/// - **Weight**: `[inputSize, outputSize]`
/// - **Bias**: `[outputSize]`
/// - **Output**: `[batch, outputSize]` or `[..., outputSize]`
///
/// ```swift
/// let layer = Linear(inputSize: 128, outputSize: 64)
///
/// // Works with different batch sizes
/// let batch32 = Tensor.randn([32, 128])
/// let out32 = layer(batch32)  // [32, 64]
///
/// let batch1 = Tensor.randn([1, 128])
/// let out1 = layer(batch1)  // [1, 64]
///
/// // Works with extra dimensions (e.g., sequences)
/// let sequences = Tensor.randn([16, 50, 128])  // [batch, seq_len, features]
/// let seqOut = layer(sequences)  // [16, 50, 64]
/// ```
///
/// ## Automatic Differentiation
///
/// Linear layers are fully differentiable and integrate seamlessly with Swift's autodiff:
///
/// ```swift
/// let layer = Linear(inputSize: 10, outputSize: 5)
/// let input = Tensor.randn([1, 10])
///
/// // Compute gradients
/// let (output, pullback) = valueWithPullback(at: layer, input) { l, x in
///     l(x)
/// }
///
/// let gradOutput = Tensor.ones([1, 5])
/// let (layerGrad, inputGrad) = pullback(gradOutput)
///
/// // layerGrad contains gradients for weight and bias
/// // inputGrad contains gradients w.r.t. input
/// ```
///
/// ## Common Patterns
///
/// ### Multi-Layer Perceptron (MLP)
///
/// ```swift
/// struct MLP: Layer {
///     var fc1: Linear
///     var fc2: Linear
///     var fc3: Linear
///     var dropout: Dropout
///
///     init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
///         self.fc1 = Linear(inputSize: inputDim, outputSize: hiddenDim)
///         self.fc2 = Linear(inputSize: hiddenDim, outputSize: hiddenDim)
///         self.fc3 = Linear(inputSize: hiddenDim, outputSize: outputDim)
///         self.dropout = Dropout(probability: 0.5)
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         var x = input
///         x = fc1(x).relu()
///         x = dropout(x)
///         x = fc2(x).relu()
///         x = dropout(x)
///         x = fc3(x)
///         return x
///     }
/// }
/// ```
///
/// ### Attention Projection
///
/// ```swift
/// // Query, Key, Value projections in attention
/// let queryProj = Linear(inputSize: 512, outputSize: 512)
/// let keyProj = Linear(inputSize: 512, outputSize: 512)
/// let valueProj = Linear(inputSize: 512, outputSize: 512)
///
/// let embeddings = Tensor.randn([32, 100, 512])  // [batch, seq, features]
/// let query = queryProj(embeddings)
/// let key = keyProj(embeddings)
/// let value = valueProj(embeddings)
/// ```
///
/// ## Performance Considerations
///
/// - Linear layers are highly optimized using BLAS/GEMM operations
/// - GPU acceleration provides significant speedup for large matrices
/// - Consider using multiple smaller layers instead of one very large layer
/// - Batch inputs together for better throughput
///
/// ## Topics
///
/// ### Creating a Linear Layer
///
/// - ``init(inputSize:outputSize:dtype:device:)``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ### Layer Properties
///
/// - ``weight``
/// - ``bias``
///
/// ## See Also
///
/// - ``Dense`` - Linear layer with activation function
/// - ``Sequential`` - Compose multiple layers
/// - ``Conv2D`` - Convolutional layers for spatial data
/// - ``MultiHeadAttention`` - Attention mechanism using linear projections
public struct Linear: Layer {
  /// The learnable weight matrix of shape `[inputSize, outputSize]`.
  ///
  /// Initialized using Glorot (Xavier) uniform initialization by default.
  /// During the forward pass, the input is matrix-multiplied with this weight.
  public var weight: Tensor  // [in, out]

  /// The learnable bias vector of shape `[outputSize]`.
  ///
  /// Initialized to zeros by default. Added to the result after matrix multiplication.
  public var bias: Tensor  // [out]

  // Layer signatures
  public typealias Input = Tensor
  public typealias Output = Tensor

  // MARK: - Manual TangentVector (avoid synthesis pitfalls)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // explicit zero/+/-
    KeyPathIterable,  // required by Module
    VectorProtocol,  // required by Module
    PointwiseMultiplicative  // required by Module
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

  // Required when manually defining TangentVector
  public mutating func move(by d: TangentVector) {
    weight += d.weight
    bias += d.bias
  }

  // MARK: - Initializers

  /// Creates a linear layer with the specified input and output dimensions.
  ///
  /// Weights are initialized using Glorot (Xavier) uniform initialization, which samples from
  /// a uniform distribution `U(-a, a)` where `a = sqrt(6 / (inputSize + outputSize))`.
  /// This initialization helps maintain roughly equal variance of activations and gradients
  /// across layers.
  ///
  /// - Parameters:
  ///   - inputSize: The number of input features. This determines the second-to-last dimension
  ///                of the input tensor.
  ///   - outputSize: The number of output features. This determines the last dimension of the
  ///                 output tensor.
  ///   - dtype: The data type for weights and biases. Defaults to `.float32`.
  ///   - device: The device where the layer's parameters reside. Defaults to `.cpu`.
  ///
  /// - Returns: A new `Linear` layer with initialized weights and biases.
  ///
  /// ```swift
  /// // Create a linear layer for MNIST (784 input pixels → 128 hidden units)
  /// let layer = Linear(inputSize: 784, outputSize: 128)
  ///
  /// // Create on GPU with float16 precision
  /// let gpuLayer = Linear(
  ///     inputSize: 512,
  ///     outputSize: 256,
  ///     dtype: .float16,
  ///     device: .cuda(0)
  /// )
  /// ```
  ///
  /// - Note: Biases are initialized to zeros, which is standard practice for most architectures.
  public init(
    inputSize inFeatures: Int,
    outputSize outFeatures: Int,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    let a = (6.0 / Double(inFeatures + outFeatures)).squareRoot()  // Glorot
    self.weight = Tensor.uniform(
      low: -a, high: a, shape: [inFeatures, outFeatures], dtype: dtype, device: device)
    self.bias = Tensor.zeros(shape: [outFeatures], dtype: dtype, device: device)
  }

  // MARK: - Forward

  /// Applies the linear transformation to the input.
  ///
  /// Performs the operation `y = xW + b` where the input is matrix-multiplied with the weight
  /// matrix and the bias is added.
  ///
  /// - Parameter x: The input tensor of shape `[..., inputSize]`. The last dimension must match
  ///                the `inputSize` specified during initialization. Can have any number of
  ///                leading batch dimensions.
  ///
  /// - Returns: The output tensor of shape `[..., outputSize]`, with the same leading dimensions
  ///            as the input.
  ///
  /// ```swift
  /// let layer = Linear(inputSize: 128, outputSize: 64)
  ///
  /// // Single sample
  /// let input = Tensor.randn([128])
  /// let output = layer(input)  // Shape: [64]
  ///
  /// // Batch of samples
  /// let batch = Tensor.randn([32, 128])
  /// let batchOutput = layer(batch)  // Shape: [32, 64]
  ///
  /// // Sequence of features
  /// let sequence = Tensor.randn([16, 100, 128])  // [batch, seq_len, features]
  /// let seqOutput = layer(sequence)  // Shape: [16, 100, 64]
  /// ```
  ///
  /// - Note: This method is marked `@differentiable`, enabling automatic gradient computation
  ///         for backpropagation.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    x.matmul(weight).adding(bias)
  }
}

// MARK: - Manual derivatives (avoid curried-self path)
extension Linear {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: Linear, _ i: Tensor) -> Tensor {
      i.matmul(s.weight).adding(s.bias)
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
