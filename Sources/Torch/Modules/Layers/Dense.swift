// Sources/Torch/Modules/Dense.swift
import Foundation
import _Differentiation

/// Public identity activation that’s safe to use across module boundaries.
@differentiable(reverse)
public func identityActivation(_ x: Tensor) -> Tensor { x }

@derivative(of: identityActivation)
public func _vjpIdentityActivation(_ x: Tensor)
  -> (value: Tensor, pullback: (Tensor) -> Tensor)
{
  (x, { v in v })
}

/// A fully connected layer with an integrated activation function.
///
/// `Dense` combines a ``Linear`` layer with an activation function into a single component.
/// This is a convenience layer that fuses the linear transformation and non-linearity,
/// making network definitions more concise.
///
/// ## Overview
///
/// Dense layers are the building blocks of feedforward neural networks. Each Dense layer:
/// 1. Applies a linear transformation: `z = xW + b`
/// 2. Applies an activation function: `y = activation(z)`
///
/// ## Creating Dense Layers
///
/// ```swift
/// // With custom activation (identity by default)
/// let dense = Dense(
///     inputSize: 784,
///     outputSize: 256,
///     activation: { $0.relu() }
/// )
///
/// // Using convenience factories
/// let reluLayer = Dense.relu(inputSize: 784, outputSize: 256)
/// let tanhLayer = Dense.tanh(inputSize: 128, outputSize: 64)
/// let sigmoidLayer = Dense.sigmoid(inputSize: 64, outputSize: 1)
/// ```
///
/// ## Usage in Networks
///
/// ```swift
/// // Building a classifier with Dense layers
/// struct Classifier: Layer {
///     var dense1: Dense
///     var dense2: Dense
///     var dense3: Dense
///     var dropout: Dropout
///
///     init() {
///         dense1 = Dense.relu(inputSize: 784, outputSize: 512)
///         dense2 = Dense.relu(inputSize: 512, outputSize: 256)
///         dense3 = Dense(inputSize: 256, outputSize: 10)  // No activation
///         dropout = Dropout(probability: 0.3)
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         var x = input
///         x = dense1(x)
///         x = dropout(x)
///         x = dense2(x)
///         x = dropout(x)
///         x = dense3(x)
///         return x
///     }
/// }
/// ```
///
/// ## Dense vs Linear
///
/// - **Linear**: Just the affine transformation `y = xW + b`
/// - **Dense**: Linear + activation function
///
/// ```swift
/// // These are equivalent:
///
/// // Using Linear + separate activation
/// let output1 = ReLU()(Linear(inputSize: 100, outputSize: 50)(input))
///
/// // Using Dense with activation
/// let output2 = Dense.relu(inputSize: 100, outputSize: 50)(input)
/// ```
///
/// ## Convenience Factories
///
/// Dense provides static methods for common activations:
///
/// ```swift
/// // ReLU activation
/// let relu = Dense.relu(inputSize: 128, outputSize: 64)
///
/// // Tanh activation
/// let tanh = Dense.tanh(inputSize: 64, outputSize: 32)
///
/// // Sigmoid activation (for binary classification)
/// let sigmoid = Dense.sigmoid(inputSize: 32, outputSize: 1)
/// ```
///
/// ## Custom Activations
///
/// You can use any differentiable function as the activation:
///
/// ```swift
/// // GELU activation
/// let geluDense = Dense(
///     inputSize: 512,
///     outputSize: 512,
///     activation: { $0.gelu() }
/// )
///
/// // SiLU/Swish activation
/// let siluDense = Dense(
///     inputSize: 256,
///     outputSize: 128,
///     activation: { $0 * $0.sigmoid() }
/// )
///
/// // Custom activation
/// let customDense = Dense(
///     inputSize: 100,
///     outputSize: 50,
///     activation: { x in x.relu().clamp(max: 6.0) }  // ReLU6
/// )
/// ```
///
/// ## Topics
///
/// ### Creating Dense Layers
///
/// - ``init(inputSize:outputSize:activation:dtype:device:)``
/// - ``relu(inputSize:outputSize:dtype:device:)``
/// - ``tanh(inputSize:outputSize:dtype:device:)``
/// - ``sigmoid(inputSize:outputSize:dtype:device:)``
///
/// ### Properties
///
/// - ``linear``
/// - ``activation``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ## See Also
///
/// - ``Linear`` - Linear layer without activation
/// - ``ReLU`` - Standalone ReLU activation
/// - ``Sequential`` - Compose multiple layers
public struct Dense: Layer {
  /// The underlying linear transformation layer.
  ///
  /// This performs the affine transformation `z = xW + b` before the activation is applied.
  public var linear: Linear

  /// The activation function applied after the linear transformation.
  ///
  /// This closure is marked `@noDerivative` because it's not a learnable parameter,
  /// but gradients still flow through it during backpropagation.
  @noDerivative public var activation: @differentiable(reverse) (Tensor) -> Tensor

  public typealias Input = Tensor
  public typealias Output = Tensor

  // MARK: - Manual TangentVector
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var linear: Linear.TangentVector

    public init(linear: Linear.TangentVector = .zero) {
      self.linear = linear
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(linear: lhs.linear + rhs.linear)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(linear: lhs.linear - rhs.linear)
    }

    // VectorProtocol (scalar ops) – delegate to the nested vector
    public func adding(_ x: Float) -> Self { .init(linear: linear.adding(x)) }
    public func subtracting(_ x: Float) -> Self { .init(linear: linear.subtracting(x)) }
    public func scaled(by s: Float) -> Self { .init(linear: linear.scaled(by: s)) }

    // PointwiseMultiplicative (.* / one / reciprocal) – delegate
    public static var one: Self { .init(linear: .one) }
    public var reciprocal: Self { .init(linear: linear.reciprocal) }
    public static func .* (lhs: Self, rhs: Self) -> Self { .init(linear: lhs.linear .* rhs.linear) }
  }

  public mutating func move(by d: TangentVector) {
    linear.move(by: d.linear)
  }

  // MARK: - Initializers

  /// Creates a dense layer with a custom activation function.
  ///
  /// The linear weights are initialized using Glorot (Xavier) uniform initialization.
  ///
  /// - Parameters:
  ///   - inputSize: The number of input features.
  ///   - outputSize: The number of output features.
  ///   - activation: The activation function to apply after the linear transformation.
  ///                Defaults to identity (no activation).
  ///   - dtype: The data type for weights and biases. Defaults to `.float32`.
  ///   - device: The device where the layer's parameters reside. Defaults to `.cpu`.
  ///
  /// ```swift
  /// // With ReLU activation
  /// let layer = Dense(
  ///     inputSize: 784,
  ///     outputSize: 256,
  ///     activation: { $0.relu() }
  /// )
  ///
  /// // With identity (no activation)
  /// let linearOnly = Dense(inputSize: 128, outputSize: 64)
  ///
  /// // With custom activation
  /// let custom = Dense(
  ///     inputSize: 256,
  ///     outputSize: 128,
  ///     activation: { $0.gelu() }
  /// )
  /// ```
  public init(
    inputSize inFeatures: Int,
    outputSize outFeatures: Int,
    activation: @escaping @differentiable(reverse) (Tensor) -> Tensor = identityActivation,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.linear = Linear(
      inputSize: inFeatures, outputSize: outFeatures, dtype: dtype, device: device)
    self.activation = activation
  }

  // MARK: - Convenience Factories

  /// Creates a dense layer with tanh activation.
  ///
  /// Tanh is a classic activation function that outputs values in the range (-1, 1).
  /// It's zero-centered, which can help with gradient flow in some architectures.
  ///
  /// - Parameters:
  ///   - inputSize: The number of input features.
  ///   - outputSize: The number of output features.
  ///   - dtype: The data type for weights and biases. Defaults to `.float32`.
  ///   - device: The device where the layer's parameters reside. Defaults to `.cpu`.
  ///
  /// - Returns: A `Dense` layer with tanh activation.
  ///
  /// ```swift
  /// // Hidden layer with tanh activation
  /// let hidden = Dense.tanh(inputSize: 256, outputSize: 128)
  ///
  /// // Common in older RNN architectures
  /// let rnnLayer = Dense.tanh(inputSize: 512, outputSize: 512)
  /// ```
  ///
  /// ## See Also
  /// - ``Tanh`` - Standalone tanh activation layer
  /// - ``sigmoid(inputSize:outputSize:dtype:device:)`` - Sigmoid activation variant
  public static func tanh(
    inputSize: Int, outputSize: Int, dtype: DType = .float32, device: Device = .cpu
  ) -> Dense {
    Dense(
      inputSize: inputSize, outputSize: outputSize, activation: { $0.tanh() }, dtype: dtype,
      device: device)
  }

  /// Creates a dense layer with sigmoid activation.
  ///
  /// Sigmoid activation squashes outputs to the range (0, 1), making it ideal for binary
  /// classification tasks or when you need outputs to represent probabilities.
  ///
  /// - Parameters:
  ///   - inputSize: The number of input features.
  ///   - outputSize: The number of output features.
  ///   - dtype: The data type for weights and biases. Defaults to `.float32`.
  ///   - device: The device where the layer's parameters reside. Defaults to `.cpu`.
  ///
  /// - Returns: A `Dense` layer with sigmoid activation.
  ///
  /// ```swift
  /// // Binary classification output layer
  /// let output = Dense.sigmoid(inputSize: 128, outputSize: 1)
  ///
  /// // Multi-label classification (independent binary predictions)
  /// let multiLabel = Dense.sigmoid(inputSize: 256, outputSize: 10)
  ///
  /// // LSTM gate computations
  /// let forgetGate = Dense.sigmoid(inputSize: 512, outputSize: 512)
  /// ```
  ///
  /// ## See Also
  /// - ``Sigmoid`` - Standalone sigmoid activation layer
  /// - ``tanh(inputSize:outputSize:dtype:device:)`` - Tanh activation variant
  public static func sigmoid(
    inputSize: Int, outputSize: Int, dtype: DType = .float32, device: Device = .cpu
  ) -> Dense {
    Dense(
      inputSize: inputSize, outputSize: outputSize, activation: { $0.sigmoid() }, dtype: dtype,
      device: device)
  }

  /// Creates a dense layer with ReLU activation.
  ///
  /// ReLU (Rectified Linear Unit) is the most commonly used activation function in modern
  /// deep learning. It's computationally efficient and helps mitigate the vanishing gradient problem.
  ///
  /// - Parameters:
  ///   - inputSize: The number of input features.
  ///   - outputSize: The number of output features.
  ///   - dtype: The data type for weights and biases. Defaults to `.float32`.
  ///   - device: The device where the layer's parameters reside. Defaults to `.cpu`.
  ///
  /// - Returns: A `Dense` layer with ReLU activation.
  ///
  /// ```swift
  /// // Standard MLP hidden layer
  /// let hidden1 = Dense.relu(inputSize: 784, outputSize: 512)
  /// let hidden2 = Dense.relu(inputSize: 512, outputSize: 256)
  ///
  /// // ResNet-style feedforward block
  /// let ffn = Dense.relu(inputSize: 512, outputSize: 2048)
  /// ```
  ///
  /// ## See Also
  /// - ``ReLU`` - Standalone ReLU activation layer
  /// - ``sigmoid(inputSize:outputSize:dtype:device:)`` - Sigmoid activation variant
  /// - ``tanh(inputSize:outputSize:dtype:device:)`` - Tanh activation variant
  public static func relu(
    inputSize: Int, outputSize: Int, dtype: DType = .float32, device: Device = .cpu
  ) -> Dense {
    Dense(
      inputSize: inputSize, outputSize: outputSize, activation: { $0.relu() }, dtype: dtype,
      device: device)
  }

  // MARK: - Forward

  /// Applies the dense layer transformation to the input.
  ///
  /// Performs a linear transformation followed by the activation function:
  /// 1. Computes `z = xW + b` using the underlying ``linear`` layer
  /// 2. Applies the activation function: `y = activation(z)`
  ///
  /// - Parameter x: The input tensor of shape `[..., inputSize]`. The last dimension must match
  ///                the `inputSize` specified during initialization.
  ///
  /// - Returns: The output tensor of shape `[..., outputSize]` after applying both the linear
  ///            transformation and activation function.
  ///
  /// ```swift
  /// let layer = Dense.relu(inputSize: 128, outputSize: 64)
  ///
  /// // Single sample
  /// let input = Tensor.randn([128])
  /// let output = layer(input)  // Shape: [64]
  ///
  /// // Batch processing
  /// let batch = Tensor.randn([32, 128])
  /// let batchOutput = layer(batch)  // Shape: [32, 64]
  ///
  /// // In a forward pass
  /// var x = Tensor.randn([16, 128])
  /// x = layer(x)  // [16, 64] with ReLU applied
  /// ```
  ///
  /// - Note: This method is marked `@differentiable`, enabling automatic gradient computation
  ///         through both the linear transformation and activation function.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    activation(linear(x))
  }
}

// MARK: - Manual derivatives (avoid curried-self path)
extension Dense {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (
      value: Tensor,
      pullback: (Tensor.TangentVector) -> (TangentVector, Tensor.TangentVector)
    )
  {
    func primal(_ s: Dense, _ i: Tensor) -> Tensor {
      s.activation(s.linear(i))
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
