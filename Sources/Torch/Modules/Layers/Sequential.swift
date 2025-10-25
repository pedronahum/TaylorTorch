import Foundation
import _Differentiation

// MARK: - Identity (parameterless pass-through)

/// A parameterless layer that returns its input unchanged.
///
/// `Identity` is a pass-through layer that performs no transformation on its input.
/// It's useful as a placeholder, for conditional logic in models, or when you need
/// a layer that maintains type compatibility without modifying data.
///
/// ## Overview
///
/// Identity layers are commonly used for:
/// - Placeholder layers during model development
/// - Optional branches in conditional models
/// - Maintaining type compatibility in generic code
/// - Debugging (insert to verify tensor shapes)
///
/// ## Usage
///
/// ```swift
/// // As a placeholder
/// let layer = Identity<Tensor>()
/// let output = layer(input)  // output === input
///
/// // In a Sequential model (though typically not needed)
/// let model = Sequential {
///     Linear(inputSize: 128, outputSize: 64)
///     Identity<Tensor>()  // No-op, for illustration
///     ReLU()
///     Linear(inputSize: 64, outputSize: 10)
/// }
///
/// // For conditional logic
/// struct ConditionalModel<T: Differentiable>: Layer {
///     let useTransform: Bool
///     let transform: Linear
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         if useTransform {
///             return transform(input)
///         } else {
///             return Identity<Tensor>()(input)
///         }
///     }
/// }
/// ```
///
/// ## Automatic Differentiation
///
/// Identity layers are differentiable with trivial gradients:
///
/// ```swift
/// let layer = Identity<Tensor>()
/// let input = Tensor.randn([10])
///
/// let (output, pullback) = valueWithPullback(at: input) { x in
///     layer(x)
/// }
/// // pullback simply returns its input unchanged
/// ```
///
/// ## See Also
///
/// - ``Sequential`` - Container for composing layers
/// - ``ParameterlessLayer`` - Protocol for layers without trainable parameters
public struct Identity<IO: Differentiable>: ParameterlessLayer {
  public typealias Input = IO
  public typealias Output = IO
  public typealias TangentVector = EmptyTangentVector

  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: IO) -> IO { x }

}

extension Sequential {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector) -> (TangentVector, Input.TangentVector)
    )
  {
    // Differentiate through the stored `body` directly.
    let (y, bodyPB) = body.appliedForBackpropagation(to: x)
    return (
      y,
      { v in
        let (dBody, dX) = bodyPB(v)  // dBody : Body.TangentVector
        return (
          TangentVector(body: dBody),  // wrap into Sequential.TangentVector
          dX
        )
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector) -> TangentVector
    )
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

// MARK: - Chain (two-layer composition with explicit tangent)

/// Compose two layers `First` → `Second` such that `First.Output == Second.Input`.
public struct Chain<First: Layer, Second: Layer>: Layer where First.Output == Second.Input {
  public var first: First
  public var second: Second

  public init(_ first: First, _ second: Second) {
    self.first = first
    self.second = second
  }

  public typealias Input = First.Input
  public typealias Output = Second.Output

  // Manual TangentVector to avoid synthesis pitfalls and guarantee AdditiveArithmetic witnesses.
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // explicit zero/+/- to avoid solver corner cases
    KeyPathIterable,  // required by Module
    VectorProtocol,  // required by Module
    PointwiseMultiplicative  // required by Module
  {
    public typealias VectorSpaceScalar = Float
    public var first: First.TangentVector
    public var second: Second.TangentVector

    public init(first: First.TangentVector = .zero, second: Second.TangentVector = .zero) {
      self.first = first
      self.second = second
    }
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(first: lhs.first + rhs.first, second: lhs.second + rhs.second)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(first: lhs.first - rhs.first, second: lhs.second - rhs.second)
    }
  }

  public mutating func move(by d: TangentVector) {
    first.move(by: d.first)
    second.move(by: d.second)
  }

  @differentiable(reverse)
  public func callAsFunction(_ x: Input) -> Output {
    let y = first(x)
    return second(y)
  }

  // Manual VJPs to avoid “curried self” solver path.
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector) -> (TangentVector, Input.TangentVector)
    )
  {
    // Forward through first, then second.
    let y1 = first(x)
    let (y, pbSecond) = second.appliedForBackpropagation(to: y1)
    return (
      y,
      { v in
        // Backprop through second, then first.
        let (dSecond, dY1) = pbSecond(v)
        let (_, pbFirst) = first.appliedForBackpropagation(to: x)
        let (dFirst, dX) = pbFirst(dY1)
        return (TangentVector(first: dFirst, second: dSecond), dX)
      }
    )
  }

  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Input)
    -> (value: Output, pullback: (Output.TangentVector) -> TangentVector)
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

// MARK: - Result builder for sequential models

@resultBuilder
public enum SequentialBuilder {
  /// First element starts the chain as-is.
  public static func buildPartialBlock<First: Layer>(first: First) -> First { first }

  /// Append the next layer, ensuring the types line up: `Accum.Output == Next.Input`.
  public static func buildPartialBlock<Accum: Layer, Next: Layer>(
    accumulated: Accum, next: Next
  ) -> Chain<Accum, Next> where Accum.Output == Next.Input {
    Chain(accumulated, next)
  }

  // If you later want `if`/`else` in builders, add:
  // public static func buildEither<TrueBranch: Layer, FalseBranch: Layer>(
  //   first: TrueBranch
  // ) -> TrueBranch { first }
  // public static func buildEither<TrueBranch: Layer, FalseBranch: Layer>(
  //   second: FalseBranch
  // ) -> FalseBranch { second }
  //
  // And possibly buildOptional/buildArray with an Identity wrapper.
}

// MARK: - Sequential (thin, differentiable wrapper over the built chain)

/// A container for composing multiple layers into a sequential pipeline.
///
/// `Sequential` provides a declarative, type-safe way to build neural networks by stacking layers
/// in order. It uses Swift's result builder syntax to create clean, readable model definitions
/// with compile-time type checking ensuring output types match input types.
///
/// ## Overview
///
/// Sequential models are the most common way to build neural networks. Each layer's output
/// becomes the next layer's input, creating a data processing pipeline:
///
/// ```
/// Input → Layer₁ → Layer₂ → ... → Layerₙ → Output
/// ```
///
/// ## Creating Sequential Models
///
/// ```swift
/// // Simple feedforward network
/// let model = Sequential {
///     Linear(inputSize: 784, outputSize: 512)
///     ReLU()
///     Dropout(probability: 0.5)
///     Linear(inputSize: 512, outputSize: 256)
///     ReLU()
///     Dropout(probability: 0.5)
///     Linear(inputSize: 256, outputSize: 10)
/// }
///
/// // Process a batch
/// let input = Tensor.randn([32, 784])
/// let output = model(input)  // Shape: [32, 10]
/// ```
///
/// ## Type Safety
///
/// Swift's type system ensures layers are compatible at compile time:
///
/// ```swift
/// // This compiles - output sizes match input sizes
/// let valid = Sequential {
///     Linear(inputSize: 128, outputSize: 64)  // Outputs 64 features
///     ReLU()                                   // Preserves shape
///     Linear(inputSize: 64, outputSize: 32)   // Accepts 64 features ✓
/// }
///
/// // This won't compile - type mismatch
/// let invalid = Sequential {
///     Linear(inputSize: 128, outputSize: 64)  // Outputs 64 features
///     Linear(inputSize: 32, outputSize: 16)   // Expects 32 features ✗
/// }
/// // Error: Cannot convert value of type 'Linear' to expected type...
/// ```
///
/// ## Common Network Architectures
///
/// ### Multi-Layer Perceptron (MLP)
///
/// ```swift
/// struct MNISTClassifier {
///     let model = Sequential {
///         Linear(inputSize: 784, outputSize: 512)
///         ReLU()
///         Dropout(probability: 0.2)
///         Linear(inputSize: 512, outputSize: 256)
///         ReLU()
///         Dropout(probability: 0.2)
///         Linear(inputSize: 256, outputSize: 10)
///     }
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         model(input)
///     }
/// }
/// ```
///
/// ### Convolutional Network
///
/// ```swift
/// let cnn = Sequential {
///     Conv2D(inputChannels: 3, outputChannels: 32, kernelSize: 3)
///     ReLU()
///     Conv2D(inputChannels: 32, outputChannels: 64, kernelSize: 3)
///     ReLU()
///     Flatten()
///     Linear(inputSize: 64 * 26 * 26, outputSize: 128)
///     ReLU()
///     Linear(inputSize: 128, outputSize: 10)
/// }
/// ```
///
/// ## Using with Training
///
/// ```swift
/// let model = Sequential {
///     Linear(inputSize: 128, outputSize: 64)
///     ReLU()
///     Linear(inputSize: 64, outputSize: 10)
/// }
///
/// // Training loop
/// for epoch in 0..<10 {
///     for batch in trainingData {
///         let (loss, grad) = valueWithGradient(at: model) { m in
///             let predictions = m(batch.input)
///             return crossEntropy(predictions, batch.labels)
///         }
///
///         optimizer.update(&model, along: grad)
///     }
/// }
/// ```
///
/// ## Automatic Differentiation
///
/// Sequential models are fully differentiable. Gradients flow backward through all layers:
///
/// ```swift
/// let model = Sequential {
///     Linear(inputSize: 10, outputSize: 5)
///     ReLU()
///     Linear(inputSize: 5, outputSize: 1)
/// }
///
/// let input = Tensor.randn([1, 10])
/// let (output, pullback) = valueWithPullback(at: model, input) { m, x in
///     m(x)
/// }
///
/// let gradOutput = Tensor.ones([1, 1])
/// let (modelGrad, inputGrad) = pullback(gradOutput)
/// // modelGrad contains gradients for all layer parameters
/// ```
///
/// ## Mixing Layer Types
///
/// You can combine any layers that have compatible input/output types:
///
/// ```swift
/// // Custom activation layers
/// let model = Sequential {
///     Dense.relu(inputSize: 256, outputSize: 128)
///     Dropout(probability: 0.3)
///     Dense(inputSize: 128, outputSize: 64, activation: { $0.gelu() })
///     Dense.tanh(inputSize: 64, outputSize: 32)
///     Linear(inputSize: 32, outputSize: 10)  // No activation
/// }
/// ```
///
/// ## Performance Considerations
///
/// - **Compile Time**: Very long Sequential chains may increase compile times
/// - **Memory**: All intermediate activations are stored for backpropagation
/// - **GPU**: Batching inputs significantly improves GPU utilization
/// - **Alternative**: For very deep networks, consider splitting into multiple Sequential blocks
///
/// ## Sequential vs Custom Layers
///
/// **Use Sequential when:**
/// - Building straightforward feedforward networks
/// - Layers connect in a simple chain
/// - You want declarative, readable code
///
/// **Use custom `Layer` structs when:**
/// - You need branching (residual connections, concatenation)
/// - Complex control flow is required
/// - You want to reuse sub-components
///
/// ```swift
/// // Complex model with residual connections - use custom struct
/// struct ResidualBlock: Layer {
///     var conv1: Conv2D
///     var conv2: Conv2D
///
///     @differentiable
///     func callAsFunction(_ input: Tensor) -> Tensor {
///         let residual = input
///         var x = conv1(input).relu()
///         x = conv2(x)
///         return x + residual  // Skip connection
///     }
/// }
/// ```
///
/// ## Topics
///
/// ### Creating Sequential Models
///
/// - ``init(_:)-2w84x``
///
/// ### Forward Pass
///
/// - ``callAsFunction(_:)``
///
/// ### Properties
///
/// - ``body``
///
/// ## See Also
///
/// - ``Layer`` - Base protocol for all layers
/// - ``Chain`` - Low-level two-layer composition
/// - ``Dense`` - Dense layers with activations
/// - ``Linear`` - Fully connected layers
public struct Sequential<Body: Layer>: Layer {
  /// The composed layer chain representing the sequential pipeline.
  ///
  /// This property holds the result of composing all layers provided in the initializer.
  /// It's typically a nested ``Chain`` structure built by ``SequentialBuilder``.
  public var body: Body

  /// Creates a sequential model by composing layers using a result builder.
  ///
  /// This initializer uses Swift's `@SequentialBuilder` to provide clean, declarative syntax
  /// for building neural networks. Layers are composed left-to-right (top-to-bottom in code),
  /// with compile-time verification that output types match input types.
  ///
  /// - Parameter layers: A closure that uses the `@SequentialBuilder` to compose layers.
  ///                     Each layer's output type must match the next layer's input type.
  ///
  /// ```swift
  /// // Simple classifier
  /// let model = Sequential {
  ///     Linear(inputSize: 784, outputSize: 256)
  ///     ReLU()
  ///     Linear(inputSize: 256, outputSize: 10)
  /// }
  ///
  /// // With dropout regularization
  /// let regularized = Sequential {
  ///     Linear(inputSize: 512, outputSize: 256)
  ///     ReLU()
  ///     Dropout(probability: 0.5)
  ///     Linear(inputSize: 256, outputSize: 128)
  ///     ReLU()
  ///     Dropout(probability: 0.5)
  ///     Linear(inputSize: 128, outputSize: 10)
  /// }
  ///
  /// // Mixed layer types
  /// let mixed = Sequential {
  ///     Dense.relu(inputSize: 100, outputSize: 50)
  ///     Dropout(probability: 0.3)
  ///     Dense.tanh(inputSize: 50, outputSize: 25)
  ///     Linear(inputSize: 25, outputSize: 10)
  /// }
  /// ```
  ///
  /// - Note: The Swift compiler verifies type compatibility at compile time. If layers
  ///         have incompatible input/output types, you'll get a compile error.
  public init(@SequentialBuilder _ layers: () -> Body) {
    self.body = layers()
  }

  public typealias Input = Body.Input
  public typealias Output = Body.Output

  // Manual TangentVector: a simple pass-through to Body’s tangent.
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var body: Body.TangentVector

    public init(body: Body.TangentVector = .zero) { self.body = body }

    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self { .init(body: lhs.body + rhs.body) }
    public static func - (lhs: Self, rhs: Self) -> Self { .init(body: lhs.body - rhs.body) }
  }

  public mutating func move(by d: TangentVector) { body.move(by: d.body) }

  /// Applies the sequential pipeline to the input.
  ///
  /// Processes the input through each layer in order, passing each layer's output as the
  /// next layer's input. This is equivalent to manually composing the layers, but with
  /// cleaner syntax and automatic type checking.
  ///
  /// - Parameter x: The input tensor. Its type must match the first layer's input type.
  ///
  /// - Returns: The output tensor after processing through all layers. Its type matches
  ///            the last layer's output type.
  ///
  /// ```swift
  /// let model = Sequential {
  ///     Linear(inputSize: 128, outputSize: 64)
  ///     ReLU()
  ///     Linear(inputSize: 64, outputSize: 10)
  /// }
  ///
  /// // Single forward pass
  /// let input = Tensor.randn([32, 128])  // Batch of 32
  /// let output = model(input)  // Shape: [32, 10]
  ///
  /// // Equivalent to manual composition:
  /// // let x1 = Linear(inputSize: 128, outputSize: 64)(input)
  /// // let x2 = ReLU()(x1)
  /// // let output = Linear(inputSize: 64, outputSize: 10)(x2)
  /// ```
  ///
  /// - Note: This method is marked `@differentiable`, enabling gradient computation
  ///         through the entire sequential pipeline during backpropagation.
  @differentiable(reverse)
  public func callAsFunction(_ x: Input) -> Output { body(x) }

}
