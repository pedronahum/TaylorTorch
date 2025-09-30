import _Differentiation

/// Dense = Linear + activation (no extra parameters).
///
/// **Why this layer?**
/// - Mirrors S4TF's ergonomic API where the activation is a differentiable closure
///   (e.g., `relu`, `tanh`) supplied at init time. It's simple, composable, and keeps
///   the parameter surface minimal—only `Linear`'s weights/bias are trainable. :contentReference[oaicite:1]{index=1}
/// - Integrates seamlessly with your existing stack:
///   - `Layer` protocol (pure `callAsFunction`, plus a contextful `call` for future
///     stochastic/stateful layers). :contentReference[oaicite:2]{index=2}
///   - `Linear` parameters/gradients and your optimizer algebra via key-path traversal.
///   - `Sequential` and `SequentialBlock` builder for ergonomic model assembly. :contentReference[oaicite:4]{index=4}
///
/// ### Examples
/// ```swift
/// // 1) Vanilla usage
/// var dense = Dense(inFeatures: 784, outFeatures: 256, activation: Activations.relu)
/// let y = dense(x) // y = relu(x W^T + b)
///
/// // 2) With the result-builder
/// let mlp = SequentialBlock {
///   Dense(inFeatures: 784, outFeatures: 256, activation: Activations.relu)
///   Dense(inFeatures: 256, outFeatures: 10, activation: Activations.identity)
/// }
///
/// // 3) Custom activation as a differentiable closure
/// let leaky: Dense.Activation = { x in x.relu().adding(0.01 * x) }
/// var custom = Dense(inFeatures: 128, outFeatures: 128, activation: leaky)
/// ```
public struct Dense: Layer {
  /// A differentiable activation function. Stored as a constant (non-parameter).
  public typealias Activation = @differentiable(reverse) (Tensor) -> Tensor

  /// The underlying affine transform (trainable parameters).
  public var linear: Linear

  /// Non-parameter activation function. Defaults to identity.
  @noDerivative public var activation: Activation

  // MARK: - Inits

  /// Create a dense layer from an existing `Linear` and an activation.
  public init(linear: Linear, activation: @escaping Activation = { $0 }) {
    self.linear = linear
    self.activation = activation
  }

  /// Create a dense layer with Glorot initialization for `Linear` and an activation.
  public init(
    inFeatures: Int,
    outFeatures: Int,
    dtype: DType = .float32,
    device: Device = .cpu,
    activation: @escaping Activation = { $0 }
  ) {
    self.linear = .glorot(
      inFeatures: inFeatures,
      outFeatures: outFeatures,
      dtype: dtype,
      device: device
    )
    self.activation = activation
  }

  // MARK: - Forward

  /// y = activation( x · Wᵀ + b )
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    activation(linear(x))
  }

  /// Contextual forward (threads `ForwardContext` through `Linear`).
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    activation(linear.call(x, context: context))
  }

  // MARK: - Parameter traversal & AD plumbing

  public mutating func move(by offset: TangentVector) {
    linear.move(by: offset.linear)
  }

  public static var parameterKeyPaths: [WritableKeyPath<Dense, Tensor>] {
    var paths: [WritableKeyPath<Dense, Tensor>] = []
    for kp in Linear.parameterKeyPaths {
      paths.append((\Dense.linear).appending(kp))
    }
    return paths
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var linear: Linear.TangentVector
    public static var zero: TangentVector { .init(linear: .zero) }
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(linear: lhs.linear + rhs.linear)
    }
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(linear: lhs.linear - rhs.linear)
    }
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      var paths: [WritableKeyPath<TangentVector, Tensor>] = []
      for kp in Linear.TangentVector.parameterKeyPaths {
        paths.append((\TangentVector.linear).appending(kp))
      }
      return paths
    }
  }
}

/// Namespaced, differentiable activation utilities that work as `Dense.Activation`.
public enum Activations {
  @differentiable(reverse) public static func identity(_ x: Tensor) -> Tensor { x }
  @differentiable(reverse) public static func relu(_ x: Tensor) -> Tensor { x.relu() }

  // TODO: Enable tanh and sigmoid (including gradients).
  // Review if these and other activations should be part of the C++ Shim hpp file
  // See (Section Unary of tensor_shim.hpp)
  // Implement the swift wrappers and make them differentiable.
  // See (Tensor+Math.swift and Tensor+Math+Differentiable.swift)
  //
  //@differentiable(reverse) public static func tanh(_ x: Tensor) -> Tensor { x.tanh() }
  //@differentiable(reverse) public static func sigmoid(_ x: Tensor) -> Tensor { x.sigmoid() }
}
