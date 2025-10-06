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
  /// - Parameters:
  ///   - linear: Underlying affine transform that supplies weights and bias.
  ///   - activation: Differentiable nonlinearity applied after the affine transform.
  public init(linear: Linear, activation: @escaping Activation = { $0 }) {
    self.linear = linear
    self.activation = activation
  }

  /// Create a dense layer with Glorot initialization for `Linear` and an activation.
  /// - Parameters:
  ///   - inFeatures: Number of input features.
  ///   - outFeatures: Number of output features.
  ///   - dtype: Element dtype for the parameters.
  ///   - device: Device on which to allocate the parameters.
  ///   - activation: Differentiable nonlinearity applied after the affine transform.
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
  /// - Parameter x: Input activations to transform.
  /// - Returns: Activated output of the dense layer.
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    activation(linear(x))
  }

  /// Contextual forward (threads `ForwardContext` through `Linear`).
  /// - Parameters:
  ///   - x: Input activations to transform.
  ///   - context: Forward-context structure propagated through the layer stack.
  /// - Returns: Activated output of the dense layer.
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    activation(linear.call(x, context: context))
  }

  // MARK: - Parameter traversal & AD plumbing

  /// Applies the tangent `offset` to the underlying linear layer.
  /// - Parameter offset: Tangent information propagated from differentiation.
  public mutating func move(by offset: TangentVector) {
    linear.move(by: offset.linear)
  }

  /// Writable key paths for the dense layer's trainable tensors.
  public static var parameterKeyPaths: [WritableKeyPath<Dense, Tensor>] {
    var paths: [WritableKeyPath<Dense, Tensor>] = []
    for kp in Linear.parameterKeyPaths {
      paths.append((\Dense.linear).appending(kp))
    }
    return paths
  }

  /// Tangent representation for `Dense`.
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    /// Tangent for the underlying linear layer.
    public var linear: Linear.TangentVector
    /// Additive identity for the tangent vector.
    public static var zero: TangentVector { .init(linear: .zero) }
    /// Adds two tangent vectors element-wise.
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(linear: lhs.linear + rhs.linear)
    }
    /// Subtracts two tangent vectors element-wise.
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(linear: lhs.linear - rhs.linear)
    }
    /// Writable key paths for the tangent components.
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
  /// Returns the input unchanged.
  @differentiable(reverse) public static func identity(_ x: Tensor) -> Tensor { x }
  /// Applies the rectified linear unit.
  @differentiable(reverse) public static func relu(_ x: Tensor) -> Tensor { x.relu() }

  // TODO: Enable tanh and sigmoid (including gradients).
  // Review if these and other activations should be part of the C++ Shim hpp file
  // See (Section Unary of tensor_shim.hpp)
  // Implement the swift wrappers and make them differentiable.
  // See (Tensor+Math.swift and Tensor+Math+Differentiable.swift)
  /// Applies the exponential function.
  @differentiable(reverse) public static func exp(_ x: Tensor) -> Tensor { x.exp() }
  /// Applies the natural logarithm.
  @differentiable(reverse) public static func log(_ x: Tensor) -> Tensor { x.log() }
  /// Applies the square-root function.
  @differentiable(reverse) public static func sqrt(_ x: Tensor) -> Tensor { x.sqrt() }
  /// Applies the hyperbolic tangent.
  @differentiable(reverse) public static func tanh(_ x: Tensor) -> Tensor { x.tanh() }
  /// Applies the logistic sigmoid.
  @differentiable(reverse) public static func sigmoid(_ x: Tensor) -> Tensor { x.sigmoid() }
  /// Applies the sine function.
  @differentiable(reverse) public static func sin(_ x: Tensor) -> Tensor { x.sin() }
  /// Applies the cosine function.
  @differentiable(reverse) public static func cos(_ x: Tensor) -> Tensor { x.cos() }
  /// Applies the tangent function.
  @differentiable(reverse) public static func tan(_ x: Tensor) -> Tensor { x.tan() }
  /// Applies the inverse sine.
  @differentiable(reverse) public static func asin(_ x: Tensor) -> Tensor { x.asin() }
  /// Applies the inverse cosine.
  @differentiable(reverse) public static func acos(_ x: Tensor) -> Tensor { x.acos() }
  /// Applies the inverse tangent.
  @differentiable(reverse) public static func atan(_ x: Tensor) -> Tensor { x.atan() }
  /// Applies the hyperbolic sine.
  @differentiable(reverse) public static func sinh(_ x: Tensor) -> Tensor { x.sinh() }
  /// Applies the hyperbolic cosine.
  @differentiable(reverse) public static func cosh(_ x: Tensor) -> Tensor { x.cosh() }
  /// Applies the inverse hyperbolic sine.
  @differentiable(reverse) public static func asinh(_ x: Tensor) -> Tensor { x.asinh() }
  /// Applies the inverse hyperbolic cosine.
  @differentiable(reverse) public static func acosh(_ x: Tensor) -> Tensor { x.acosh() }
  /// Applies the inverse hyperbolic tangent.
  @differentiable(reverse) public static func atanh(_ x: Tensor) -> Tensor { x.atanh() }
  /// Applies the Gaussian error function.
  @differentiable(reverse) public static func erf(_ x: Tensor) -> Tensor { x.erf() }
  /// Applies the complementary error function.
  @differentiable(reverse) public static func erfc(_ x: Tensor) -> Tensor { x.erfc() }

}
