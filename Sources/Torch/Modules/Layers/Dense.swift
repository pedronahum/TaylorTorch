import _Differentiation

/// Dense = Linear followed by an activation (no extra parameters).
///
/// We intentionally avoid storing a `@differentiable` closure in the layer to
/// sidestep a Differentiation pass assertion in current dev toolchains. Instead,
/// we carry a tiny activation "kind" and apply it in `call`.
public struct Dense: Layer {
  // MARK: Activation

  @frozen
  public enum ActivationKind: Sendable, Equatable {
    case identity
    case relu
    // Add more cases if you want to route through here.
  }

  // MARK: Stored state

  public var linear: Linear
  @noDerivative public var activation: ActivationKind

  // MARK: Inits (no default-argument closures)

  @inlinable
  public init(linear: Linear, activation: ActivationKind) {
    self.linear = linear
    self.activation = activation
  }

  @inlinable
  public init(linear: Linear) {
    self.init(linear: linear, activation: .identity)
  }

  @inlinable
  public init(
    inFeatures: Int,
    outFeatures: Int,
    dtype: DType,
    device: Device,
    activation: ActivationKind
  ) {
    self.linear = .glorot(
      inFeatures: inFeatures,
      outFeatures: outFeatures,
      dtype: dtype,
      device: device
    )
    self.activation = activation
  }

  @inlinable
  public init(
    inFeatures: Int,
    outFeatures: Int,
    dtype: DType,
    device: Device
  ) {
    self.init(
      inFeatures: inFeatures,
      outFeatures: outFeatures,
      dtype: dtype,
      device: device,
      activation: .identity
    )
  }

  @inlinable
  public init(inFeatures: Int, outFeatures: Int) {
    self.init(
      inFeatures: inFeatures,
      outFeatures: outFeatures,
      dtype: .float32,
      device: .cpu,
      activation: .identity
    )
  }

  @inlinable
  public init(
    inFeatures: Int,
    outFeatures: Int,
    activation: ActivationKind
  ) {
    self.init(
      inFeatures: inFeatures,
      outFeatures: outFeatures,
      dtype: .float32,
      device: .cpu,
      activation: activation
    )
  }

  // MARK: Forward

  @inlinable
  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    applyActivation(linear(x))
  }

  @inlinable
  @differentiable(reverse)
  public func call(_ x: Tensor, context: ForwardContext) -> Tensor {
    applyActivation(linear.call(x, context: context))
  }

  @inlinable
  @differentiable(reverse)
  func applyActivation(_ z: Tensor) -> Tensor {
    switch activation {
    case .identity: return z
    case .relu: return z.relu()
    }
  }

  // MARK: Parameter traversal & AD plumbing

  @inlinable
  public mutating func move(by offset: TangentVector) {
    linear.move(by: offset.linear)
  }

  @inlinable
  public static var parameterKeyPaths: [WritableKeyPath<Dense, Tensor>] {
    var paths: [WritableKeyPath<Dense, Tensor>] = []
    for kp in Linear.parameterKeyPaths {
      paths.append((\Dense.linear).appending(kp))
    }
    return paths
  }

  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var linear: Linear.TangentVector

    /// Make this explicit so `@inlinable` uses below can reference it.
    @inlinable
    public init(linear: Linear.TangentVector) { self.linear = linear }

    @inlinable public static var zero: TangentVector { .init(linear: .zero) }
    @inlinable public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(linear: lhs.linear + rhs.linear)
    }
    @inlinable public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(linear: lhs.linear - rhs.linear)
    }

    @inlinable
    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] {
      var paths: [WritableKeyPath<TangentVector, Tensor>] = []
      for kp in Linear.TangentVector.parameterKeyPaths {
        paths.append((\TangentVector.linear).appending(kp))
      }
      return paths
    }
  }
}
