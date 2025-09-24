// Sources/ATen/ScalarTensor.swift

/// Lightweight value wrapper that exposes a scalar tensor as a `Numeric` value.
public struct ScalarTensor<T: TorchArithmetic & Comparable>: Sendable, Equatable, Comparable, AdditiveArithmetic, Numeric {

  /// Underlying single-element tensor.
  public var base: Tensor

  /// Creates a scalar tensor storing `value` on the provided `device`.
  public init(_ value: T, device: Device = .cpu) { self.base = Tensor(value, device: device) }
  /// Extracts the scalar value from the backing tensor.
  public var value: T { base.toArray(as: T.self)[0] }

  // Comparable
  public static func < (lhs: ScalarTensor<T>, rhs: ScalarTensor<T>) -> Bool {
    return lhs.value < rhs.value // This will now compile
  }

  // AdditiveArithmetic
  public static var zero: Self { .init(T.zero) }
  public static func + (l: Self, r: Self) -> Self { .init((l.base + r.base).toArray(as: T.self)[0]) }
  public static func - (l: Self, r: Self) -> Self { .init((l.base - r.base).toArray(as: T.self)[0]) }

  // Numeric & ExpressibleByIntegerLiteral
  public typealias Magnitude = Self
  public typealias IntegerLiteralType = Int

  public var magnitude: Magnitude { self }

  /// Initializes the scalar tensor from an integer literal when `T` can exactly represent it.
  public init(integerLiteral value: IntegerLiteralType) {
      guard let val = T(exactly: value) else {
          fatalError("Cannot initialize ScalarTensor with integer literal \(value)")
      }
      self.init(val)
  }

  /// Failable initializer that attempts to convert `source` into `T` exactly.
  public init?<U: BinaryInteger>(exactly source: U) {
    guard let value = T(exactly: source) else { return nil }
    self.init(value)
  }

  public static func * (l: Self, r: Self) -> Self { .init((l.base * r.base).toArray(as: T.self)[0]) }
  public static func *= (l: inout Self, r: Self) { l = l * r }
}
