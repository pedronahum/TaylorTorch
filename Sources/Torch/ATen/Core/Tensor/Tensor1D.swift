// Sources/ATen/Tensor1D.swift
/// Strongly typed wrapper over a rank-1 tensor that exposes collection semantics.
public struct Tensor1D<T: TorchArithmetic>: RandomAccessCollection, MutableCollection, Sendable, ExpressibleByArrayLiteral {
  public typealias Index = Int
  public var base: Tensor   // rank-1, dtype == T.torchDType

  /// Creates a 1-D view over `t`, asserting rank and dtype compatibility.
  public init(_ t: Tensor) {
    precondition(t.rank == 1 && t.dtype == T.torchDType)
    base = t
  }
  /// Builds a tensor literal from array literal syntax.
  public init(arrayLiteral elements: T...) {
    base = Tensor(array: elements, shape: [elements.count])
  }

  /// Returns a tensor filled with zeros of length `count`.
  public static func zeros(count: Int, device: Device = .cpu) -> Self {
    .init(Tensor.zeros(shape: [count], dtype: T.torchDType, device: device))
  }

  public var startIndex: Int { 0 }
  public var endIndex: Int { base.shape[0] }

  /// Gets or sets the element at `i`, materializing a copy-on-write update for writes.
  public subscript(i: Int) -> T {
    get { base[i].toArray(as: T.self)[0] }
    set {
      // simple path: copy back (80/20). Can be optimized with a masked update.
      var a = base.toArray(as: T.self)
      a[i] = newValue
      base = Tensor(array: a, shape: [a.count], device: base.device)
    }
  }
}

// Elementwise operators via your Tensor operators
/// Element-wise addition of two 1-D tensor wrappers.
@inlinable public func + <T>(lhs: Tensor1D<T>, rhs: Tensor1D<T>) -> Tensor1D<T> where T: TorchArithmetic {
  Tensor1D<T>(lhs.base + rhs.base)
}
/// Element-wise multiplication of two 1-D tensor wrappers.
@inlinable public func * <T>(lhs: Tensor1D<T>, rhs: Tensor1D<T>) -> Tensor1D<T> where T: TorchArithmetic {
  Tensor1D<T>(lhs.base * rhs.base)
}
/// Element-wise subtraction of two 1-D tensor wrappers.
@inlinable public func - <T>(lhs: Tensor1D<T>, rhs: Tensor1D<T>) -> Tensor1D<T> where T: TorchArithmetic {
  Tensor1D<T>(lhs.base - rhs.base)
}
/// Element-wise division of two 1-D tensor wrappers (requires floating scalars).
@inlinable public func / <T>(lhs: Tensor1D<T>, rhs: Tensor1D<T>) -> Tensor1D<T> where T: TorchFloating {
  Tensor1D<T>(lhs.base / rhs.base)
}
