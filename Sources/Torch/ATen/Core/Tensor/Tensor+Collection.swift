// Sources/ATen/Tensor+Collection.swift

/// Materialized collection view over a rank-1 tensor's elements.
public struct TensorElements<T: TorchTensorScalar>: RandomAccessCollection, Sendable {
  public typealias Index = Int
  public let startIndex: Int = 0
  public let endIndex: Int
  @usableFromInline var _buf: [T]

  /// Copies the tensor data into an internal buffer to enable collection traversal.
  public init(_ base: Tensor, as: T.Type) {
    precondition(base.rank == 1, "TensorElements: expected rank-1 tensor")
    precondition(base.dtype == T.torchDType, "dtype mismatch: \(String(describing: base.dtype)) vs \(T.torchDType)")
    _buf = base.toArray(as: T.self)
    endIndex = _buf.count
  }

  /// Element accessor binding to the copied storage buffer.
  public subscript(position: Int) -> T { _buf[position] }
}

public extension Tensor {
  /// Returns a random-access snapshot of 1-D tensor elements.
  func elements<T: TorchTensorScalar>(as: T.Type) -> TensorElements<T> {
    TensorElements(self, as: T.self)
  }
}

// TODO: 
// 1.2 (Optional) zero‑copy(ish) “borrowed view”
// When we later add a C++ shim that exposes an untyped data pointer 
// only for CPU+contiguous tensors, you can provide an alternative view using 
// withHostBuffer you already shipped; for now we can also pipe through that 
// (it still copies, but the API shape is right).
