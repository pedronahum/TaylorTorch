@preconcurrency import ATenCXX

public extension Tensor {
  /// Concatenate tensors along an existing dimension.
  /// - Parameters:
  ///   - tensors: Non-empty list of tensors to concatenate.
  ///   - dim: Dimension along which to concatenate (defaults to `0`).
  static func cat(_ tensors: [Tensor], dim: Int = 0) -> Tensor {
    precondition(!tensors.isEmpty, "cat: empty tensor list")
    let impls = tensors.map { $0._impl }
    return impls.withUnsafeBufferPointer { bp in
      precondition(bp.baseAddress != nil)
      return Tensor(TTSTensor.cat(bp.baseAddress!, bp.count, Int64(dim)))
    }
  }

  /// Stack tensors along a new dimension.
  /// - Parameters:
  ///   - tensors: Non-empty list of tensors to stack (must share shape).
  ///   - dim: Newly created dimension for the stack (defaults to `0`).
  static func stack(_ tensors: [Tensor], dim: Int = 0) -> Tensor {
    precondition(!tensors.isEmpty, "stack: empty tensor list")
    let impls = tensors.map { $0._impl }
    return impls.withUnsafeBufferPointer { bp in
      precondition(bp.baseAddress != nil)
      return Tensor(TTSTensor.stack(bp.baseAddress!, bp.count, Int64(dim)))
    }
  }
}
