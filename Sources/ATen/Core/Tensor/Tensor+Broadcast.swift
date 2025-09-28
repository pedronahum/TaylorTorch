@preconcurrency import ATenCXX

extension Tensor {
  /// Expand singleton dimensions to match the supplied shape. Returns a view when possible.
  /// - Parameters:
  ///   - shape: Target shape compatible with current broadcast semantics.
  ///   - implicit: Pass `true` for PyTorch-style implicit broadcasting checks.
  @inlinable
  public func expanded(to shape: [Int], implicit: Bool = false) -> Tensor {
    var s64 = shape.map(Int64.init)
    return Tensor(_impl.expand(&s64, s64.count, implicit))
  }

  /// Expand singleton dimensions to match the layout of another tensor. Returns a view when possible.
  /// - Parameter other: Tensor whose shape should be adopted.
  @inlinable
  public func expanded(as other: Tensor) -> Tensor {
    Tensor(_impl.expandAs(other._impl))
  }

  /// Broadcast the tensor to an explicit shape, materializing a view when feasible.
  /// - Parameter shape: Target broadcast shape.
  @inlinable
  public func broadcasted(to shape: [Int]) -> Tensor {
    var s64 = shape.map(Int64.init)
    return Tensor(_impl.broadcastTo(&s64, s64.count))
  }
}

/// Calculates the resulting shape when broadcasting two shapes.
///
/// - Parameters:
///   - shapeA: The first shape.
///   - shapeB: The second shape.
/// - Returns: The broadcasted shape.
/// - Precondition: The shapes must be broadcast-compatible.
public func broadcastShapes(_ shapeA: [Int], _ shapeB: [Int]) -> [Int] {
  // Determine the length of the new shape.
  let countA = shapeA.count
  let countB = shapeB.count
  let resultCount = max(countA, countB)
  var resultShape: [Int] = []
  resultShape.reserveCapacity(resultCount)

  // Iterate backwards from the trailing dimensions.
  for i in 1...resultCount {
    // Get dimensions, using 1 for shorter shapes (implicit dimension).
    let dimA = i <= countA ? shapeA[countA - i] : 1
    let dimB = i <= countB ? shapeB[countB - i] : 1

    let resultDim: Int
    if dimA == dimB {
      // Rule 1: Dimensions are equal.
      resultDim = dimA
    } else if dimA == 1 {
      // Rule 2: One dimension is 1.
      resultDim = dimB
    } else if dimB == 1 {
      // Rule 2: The other dimension is 1.
      resultDim = dimA
    } else {
      // Rule 3: Incompatible dimensions.
      preconditionFailure(
        """
        Shapes \(shapeA) and \(shapeB) are not broadcast-compatible.
        Dimension mismatch at index \(resultCount - i): \(dimA) vs \(dimB).
        """)
    }
    resultShape.append(resultDim)
  }

  // The shape was built in reverse, so correct the order.
  return resultShape.reversed()
}
