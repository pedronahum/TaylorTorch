@preconcurrency import ATenCXX

// ✅ Add the full enum definition back to the top of this file
/// Represents a single component of a multi-axis tensor subscript operation.
public enum TensorIndex: ExpressibleByIntegerLiteral {
  case i(Int)
  case range(Range<Int>)
  case closed(ClosedRange<Int>)
  case all
  case newAxis
  case ellipsis // This is the case the compiler couldn't find

  /// Allows integer literals to index directly into a dimension.
  public init(integerLiteral value: Int) {
    self = .i(value)
  }
}

public extension Tensor {
  /// Variadic subscript routing into the array-based implementation.
  subscript(_ indices: TensorIndex...) -> Tensor { self[indices] }

  /// Advanced indexing entry point that supports integers, ranges, ellipsis, and new-axis markers.
  subscript(_ indices: [TensorIndex]) -> Tensor {
    precondition(!indices.isEmpty, "indices cannot be empty")
    var result = self
    var dim = 0

    // Expand .ellipsis ahead of time
    let hasEllipsis = indices.contains { if case .ellipsis = $0 { return true } else { return false } }
    var expanded: [TensorIndex] = []
    if hasEllipsis {
      let before = indices.prefix { if case .ellipsis = $0 { false } else { true } }.count
      let after  = indices.reversed().prefix { if case .ellipsis = $0 { false } else { true } }.count
      let consumed = indices[0..<before].filter { if case .newAxis = $0 { false } else { true } }.count
                   + indices[(indices.count - after)..<indices.count].filter { if case .newAxis = $0 { false } else { true } }.count
      let toFill = Swift.max(0, result.rank - consumed)
      expanded.append(contentsOf: indices[0..<before])
      expanded.append(contentsOf: Array(repeating: .all, count: toFill))
      expanded.append(contentsOf: indices[(indices.count - after)..<indices.count])
    } else {
      expanded = indices
    }

    for idx in expanded {
      switch idx {
      case .i(let k):
        result = result.select(dim: dim, index: Int64(k))
      case .range(let r):
        result = result.slice(dim: dim, start: r.lowerBound, end: r.upperBound, step: 1)
        dim += 1
      case .closed(let cr):
        result = result.slice(dim: dim, start: cr.lowerBound, end: cr.upperBound &+ 1, step: 1)
        dim += 1
      case .all:
        result = result.slice(dim: dim, start: 0, end: result._dimSize(dim), step: 1)
        dim += 1
      case .newAxis:
        result = result.unsqueezed(dim: dim)
        dim += 1
      case .ellipsis:
        continue
      }
    }
    return result
  }
}
