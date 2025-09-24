// Sources/ATen/Axis.swift

/// Represents a logical axis used to address tensor dimensions in a human-friendly manner.
public struct Axis: Hashable, Sendable, ExpressibleByIntegerLiteral {
  /// Identifies the underlying dimension reference strategy for an `Axis`.
  public enum Kind: Hashable, Sendable {
    case absolute(Int), last, penultimate,
         batch, channel, height, width, time, feature
  }
  public let kind: Kind

  /// Creates an axis that refers to the dimension at `value`, supporting negative indexing.
  public init(integerLiteral value: Int) {
    self.kind = .absolute(value)
  }

  // ✅ Make this initializer 'internal' and '@usableFromInline'
  @usableFromInline
  internal init(kind: Kind) {
    self.kind = kind
  }

  /// Convenience axis describing the final dimension in the tensor.
  @inlinable public static var last: Axis { .init(kind: .last) }
  /// Convenience axis describing the second-to-last dimension in the tensor.
  @inlinable public static var penultimate: Axis { .init(kind: .penultimate) }
  /// Convenience axis describing the leading (batch) dimension.
  @inlinable public static var batch: Axis { .init(kind: .batch) }
  /// Convenience axis describing the channel dimension for NCHW/NHWC layouts.
  @inlinable public static var channel: Axis { .init(kind: .channel) }
  /// Convenience axis describing the height dimension in 2D data.
  @inlinable public static var height: Axis { .init(kind: .height) }
  /// Convenience axis describing the width dimension in 2D data.
  @inlinable public static var width: Axis { .init(kind: .width) }
  /// Convenience axis describing the time dimension for sequence data.
  @inlinable public static var time: Axis { .init(kind: .time) }
  /// Convenience axis describing the feature dimension in tabular or sequential data.
  @inlinable public static var feature: Axis { .init(kind: .feature) }
}

public extension Axis {
  /// Resolves the logical axis into an absolute dimension index for the provided tensor `rank`.
  @inlinable func resolve(forRank r: Int) -> Int {
    switch kind {
    case .absolute(let d): return d < 0 ? d + r : d
    case .last: return r - 1
    case .penultimate: return r - 2
    case .batch:     precondition(r >= 1); return 0
    case .channel:   precondition(r >= 2); return 1
    case .height:    precondition(r >= 3); return r - 2
    case .width:     precondition(r >= 4); return r - 1
    case .time:      precondition(r >= 2); return 1
    case .feature:   precondition(r >= 2); return r - 1
    }
  }
}
