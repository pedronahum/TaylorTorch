import Foundation

/// Global toggle for runtime shape checking.
/// - Debug builds: enabled by default
/// - Release builds: enable by setting `TORCH_SHAPE_CHECKS=1` (or `true`)
@usableFromInline
enum _TTShapeCheck {
  static public let enabled: Bool = {
    // 1) Explicit env override
    if let v = ProcessInfo.processInfo.environment["TORCH_SHAPE_CHECKS"] {
      let on = v == "1" || v.lowercased() == "true" || v.lowercased() == "yes"
      return on
    }
    // 2) Default: enabled in Debug, disabled in Release
    #if DEBUG
      return true
    #else
      return false
    #endif
  }()
}

/// Inlined precondition that only fires when shape checking is enabled.
/// Keep this tiny so the optimizer removes it when disabled.
@inlinable @inline(__always)
func _ttPrecondition(
  _ condition: @autoclosure () -> Bool,
  _ message: @autoclosure () -> String,
  file: StaticString = #fileID, line: UInt = #line
) {
  if _TTShapeCheck.enabled {
    precondition(condition(), message(), file: file, line: line)
  }
}
