

public extension Tensor {
  /// Reshape while inferring exactly one dimension marked with `-1`.
  /// Example: `t.reshaped(inferring: [2, -1])`
  func reshaped(inferring shape: [Int]) -> Tensor {
    precondition(shape.filter { $0 == -1 }.count == 1, "exactly one -1 is required")
    return reshaped(shape)
  }
}
