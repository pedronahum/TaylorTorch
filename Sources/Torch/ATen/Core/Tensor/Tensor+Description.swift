// Sources/ATen/Tensor+Description.swift
extension Tensor: CustomStringConvertible {
  /// Formatted string describing the tensor's dtype, shape, and (for small tensors) values.
  public var description: String {
    let header = "Tensor(dtype: \(dtype.map { "\($0)" } ?? "unknown"), shape: \(shape))"
    guard rank <= 2, count > 0, count <= 64, let dt = dtype else { return header }

    func render<T>(_ a: [T], rows: Int, cols: Int) -> String {
      let strings = a.map { "\($0)" }
      var widths = Array(repeating: 0, count: cols)
      for r in 0..<rows {
        for c in 0..<cols {
          let s = strings[r*cols + c]
          widths[c] = Swift.max(widths[c], s.count)
        }
      }
      var lines: [String] = []
      for r in 0..<rows {
        var colsStr: [String] = []
        for c in 0..<cols {
          let s = strings[r*cols + c]
          colsStr.append(s.padding(toLength: widths[c], withPad: " ", startingAt: 0))
        }
        lines.append("[ " + colsStr.joined(separator: " ") + " ]")
      }
      return lines.joined(separator: "\n")
    }

    switch (dt, rank) {
    case (.bool, 1):
      let v: [Bool] = toArray()
      return header + " =\n" + render(v, rows: 1, cols: v.count)
    case (.bool, 2):
      let rows = shape[0], cols = shape[1]
      let v: [Bool] = reshaped([rows, cols]).toArray()
      return header + " =\n" + render(v, rows: rows, cols: cols)

    case (.int8, 1):   fallthrough
    case (.int16, 1):  fallthrough
    case (.int32, 1):
      let v: [Int32] = toArray()
      return header + " =\n" + render(v, rows: 1, cols: v.count)
    case (.int8, 2):   fallthrough
    case (.int16, 2):  fallthrough
    case (.int32, 2):
      let rows = shape[0], cols = shape[1]
      let v: [Int32] = reshaped([rows, cols]).toArray()
      return header + " =\n" + render(v, rows: rows, cols: cols)

    case (.int64, 1):
      let v: [Int64] = toArray()
      return header + " =\n" + render(v, rows: 1, cols: v.count)
    case (.int64, 2):
      let rows = shape[0], cols = shape[1]
      let v: [Int64] = reshaped([rows, cols]).toArray()
      return header + " =\n" + render(v, rows: rows, cols: cols)

    case (.float32, 1):
      let v: [Float] = toArray()
      return header + " =\n" + render(v, rows: 1, cols: v.count)
    case (.float32, 2):
      let rows = shape[0], cols = shape[1]
      let v: [Float] = reshaped([rows, cols]).toArray()
      return header + " =\n" + render(v, rows: rows, cols: cols)

    case (.float64, 1):
      let v: [Double] = toArray()
      return header + " =\n" + render(v, rows: 1, cols: v.count)
    case (.float64, 2):
      let rows = shape[0], cols = shape[1]
      let v: [Double] = reshaped([rows, cols]).toArray()
      return header + " =\n" + render(v, rows: rows, cols: cols)

    default:
      return header
    }
  }
}
