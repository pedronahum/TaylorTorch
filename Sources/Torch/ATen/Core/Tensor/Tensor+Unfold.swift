// Sources/ATen/Tensor+Unfold.swift
@preconcurrency import ATenCXX

public extension Tensor {
  /// Sliding window view along `dim` with the specified window `size` and stride `step`.
  /// Example: a tensor shaped `[N, C, L]` unfolded along `L` yields `[N, C, L_out, size]`.
  /// - Parameters:
  ///   - dim: Dimension over which to create sliding windows.
  ///   - size: Window length (must be > 0).
  ///   - step: Stride between consecutive windows (must be > 0).
  func unfolded(dim: Int, size: Int, step: Int) -> Tensor {
    Tensor(_impl.unfold(Int64(dim), Int64(size), Int64(step)))
  }
}
