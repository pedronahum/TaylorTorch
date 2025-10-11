// Sources/Torch/Modules/Graph/Segment.swift
import _Differentiation

@inlinable
func _asLongIndices(_ xs: [Int]) -> Tensor {
  Tensor(array: xs.map(Int64.init), shape: [xs.count], dtype: .int64)
}

/// Sum rows of `data` into `numSegments` bins given 1-D `segmentIDs`.
/// - data: [M, C...]  - segmentIDs: [M] int - returns [numSegments, C...]
@inlinable
@differentiable(reverse, wrt: data)
public func segmentSum(
  data: Tensor,
  segmentIDs: [Int],
  numSegments: Int
) -> Tensor {
  precondition(data.rank >= 1, "segmentSum expects rank >= 1")
  let head = withoutDerivative(at: [numSegments])
  let tail = withoutDerivative(at: Array(data.shape.dropFirst()))
  let outShape = withoutDerivative(at: head + tail)
  let zeros = Tensor.zeros(shape: outShape, dtype: data.dtype ?? .float32, device: data.device)
  let idx = withoutDerivative(at: _asLongIndices(segmentIDs))
  // zeros.indexAdd along dim 0 at positions idx by rows in `data`
  return zeros.indexAdd(dim: 0, index: idx, source: data)  // uses your indexAdd
  //                                        ^^^^^^^^^^^^^  Tensor+AdvancedIndexing.swift
  //                                                                                 ⤷
  //                                                        (scatter-add semantics)
}

@inlinable
@derivative(of: segmentSum, wrt: data)
public func _vjpSegmentSum(
  data: Tensor,
  segmentIDs: [Int],
  numSegments: Int
) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
  let value = segmentSum(data: data, segmentIDs: segmentIDs, numSegments: numSegments)
  return (
    value,
    { v in
      v.indexSelect(dim: 0, indices: segmentIDs)
    }
  )
}

/// Mean = sum / count per segment.
@inlinable
public func segmentMean(data: Tensor, segmentIDs: [Int], numSegments: Int) -> Tensor {
  let sums = segmentSum(data: data, segmentIDs: segmentIDs, numSegments: numSegments)
  // compute counts per segment on host, then broadcast-divide
  var counts = [Int](repeating: 0, count: numSegments)
  for s in segmentIDs { counts[s] &+= 1 }
  // Avoid divide-by-zero: replace 0 with 1 (no edges into a node/graph).
  let safe = counts.map { max($0, 1) }
  let broadcastShape = withoutDerivative(
    at: [numSegments] + Array(repeating: 1, count: sums.rank - 1))
  let denom = Tensor(
    array: safe.map(Float.init),
    shape: broadcastShape,
    dtype: .float32)
  return sums.dividing(denom)  // elementwise / via Tensor+Math.swift
}

// Supporting helpers for per‑edge/node graph indices (needed for global pooling)

@inlinable
public func graphIndexOfNodes(nNode: [Int]) -> [Int] {
  var out: [Int] = []
  out.reserveCapacity(nNode.reduce(0, +))
  for (g, n) in nNode.enumerated() { out.append(contentsOf: repeatElement(g, count: n)) }
  return out
}
@inlinable
public func graphIndexOfEdges(nEdge: [Int]) -> [Int] {
  var out: [Int] = []
  out.reserveCapacity(nEdge.reduce(0, +))
  for (g, e) in nEdge.enumerated() { out.append(contentsOf: repeatElement(g, count: e)) }
  return out
}
