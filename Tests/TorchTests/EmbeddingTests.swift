import Testing
import _Differentiation

@testable import Torch

@Test("Embedding forward: 1D indices pick matching rows")
func embeddingForward1D() throws {
  // weight: 4 x 3
  let w = Tensor(
    array: [
      10.0, 11.0, 12.0,  // id 0
      20.0, 21.0, 22.0,  // id 1
      30.0, 31.0, 32.0,  // id 2
      40.0, 41.0, 42.0,  // id 3
    ], shape: [4, 3])

  let emb = Embedding(weight: w)
  let ids = Tensor(array: [2, 0, 3], shape: [3]).to(dtype: .int64)

  let y = emb(ids)
  #expect(y.shape == [3, 3])
  #expect(
    y.toArray(as: Double.self) == [
      30.0, 31.0, 32.0,
      10.0, 11.0, 12.0,
      40.0, 41.0, 42.0,
    ])
}

@Test("Embedding forward: 2D indices -> shape [B, T, D]")
func embeddingForward2DShape() throws {
  let w = Tensor.arange(Double(0), to: Double(12), step: 1).reshaped([4, 3])
  let emb = Embedding(weight: w)
  let ids = Tensor(
    array: [
      0, 1,
      2, 3,
    ], shape: [2, 2]
  ).to(dtype: .int64)

  let y = emb(ids)
  #expect(y.shape == [2, 2, 3])
  // spot check a couple of rows
  #expect(
    y.select(dim: -2, index: 0).select(dim: -2, index: 0).isClose(
      to: w.select(dim: 0, index: 0), rtol: 1e-6, atol: 1e-6, equalNan: false))
  #expect(
    y.select(dim: -2, index: 1).select(dim: -2, index: 1).isClose(
      to: w.select(dim: 0, index: 3), rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Embedding backward: scatter-add accumulates per-id gradients")
func embeddingBackwardScatterAdds() throws {
  // Small table so we can eyeball counts: 5 ids, dim=2
  let w = Tensor.zeros(shape: [5, 2], dtype: .float32)
  var emb = Embedding(weight: w)

  // Indices with repeats: id 1 appears twice, id 3 once, id 0 once
  let ids = Tensor(array: [1, 3, 1, 0], shape: [4]).to(dtype: .int64)

  // y = emb(ids); loss = sum(y)
  let (y, pb) = valueWithPullback(at: emb) { e in e(ids) }
  let upstream = Tensor.ones(shape: y.shape, dtype: y.dtype!)
  let grad = pb(upstream).weight

  // Expected grad: for each id, count * ones(dim)
  var expected = Tensor.zeros(shape: [5, 2], dtype: grad.dtype!, device: grad.device)
  let onesRows = Tensor.ones(shape: [ids.shape[0], 2], dtype: grad.dtype!, device: grad.device)
  expected = expected.indexAdd(dim: 0, index: ids, source: onesRows)

  #expect(grad.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Embedding backward: paddingIndex row receives zero gradient")
func embeddingPaddingIndexZeroGrad() throws {
  // Make pad row obvious (non-zero), to verify only gradient is zeroed.
  let w = Tensor(
    array: [
      9.9, 9.9,  // id 0 (pad)
      1.0, 2.0,  // id 1
      3.0, 4.0,  // id 2
    ], shape: [3, 2], dtype: .float32)

  var emb = Embedding(weight: w, paddingIndex: 0)
  let ids = Tensor(array: [0, 1, 2, 0, 1], shape: [5]).to(dtype: .int64)

  let (y, pb) = valueWithPullback(at: emb) { e in e(ids) }
  let upstream = Tensor.ones(shape: y.shape, dtype: y.dtype!)
  let gw = pb(upstream).weight

  // Build the "naive" scatter grad, then zero-out the pad row.
  var expected = Tensor.zeros(shape: [3, 2], dtype: gw.dtype!, device: gw.device)
  let onesRows = Tensor.ones(shape: [ids.shape[0], 2], dtype: gw.dtype!, device: gw.device)
  expected = expected.indexAdd(dim: 0, index: ids, source: onesRows)
  let padIdx = Tensor(Int64(0), dtype: .int64, device: expected.device)
  let zerosRow = Tensor.zeros(shape: [2], dtype: expected.dtype!, device: expected.device)
  expected = expected.indexPut(indices: [padIdx], values: zerosRow, accumulate: false)

  #expect(gw.isClose(to: expected, rtol: 1e-6, atol: 1e-6, equalNan: false))
}
