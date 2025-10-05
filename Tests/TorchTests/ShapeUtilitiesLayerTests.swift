import Testing
import _Differentiation

@testable import Torch

// MARK: - Flatten

@Test("Flatten: default [B,*] -> [B,-1] equals manual reshaped")
func flatten_default_matches_manual() throws {
  // [B,C,H,W] = [2, 3, 2, 4]
  let x = Tensor.arange(Double(0), to: Double(2 * 3 * 2 * 4), step: 1, dtype: .float64)
    .reshaped([2, 3, 2, 4])

  let layer = Flatten()  // startDim: 1 .. endDim: -1
  let y = layer(x)

  let expected = x.reshaped([2, 3 * 2 * 4])
  #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))

  // Gradient: d/dx sum(flatten(x)) == ones shaped like x
  let (_, pb) = valueWithPullback(at: x) { t in layer(t).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))
  let ones = Tensor.ones(shape: x.shape, dtype: .float64)
  #expect(g.isClose(to: ones, rtol: 0, atol: 0, equalNan: false))
}

@Test("Flatten: range [s...e] merges only part of the shape")
func flatten_partial_range() throws {
  // Shape [B, T, D] = [2, 3, 4]  → Flatten(2...2) no-op; Flatten(1...2) → [2, 12]
  let x = Tensor.arange(Double(0), to: Double(2 * 3 * 4), step: 1, dtype: .float64).reshaped([
    2, 3, 4,
  ])

  let keep = Flatten(startDim: 2, endDim: 2)
  #expect(keep(x).isClose(to: x, rtol: 0, atol: 0, equalNan: false))

  let merge = Flatten(startDim: 1, endDim: 2)
  let expected = x.reshaped([2, 12])
  #expect(merge(x).isClose(to: expected, rtol: 0, atol: 0, equalNan: false))
}

// MARK: - Reshape

@Test("Reshape: single -1 dimension is inferred (element count preserved)")
func reshape_infers_negative_one() throws {
  // [2, 3, 4] -> [-1, 8] == [3, 8]
  let x = Tensor.arange(Double(0), to: Double(2 * 3 * 4), step: 1, dtype: .float64).reshaped([
    2, 3, 4,
  ])
  let layer = Reshape([-1, 8])
  let y = layer(x)
  let expected = x.reshaped([3, 8])
  #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))

  // Gradient sanity under sum
  let (_, pb) = valueWithPullback(at: x) { t in layer(t).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))
  #expect(
    g.isClose(to: Tensor.ones(shape: x.shape, dtype: .float64), rtol: 0, atol: 0, equalNan: false))
}

// MARK: - Permute

@Test("Permute: forward equals Tensor.permuted and inverse brings it back")
func permute_forward_and_inverse() throws {
  // [B,T,D] = [2, 3, 4] → [B,D,T] = axes [0,2,1]
  let x = Tensor.arange(Double(0), to: Double(2 * 3 * 4), step: 1, dtype: .float64).reshaped([
    2, 3, 4,
  ])
  let p = Permute([0, 2, 1])
  let y = p(x)
  let expected = x.permuted([0, 2, 1])
  #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))

  // Inverse: [0,2,1]^{-1} = [0,2,1] again for rank 3 swapping last two dims
  let inv = Permute([0, 2, 1])
  let back = inv(y)
  #expect(back.isClose(to: x, rtol: 0, atol: 0, equalNan: false))
}

@Test("Permute: gradient matches inverse-permuted upstream under linear functional")
func permute_pullback_matches_inverse() throws {
  // Use a linear functional: L = sum( permute(x) .* W ) → dL/dx = inversePermute(W)
  let x = Tensor.arange(Double(0), to: Double(2 * 3 * 4), step: 1, dtype: .float64).reshaped([
    2, 3, 4,
  ])
  let permAxes = [0, 2, 1]  // [B,T,D] -> [B,D,T]
  let p = Permute(permAxes)
  let W = Tensor.arange(Double(0), to: Double(2 * 4 * 3), step: 1, dtype: .float64).reshaped([
    2, 4, 3,
  ])

  let (_, pb) = valueWithPullback(at: x) { t in p(t).multiplying(W).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  // Expected gradient: inverse permute of W
  let inverseAxes = [0, 2, 1]  // self-inverse here
  let expected = W.permuted(inverseAxes)
  #expect(g.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))
}

// MARK: - Composition with builder (Embedding → Permute → Flatten → Linear)

@Test("Builder: Embedding -> Permute -> Flatten -> Linear composes and matches manual")
func embedding_permute_flatten_compose() throws {
  // This test assumes an Embedding layer exists in the module (as previously added).
  // Shapes: indices [B,T] -> embed [B,T,D] -> permute [B,D,T] -> flatten [B, D*T] -> Linear [B, O]
  let B = 2
  let T = 3
  let D = 4
  let O = 5
  let V = 10

  let indices = Tensor(array: [0, 1, 4, 2, 3, 1] as [Int64], shape: [B, T])
  var embed = Embedding(numEmbeddings: V, embeddingDim: D)

  // Make embedding deterministic for comparison.
  embed.weight = Tensor.arange(Double(0), to: Double(V * D), step: 1, dtype: .float64)
    .reshaped([V, D])

  let W = Tensor.arange(Double(0), to: Double(O * (D * T)), step: 1, dtype: .float64)
    .reshaped([O, D * T])
  let b = Tensor.zeros(shape: [O], dtype: .float64)
  var lin = Linear(weight: W, bias: b)

  let model = SequentialBlock {
    embed
    Permute([0, 2, 1])  // [B,T,D] -> [B,D,T]
    Flatten(startDim: 1)  // [B,D,T] -> [B, D*T]
    lin  // [B, D*T] -> [B, O]
  }

  // Forward
  let yModel = model(indices)

  // Manual baseline (same ops)
  let yManual = lin(Flatten(startDim: 1)(Permute([0, 2, 1])(embed(indices))))
  #expect(yModel.isClose(to: yManual, rtol: 0, atol: 0, equalNan: false))
}
