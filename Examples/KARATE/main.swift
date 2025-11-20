import Foundation
import Torch
import _Differentiation

// MARK: - Tiny, reproducible Karate Club node classification
//
// This example trains a two-layer message-passing model on Zachary’s Karate Club (34 nodes).
// It uses one-hot node features, undirected edges (duplicated both directions),
// and the canonical 2-class split: Mr. Hi (0) vs Officer (1).

// ---------- Data: Karate Club (34 nodes)

// 78 undirected edges.
// Indices are 0-based. This is a compact, standard listing used in many examples.
private let UNDIRECTED_EDGES: [(Int, Int)] = [
  (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12),
  (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
  (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
  (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
  (3, 7), (3, 12), (3, 13),
  (4, 6), (4, 10),
  (5, 6), (5, 10), (5, 16),
  (6, 16),
  (8, 30), (8, 32), (8, 33),
  (9, 33),
  (13, 33),
  (14, 32), (14, 33),
  (15, 32), (15, 33),
  (18, 32), (18, 33),
  (19, 33),
  (20, 32), (20, 33),
  (22, 32), (22, 33),
  (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
  (24, 25), (24, 27), (24, 31),
  (25, 31),
  (26, 29), (26, 33),
  (27, 33),
  (28, 31), (28, 33),
  (29, 32), (29, 33),
  (30, 32), (30, 33),
  (31, 32), (31, 33),
  (32, 33),
]

// Canonical “Mr. Hi” club membership (label 0); complement are “Officer” (label 1).
private let MR_HI: Set<Int> = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]

// Train split: a small, balanced set; test = others.
private let TRAIN_HI: [Int] = [0, 1, 2, 3]  // 4 from Mr. Hi
private let TRAIN_OFFICER: [Int] = [33, 32, 31, 30]  // 4 from Officer
private let TRAIN_IDX: [Int] = TRAIN_HI + TRAIN_OFFICER

// ---------- Graph construction helpers

private func oneHotFeatures(nodeCount: Int) -> Tensor {
  var buf = [Float](repeating: 0, count: nodeCount * nodeCount)
  for i in 0..<nodeCount { buf[i * nodeCount + i] = 1 }
  return Tensor(array: buf, shape: [nodeCount, nodeCount], dtype: .float32)
}

private func symmetrizedEdges(_ undirected: [(Int, Int)]) -> ([Int], [Int]) {
  var s: [Int] = []
  var r: [Int] = []
  s.reserveCapacity(undirected.count * 2)
  r.reserveCapacity(undirected.count * 2)
  for (u, v) in undirected {
    s.append(u)
    r.append(v)
    s.append(v)
    r.append(u)
  }
  return (s, r)
}

private func buildKarateGraphs() -> (g: Graphs, labels: Tensor, trainIdx: [Int], testIdx: [Int]) {
  let N = 34
  let (senders, receivers) = symmetrizedEdges(UNDIRECTED_EDGES)
  let E = senders.count

  // Node features: one‑hot [34, 34] for easy separability after message passing.
  let x = oneHotFeatures(nodeCount: N)  // [N, Cn], Cn = 34

  // Edge features: scalar zeros [E, 1] (we don’t use them, but GraphNetwork concatenates them).
  let e = Tensor.zeros(shape: [E, 1], dtype: .float32)

  // Globals: a single scalar per-graph [B, Cg] with B=1, Cg=1.
  let u = Tensor.zeros(shape: [1, 1], dtype: .float32)

  // Graph counts (one graph): nNode = [34], nEdge = [E].
  let nNode = [N]
  let nEdge = [E]

  // Tensors for edges’ endpoints (int64).
  let sTensor = Tensor(array: senders.map(Int64.init), shape: [E], dtype: .int64)
  let rTensor = Tensor(array: receivers.map(Int64.init), shape: [E], dtype: .int64)

  // Labels: 0 for Mr. Hi, 1 for Officer.
  let yHost: [Int64] = (0..<N).map { MR_HI.contains($0) ? 0 : 1 }
  let y = Tensor(array: yHost, shape: [N], dtype: .int64)

  // Train/Test split
  var all = Set(0..<N)
  TRAIN_IDX.forEach { all.remove($0) }
  let testIdx = Array(all).sorted()

  // Build Graphs container (shape conventions follow Graphs.swift).
  let g = Graphs(
    nodes: x,
    edges: e,
    senders: sTensor,
    receivers: rTensor,
    globals: u,
    nNode: nNode,
    nEdge: nEdge
  )

  return (g, y, TRAIN_IDX, testIdx)
}

// ---------- Model: two-stage message passing into 2-class logits per node.
//
// We use your GraphNetwork layer (phiE/phiV/phiU). Shapes follow GraphNetwork.swift:
//  e-cat = [e, n_s, n_r, u_e], v-cat = [m^v, n, u^v], u-cat = [m^u, n^u, u].
//  Aggregations use segmentSum/segmentMean with host indices (senders/receivers, graphIDs).
//                                                             ^ indexSelect from host arrays.  :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

struct KarateGNN: Layer {
  typealias Input = Graphs
  typealias Output = Tensor  // node logits [N, 2]

  // Convenience alias for the 2-layer MLP used in phiE/phiV blocks.
  typealias MLP2 = Sequential<Chain<Chain<Dense, ReLU>, Dense>>

  // First message-passing stage
  var gnn1: GraphNetwork<MLP2, MLP2, Dense>

  // Second stage maps hiddenN -> logits(2)
  var gnn2: GraphNetwork<MLP2, MLP2, Dense>

  init(
    nodeDim: Int = 34, edgeInDim: Int = 1, gInDim: Int = 1,
    hiddenE: Int = 16, hiddenN: Int = 16, gDim: Int = 1
  ) {
    // Stage 1 input dims
    let eCat1 = edgeInDim + nodeDim + nodeDim + gInDim  // 1 + 34 + 34 + 1 = 70
    let vCat1 = hiddenE + nodeDim + gInDim  // 16 + 34 + 1 = 51
    let uCat1 = hiddenE + hiddenN + gInDim  // 16 + 16 + 1 = 33

    gnn1 = GraphNetwork(
      phiE: Sequential {
        Dense(inputSize: eCat1, outputSize: hiddenE)
        ReLU()
        Dense(inputSize: hiddenE, outputSize: hiddenE)
      },
      phiV: Sequential {
        Dense(inputSize: vCat1, outputSize: hiddenN)
        ReLU()
        Dense(inputSize: hiddenN, outputSize: hiddenN)
      },
      phiU: Dense(inputSize: uCat1, outputSize: gDim)  // keep globals small/constant
    )

    // Stage 2 dims (note node+edge dims now hiddenE / hiddenN; globals kept at gDim)
    let eCat2 = hiddenE + hiddenN + hiddenN + gDim  // 16 + 16 + 16 + 1 = 49
    let vCat2 = hiddenE + hiddenN + gDim  // 16 + 16 + 1 = 33
    let uCat2 = hiddenE + /* logits */ 2 + gDim  // 16 + 2 + 1 = 19

    gnn2 = GraphNetwork(
      phiE: Sequential {
        Dense(inputSize: eCat2, outputSize: hiddenE)
        ReLU()
        Dense(inputSize: hiddenE, outputSize: hiddenE)
      },
      // Map nodes to 2-class logits at the end of stage 2
      phiV: Sequential {
        Dense(inputSize: vCat2, outputSize: hiddenN)
        ReLU()
        Dense(inputSize: hiddenN, outputSize: 2)  // logits per node
      },
      phiU: Dense(inputSize: uCat2, outputSize: gDim)
    )
  }

  @differentiable(reverse)
  func callAsFunction(_ x: Graphs) -> Tensor {
    let h1 = gnn1(x)
    let h2 = gnn2(h1)
    return h2.nodes  // [N, 2]
  }
}

// ---------- Utilities: masked CE and accuracy on index sets

@inline(__always)
func logitsAtRows(_ logits: Tensor, _ rows: [Int]) -> Tensor {
  logits.indexSelect(dim: 0, indices: rows)  // host indices → device rows. :contentReference[oaicite:10]{index=10}
}

@inline(__always)
func labelsAtRows(_ y: Tensor, _ rows: [Int]) -> Tensor {
  return y.indexSelect(dim: 0, indices: rows)
}

func accuracy(_ logits: Tensor, _ y: Tensor, rows: [Int]) -> Double {
  let sub = logitsAtRows(logits, rows)
  let preds = sub.argmax(dim: 1)  // indices of max logit per row. :contentReference[oaicite:11]{index=11}
  let tgt = labelsAtRows(y, rows)
  let correctMask = (preds .== tgt).to(dtype: .int64)
  let correct = correctMask.sum().toArray(as: Int64.self)[0]
  return Double(correct) / Double(rows.count)
}

// ---------- Train

@main
struct KarateExample {
  static func main() {
    // Build data
    let (g, y, trainIdx, testIdx) = buildKarateGraphs()

    // Model & optimizer
    // Note: Using SGD instead of Adam due to keypath issues with complex models on Linux
    // See KNOWN_ISSUES.md for details
    var model = KarateGNN()
    var opt = SGD(for: model, learningRate: 0.001, momentum: 0.9)

    print("Karate Club • nodes: \(g.nNode[0]), edges: \(g.nEdge[0])")
    print("Train: \(trainIdx.count) nodes • Test: \(testIdx.count) nodes")

    // Full-batch training (small graph)
    let epochs = 400
    withLearningPhase(.training) {
      for epoch in 1...epochs {
        let (loss, pullback) = valueWithPullback(at: model) { current -> Tensor in
          let logits = current(g)  // [N, 2]
          let sub = logitsAtRows(logits, trainIdx)  // [Ntrain, 2]
          let yTrain = labelsAtRows(y, trainIdx)  // [Ntrain]
          return softmaxCrossEntropy(logits: sub, labels: yTrain)
        }

        let grad = pullback(Tensor(1.0, dtype: .float32))
        opt.update(&model, along: grad)

        if epoch % 25 == 0 || epoch == 1 {
          let logits = model(g)
          let trainAcc = accuracy(logits, y, rows: trainIdx)
          let testAcc = accuracy(logits, y, rows: testIdx)
          let lossValue = Double(loss.toArray(as: Float.self)[0])
          print(
            String(
              format: "epoch %3d • loss %.4f • train acc %.3f • test acc %.3f",
              epoch, lossValue, trainAcc, testAcc))
        }
      }
    }

    // Show a few predicted labels at the end
    let finalLogits = model(g)
    let finalPreds = finalLogits.argmax(dim: 1)
    print("Pred (0=MrHi, 1=Officer):", finalPreds.toArray(as: Int64.self))
  }
}
