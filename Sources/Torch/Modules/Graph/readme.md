# Graph Modules

This directory hosts the graph-centric building blocks that power TaylorTorch’s
Graph Neural Network examples. The types here wrap host-side topology
information in Swift-friendly structures while leaning on ATen operations for
the heavy lifting.

## Files at a glance

| File | Highlights |
| --- | --- |
| `Graphs.swift` | Defines the `Graphs` container: node/edge/global feature tensors plus sender/receiver index tensors and per-graph counts. Conforms to `Differentiable` with a manually curated `TangentVector`. |
| `Batching.swift` | Utilities for concatenating multiple `Graphs` into a single batched graph (`batch(_:)`) while offsetting sender/receiver indices and preserving per-graph counts. |
| `GraphNetwork.swift` | Implements the Jraph/Graph Nets-style message passing primitive (`GraphNetwork`). Handles node/edge/global updates via user-provided `Layer`s, with custom pullbacks, host-index gathers, and segment reductions. |
| `Segment.swift` | Differentiable segment operations (`segmentSum`, `segmentMean`) and helpers (`graphIndexOfNodes`, `graphIndexOfEdges`) used by both batching and the graph network. |

## Implementation notes

- **Host indices** – sender/receiver lists live in standard Swift arrays and are
  converted to tensors when necessary. This mirrors common GNN pipelines and
  keeps batching straightforward.
- **Batch safety** – the `Graphs` struct stores per-graph node/edge counts, and
  `batch(_:)` offsets indices accordingly so segment reductions remain valid in
  the batched graph.
- **Custom VJPs** – `GraphNetwork` provides its own pullbacks to glue together
  edge/node/global updates; segment ops also specify derivatives, ensuring the
  autodiff pipeline remains well-defined.
- **Interoperability** – the message-passing layers expect standard TaylorTorch
  `Layer` implementations (e.g., `Sequential`, `Dense`), letting you reuse the
  same MLP blocks found in `Modules/Layers` when building phiE/phiV/phiU.

## Quick example

```swift
let g1 = Graphs(
  nodes: Tensor.randn(shape: [3, 4]),
  edges: Tensor.randn(shape: [2, 1]),
  senders: Tensor(array: [0, 1], shape: [2], dtype: .int64),
  receivers: Tensor(array: [1, 2], shape: [2], dtype: .int64),
  globals: Tensor.zeros(shape: [1, 1]),
  nNode: [3],
  nEdge: [2])

let g2 = Graphs(
  nodes: Tensor.randn(shape: [2, 4]),
  edges: Tensor.randn(shape: [1, 1]),
  senders: Tensor(array: [0], shape: [1], dtype: .int64),
  receivers: Tensor(array: [1], shape: [1], dtype: .int64),
  globals: Tensor.zeros(shape: [1, 1]),
  nNode: [2],
  nEdge: [1])

let batched = batch([g1, g2])

let gnn = GraphNetwork(
  phiE: Sequential {
    Dense(inputSize: 4 + 4 + 4 + 1, outputSize: 8)
    ReLU()
    Dense(inputSize: 8, outputSize: 8)
  },
  phiV: Sequential {
    Dense(inputSize: 8 + 4 + 1, outputSize: 8)
    ReLU()
    Dense(inputSize: 8, outputSize: 4)
  },
  phiU: Dense(inputSize: 8 + 4 + 1, outputSize: 1))

let updated = gnn(batched)
print(updated.nodes.shape)  // still [totalNodes, featureDim]
```

These primitives are what the Karate Club example builds on: batching a single
graph, wiring MLPs into a `GraphNetwork`, and optimising the resulting model via
TaylorTorch’s optimiser stack.
