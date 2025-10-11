// Sources/Torch/Modules/Graph/Batching.swift

/// Concatenate a list of graphs into one batched `Graphs`.
public func batch(_ graphs: [Graphs]) -> Graphs {
  precondition(!graphs.isEmpty)
  // 1) Concatenate features.
  let nodes = Tensor.cat(graphs.map { $0.nodes }, dim: 0)
  let edges = Tensor.cat(graphs.map { $0.edges }, dim: 0)
  let globals = Tensor.cat(graphs.map { $0.globals }, dim: 0)

  // 2) Offset senders/receivers by cumulative node counts.
  var sendersAll: [Int64] = []
  var receiversAll: [Int64] = []
  var nNodeAll: [Int] = []
  var nEdgeAll: [Int] = []
  var nodeOffset: Int64 = 0
  for g in graphs {
    let sendersHost = g.senders.toArray(as: Int64.self)
    let receiversHost = g.receivers.toArray(as: Int64.self)
    sendersAll.append(contentsOf: sendersHost.map { $0 + nodeOffset })
    receiversAll.append(contentsOf: receiversHost.map { $0 + nodeOffset })
    let nodeCount = g.nodes.shape[0]
    nNodeAll.append(nodeCount)
    nEdgeAll.append(g.edges.shape[0])
    nodeOffset += Int64(nodeCount)
  }

  let senders = Tensor(array: sendersAll, shape: [sendersAll.count], dtype: .int64)
  let receivers = Tensor(array: receiversAll, shape: [receiversAll.count], dtype: .int64)

  return Graphs(
    nodes: nodes, edges: edges,
    senders: senders, receivers: receivers,
    globals: globals,
    nNode: nNodeAll, nEdge: nEdgeAll)
}
