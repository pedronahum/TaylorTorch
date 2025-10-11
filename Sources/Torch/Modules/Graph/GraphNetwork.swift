// Sources/Torch/Modules/Graph/GraphNetwork.swift
import _Differentiation

/// Concatenates tensors along their last axis.
@inlinable
func _concatLast(_ xs: [Tensor]) -> Tensor {
  precondition(!xs.isEmpty)
  let firstRank = withoutDerivative(at: xs[0].rank)
  let axis = withoutDerivative(at: _normalizeDimension(-1, rank: firstRank))
  return Tensor.cat(xs, dim: axis)
}

// ϕ^e, ϕ^v, ϕ^u are arbitrary Layers operating on dense feature tensors.
public struct GraphNetwork<EdgeModel: Layer, NodeModel: Layer, GlobalModel: Layer>: Layer
where
  EdgeModel.Input == Tensor, EdgeModel.Output == Tensor,
  NodeModel.Input == Tensor, NodeModel.Output == Tensor,
  GlobalModel.Input == Tensor, GlobalModel.Output == Tensor
{
  public var phiE: EdgeModel
  public var phiV: NodeModel
  public var phiU: GlobalModel

  @differentiable(reverse)
  public init(phiE: EdgeModel, phiV: NodeModel, phiU: GlobalModel) {
    self.phiE = phiE
    self.phiV = phiV
    self.phiU = phiU
  }

  public typealias Input = Graphs
  public typealias Output = Graphs

  @differentiable(reverse)
  public func callAsFunction(_ g: Graphs) -> Graphs {
    let B = withoutDerivative(at: g.batchCount)
    let N = withoutDerivative(at: g.nodeCount)
    let sendersHost = withoutDerivative(at: g.senders.toArray(as: Int.self))
    let receiversHost = withoutDerivative(at: g.receivers.toArray(as: Int.self))
    // --- Gather sender/receiver node and per-edge global features.
    let nSend = g.nodes.indexSelect(dim: 0, indices: sendersHost)  // [E, Cn] host indices ok
    let nRecv = g.nodes.indexSelect(dim: 0, indices: receiversHost)  // [E, Cn]
    let edgeGraphIDs = withoutDerivative(at: graphIndexOfEdges(nEdge: g.nEdge))  // [E]
    let uEdge = g.globals.indexSelect(dim: 0, indices: edgeGraphIDs)  // [E, Cg] host indices ok

    // --- Edge update: e' = ϕ^e([e, n_s, n_r, u_g])
    let eCat = _concatLast([g.edges, nSend, nRecv, uEdge])
    let ePrime = phiE(eCat)  // [E, Ce']

    // --- Aggregate to nodes: ρ^{e->v} over incoming edges (by receiver).
    let mV = segmentSum(data: ePrime, segmentIDs: receiversHost, numSegments: N)  // [N, Ce'] (sum)

    // --- Node update: v' = ϕ^v([ρ, v, u_g])
    let nodeGraphIDs = withoutDerivative(at: graphIndexOfNodes(nNode: g.nNode))  // [N]
    let uNode = g.globals.indexSelect(dim: 0, indices: nodeGraphIDs)  // [N, Cg]
    let vCat = _concatLast([mV, g.nodes, uNode])
    let vPrime = phiV(vCat)  // [N, Cn']

    // --- Aggregate to globals: ρ^{e->u}, ρ^{v->u} per graph.
    let eU = segmentSum(data: ePrime, segmentIDs: edgeGraphIDs, numSegments: B)  // [B, Ce']
    let vU = segmentSum(data: vPrime, segmentIDs: nodeGraphIDs, numSegments: B)  // [B, Cn']

    // --- Global update: u' = ϕ^u([ρ_e, ρ_v, u])
    let uCat = _concatLast([eU, vU, g.globals])
    let uPrime = phiU(uCat)  // [B, Cg']

    // Return a new graph with updated features, same structure.
    var out = g
    out.nodes = vPrime
    out.edges = ePrime
    out.globals = uPrime
    return out
  }

  // Manual TangentVector avoids the AD pitfalls you’ve hit elsewhere.
  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float
    public var phiE: EdgeModel.TangentVector
    public var phiV: NodeModel.TangentVector
    public var phiU: GlobalModel.TangentVector
    public static var zero: Self { .init(phiE: .zero, phiV: .zero, phiU: .zero) }
    public static func + (l: Self, r: Self) -> Self {
      .init(phiE: l.phiE + r.phiE, phiV: l.phiV + r.phiV, phiU: l.phiU + r.phiU)
    }
    public static func - (l: Self, r: Self) -> Self {
      .init(phiE: l.phiE - r.phiE, phiV: l.phiV - r.phiV, phiU: l.phiU - r.phiU)
    }
  }

  public mutating func move(by d: TangentVector) {
    phiE.move(by: d.phiE)
    phiV.move(by: d.phiV)
    phiU.move(by: d.phiU)
  }
}
