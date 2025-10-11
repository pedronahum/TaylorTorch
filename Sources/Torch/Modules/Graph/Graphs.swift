// Sources/Torch/Graph/Graphs.swift
import _Differentiation

/// Minimal graphs-tuple in the spirit of Graph Nets / jraph:
/// - nodes:    [∑n_node, Cn]
/// - edges:    [∑n_edge, Ce]
/// - senders:  [∑n_edge] (int indices into `nodes`)
/// - receivers:[∑n_edge] (int indices into `nodes`)
/// - globals:  [G] or [B, G]
public struct Graphs: Differentiable, KeyPathIterable {
  // Differentiable fields (participate in AD).
  public var nodes: Tensor
  public var edges: Tensor
  public var globals: Tensor

  // Non-differentiable graph topology.
  @noDerivative public var senders: Tensor  // int64, shape [∑n_edge]
  @noDerivative public var receivers: Tensor  // int64, shape [∑n_edge]
  @noDerivative public var nNode: [Int]  // per-graph node counts
  @noDerivative public var nEdge: [Int]  // per-graph edge counts

  public init(
    nodes: Tensor,
    edges: Tensor,
    senders: Tensor,
    receivers: Tensor,
    globals: Tensor,
    nNode: [Int],
    nEdge: [Int]
  ) {
    self.nodes = nodes
    self.edges = edges
    self.globals = globals
    self.senders = senders
    self.receivers = receivers
    self.nNode = nNode
    self.nEdge = nEdge
  }

  /// Number of graphs in the batch.
  public var batchCount: Int {
    withoutDerivative(at: nNode.count)
  }

  /// Total number of nodes across the batch.
  public var nodeCount: Int {
    withoutDerivative(at: nodes.shape[0])
  }

  /// Total number of edges across the batch.
  public var edgeCount: Int {
    withoutDerivative(at: senders.shape[0])
  }

  // ---- Manual TangentVector: explicit zero/+/- to avoid synthesis pitfalls.
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,
    KeyPathIterable,
    VectorProtocol,
    PointwiseMultiplicative
  {
    public typealias VectorSpaceScalar = Float

    public var nodes: Tensor
    public var edges: Tensor
    public var globals: Tensor

    /// Use scalar zeros so broadcasting works even if a field is unused in a path.
    public init(
      nodes: Tensor = Tensor(0),
      edges: Tensor = Tensor(0),
      globals: Tensor = Tensor(0)
    ) {
      self.nodes = nodes
      self.edges = edges
      self.globals = globals
    }

    // AdditiveArithmetic — spelled out (no synthesis).
    public static var zero: Self { .init() }

    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        nodes: lhs.nodes + rhs.nodes,
        edges: lhs.edges + rhs.edges,
        globals: lhs.globals + rhs.globals
      )
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        nodes: lhs.nodes - rhs.nodes,
        edges: lhs.edges - rhs.edges,
        globals: lhs.globals - rhs.globals
      )
    }

    // Optional but helpful: explicit VJPs for +/-, mirrors what fixed MHA/LSTM.
    @derivative(of: +)
    public static func _vjpAdd(_ lhs: Self, _ rhs: Self)
      -> (value: Self, pullback: (Self) -> (Self, Self))
    {
      let y = lhs + rhs
      return (y, { v in (v, v) })
    }

    @derivative(of: -)
    public static func _vjpSub(_ lhs: Self, _ rhs: Self)
      -> (value: Self, pullback: (Self) -> (Self, Self))
    {
      let y = lhs - rhs
      return (y, { v in (v, .zero - v) })
    }
  }

  // Avoid synthesized move(by:).
  public mutating func move(by direction: TangentVector) {
    nodes += direction.nodes
    edges += direction.edges
    globals += direction.globals
  }
}
